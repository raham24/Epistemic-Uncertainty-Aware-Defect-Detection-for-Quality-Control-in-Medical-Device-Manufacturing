"""Cascaded multi-head MLP for the v2 (joint-mechanism) dataset.

This is OUR chained model, distinct from the paper baseline in mlp_paper.py /
mlp_common.py. It consumes data/smt_synthetic_v2.csv (from generator_v2.py) and
predicts a 3-head RCA chain, where each head sees what the previous heads
predicted:

  x --trunk--> h
                |- head1 defect : Linear(h)                  -> 3-way softmax (CE)
                |- head2 mech   : Linear(h (+) p_defect)     -> 9-way softmax (CE)
                |- head3 risk   : Linear(h (+) p_defect (+) p_mech) -> 6 sigmoids (BCE)

Differences from the paper baseline (mlp_common.MultiHeadMLP):
  * TRUE CASCADE, not parallel heads. Each downstream head receives the trunk
    latent concatenated with the SOFTMAX distributions of the upstream heads
    (soft, so it stays differentiable and downstream losses also sharpen the
    upstream heads). Trained end-to-end on the model's own predictions -- no
    teacher forcing, so there is no train/inference mismatch.
  * head2 is ONE 9-way softmax over the JOINT mechanism label mechanism_joint
    ("<printing>__<reflow>"), the prof's Cartesian product, instead of two
    independent per-stage heads. Only 7 of the 9 classes occur.
  * head3 (per-parameter risk) is the GLOBAL, two-sided graded risk risk_<param>
    -- a function of how far each parameter drifted, NOT of the mechanism label.
    So it always reports a per-parameter risk, even when head2 predicts
    no_mechanism: a clean board sits at the ~p_L floor, a sub-threshold-drifting
    board shows elevated risk on the drifting parameter. The cascade feeds head2's
    prediction to head3 as a HINT, but the target never suppresses, so head3 never
    learns to gate itself to zero.
  * head4 (the per-mechanism gated risk risk_mech_<m>_<param>) is DROPPED for now.
    Those 30 columns stay in the CSV, simply unused here.

The loss is swappable from the CLI via --loss, mirroring v1/mlp.py:
  --loss cascade    (default) CE on defect + mechanism, BCE on risk -- paper-style.
  --loss abstention the selective-classification ("gambler") term -log(o*p_y + r)
                    on the defect AND 9-class mechanism heads (each widened by one
                    abstain column); the sigmoid/BCE risk head is unchanged. --o
                    sets the payoff. --loss cascade is byte-identical to the plain
                    model, so `python mlp_v2.py` is unchanged.

Run: python mlp_v2.py                         # train + eval on the v2 csv (cascade)
     python mlp_v2.py --loss abstention --o 2.0
     python mlp_v2.py --epochs 60 --device mps
     python mlp_v2.py --load results/mlp_v2_model.pt   # eval a checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# v1's shared maths (generator.py) and the paper engine (mlp_common.py) now live
# in v1/. Put that folder on the path so the bare imports below resolve to the
# v1/ copies regardless of the current working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1"))

from generator import defect_names, load_spec, param_ids, seed_everything
from generator_v2 import JOINT_SEP
# schema-agnostic helpers + the gambler term reused straight from the paper engine
from mlp_common import (BATCH, DROPOUT, EPOCHS, HIDDEN, LAMBDA_D, LAMBDA_M,
                        LAMBDA_R, LR, PATIENCE, PAPER_REF, _class_weights,
                        _clone, _print_split_summary, _standardize, gambler_term,
                        split_summary)

MODEL_VERSION = "mlp-v2-cascade-v0.1"


# Vocab

def joint_mech_vocab(spec: dict) -> list[str]:
    """Algorithm — The 9 joint mechanism classes, in a fixed Cartesian order.

    Input: spec (its two ordered per-stage mechanism lists).
    Return: ["<printing>__<reflow>", ...] for every printing x reflow pair, in
            printing-major spec order -- the label set for head2 (9 classes).
    """

    stages = list(spec["mechanisms"].keys())
    printing = spec["mechanisms"][stages[0]]
    reflow = spec["mechanisms"][stages[1]]
    return [f"{p}{JOINT_SEP}{r}" for p in printing for r in reflow]


# Data

def encode_v2(df: pd.DataFrame, spec: dict) -> dict:
    """Algorithm — Turn the v2 dataframe into model arrays for the 3-head cascade.

    Input: the generated v2 dataframe and the spec.
    Return: features X, integer defect/joint-mechanism targets, float risk
            targets, and the label vocabularies (numpy throughout).
    """

    ids = param_ids(spec)
    names = defect_names(spec)
    vocab = joint_mech_vocab(spec)

    # features: the 6 raw parameter values
    X = df[ids].to_numpy(np.float32)

    # head1 target: defect class name -> index (order fixed by the spec)
    d_idx = {n: i for i, n in enumerate(names)}
    y_def = df["defect_label"].map(d_idx).to_numpy(np.int64)

    # head2 target: the joint mechanism label -> index in the 9-class vocab.
    # A label outside the vocab maps to NaN -> fail loudly (catches a bad join).
    m_idx = {m: i for i, m in enumerate(vocab)}
    mapped = df["mechanism_joint"].map(m_idx)
    if mapped.isna().any():
        bad = sorted(df.loc[mapped.isna(), "mechanism_joint"].unique())
        raise ValueError(f"mechanism_joint has values outside the 9-class vocab "
                         f"{vocab}: {bad}")
    y_mech = mapped.to_numpy(np.int64)

    # head3 target: the global per-parameter graded risk, already in (0,1)
    y_risk = df[[f"risk_{p}" for p in ids]].to_numpy(np.float32)

    return {"X": X, "y_def": y_def, "y_mech": y_mech, "y_risk": y_risk,
            "names": names, "mech_vocab": vocab, "ids": ids}


def _loader(enc: dict, idx: np.ndarray, batch: int, shuffle: bool,
            generator: torch.Generator | None = None) -> DataLoader:
    """Wrap one split's tensors (X, y_def, y_mech, y_risk) in a DataLoader."""

    ds = TensorDataset(torch.from_numpy(enc["X"][idx]),
                       torch.from_numpy(enc["y_def"][idx]),
                       torch.from_numpy(enc["y_mech"][idx]),
                       torch.from_numpy(enc["y_risk"][idx]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      generator=generator)


# Model

class CascadeMLP(nn.Module):
    """Shared trunk h = f(x), then a 3-head cascade: defect -> mechanism -> risk.

    Each downstream head reads the trunk latent concatenated with the upstream
    heads' softmax distributions (cumulatively threaded), so the chain is
    feedforward: head2 sees p(defect), head3 sees p(defect) and p(mechanism).
    The risk head is P independent sigmoids (one per parameter), not a softmax.
    """

    def __init__(self, n_features: int, n_defects: int, n_mech: int,
                 n_params: int, hidden: tuple[int, ...] = HIDDEN,
                 dropout: float = DROPOUT, abstain: bool = False) -> None:
        super().__init__()

        self.abstain = abstain
        self.n_defects, self.n_mech, self.n_params = n_defects, n_mech, n_params

        # shared trunk: Linear -> ReLU -> Dropout, stacked
        layers: list[nn.Module] = []
        d = n_features
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(dropout)]
            d = w
        self.trunk = nn.Sequential(*layers)

        # abstain widens BOTH classification heads by one "abstain" output (so the
        # gambler loss has a column for reject mass). The widened distributions --
        # reject mass included -- are what the downstream heads see; the risk head
        # is never widened. abstain=False is byte-identical to the plain cascade.
        extra = 1 if abstain else 0
        self.defect_out = n_defects + extra
        self.mech_out = n_mech + extra

        # cascade heads: each takes h plus the upstream soft predictions
        self.defect_head = nn.Linear(d, self.defect_out)
        self.mech_head = nn.Linear(d + self.defect_out, self.mech_out)
        self.risk_head = nn.Linear(d + self.defect_out + self.mech_out, n_params)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # shared latent
        h = self.trunk(x)

        # head1 -> its softmax feeds head2
        defect = self.defect_head(h)
        p_def = F.softmax(defect, dim=1)

        # head2 sees h (+) p(defect); its softmax feeds head3
        mech = self.mech_head(torch.cat([h, p_def], dim=1))
        p_mech = F.softmax(mech, dim=1)

        # head3 sees h (+) p(defect) (+) p(mechanism)
        risk = self.risk_head(torch.cat([h, p_def, p_mech], dim=1))

        return {"defect": defect, "mech": mech, "risk": risk,
                "defect_prob": p_def, "mech_prob": p_mech}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-head probabilities + argmax picks (the evidence-layer payload)."""

        self.eval()
        out = self.forward(x)
        risk_prob = torch.sigmoid(out["risk"])               # per-parameter risk

        # abstain: each classification head is one column wider; the LAST column is
        # the abstain prob. Pick the class over the REAL columns only (never the
        # abstain column) and expose the abstain prob alongside it.
        if self.abstain:
            m, k = self.n_defects, self.n_mech
            d_full, m_full = out["defect_prob"], out["mech_prob"]
            d_real, m_real = d_full[:, :m], m_full[:, :k]
            return {
                "defect_prob": d_real,
                "defect_argmax": d_real.argmax(1),           # over m real classes
                "abstain_prob": d_full[:, m],                # P(abstain) = r(x)
                "mech_prob": m_real,
                "mech_argmax": m_real.argmax(1),             # over k real classes
                "mech_abstain_prob": m_full[:, k],
                "risk_prob": risk_prob,
            }

        # default (no abstain) path -- unchanged
        return {
            "defect_prob": out["defect_prob"],
            "defect_argmax": out["defect_prob"].argmax(1),  # chain root
            "mech_prob": out["mech_prob"],
            "mech_argmax": out["mech_prob"].argmax(1),       # joint mechanism
            "risk_prob": risk_prob,
        }


class CascadeLoss(nn.Module):
    """Composite chain loss: lam_d*CE(defect) + lam_m*CE(mech_9) + lam_r*BCE(risk).

    CE on the two classification heads; BCE-with-logits (soft targets) on the
    per-parameter risk head, mean-reduced over batch and the P parameters so it
    sits on a per-head scale comparable to the CE terms.

    Each per-head TERM is its own method so a subclass can override just the
    classification terms (e.g. an abstention/gambler defect+mech term) without
    rewriting the combine. Subclasses needing the +1-wide abstain heads flip
    ABSTAIN to True (train() then builds the model with abstain=True).
    """

    ABSTAIN = False

    def __init__(self, class_weight: torch.Tensor | None = None,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        self.lam_d, self.lam_m, self.lam_r = lam_d, lam_m, lam_r

    # per-head TERMS (override the classification ones in a subclass)
    def defect_term(self, out: dict, y_def: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["defect"], y_def, weight=self.class_weight)

    def mech_term(self, out: dict, y_mech: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["mech"], y_mech)

    def risk_term(self, out: dict, y_risk: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(out["risk"], y_risk)

    # COMBINE (subclasses override the terms above, not this)
    def forward(self, out: dict, y_def: torch.Tensor, y_mech: torch.Tensor,
                y_risk: torch.Tensor) -> torch.Tensor:
        return (self.lam_d * self.defect_term(out, y_def)
                + self.lam_m * self.mech_term(out, y_mech)
                + self.lam_r * self.risk_term(out, y_risk))


class CascadeAbstentionLoss(CascadeLoss):
    """Our variant: the selective-classification ("gambler") term on BOTH the
    defect and the 9-class mechanism heads, the sigmoid/BCE risk head unchanged.

    Each classification head is one column wider (the abstain output); the term is
    -log(o*p_y + r) where r is the abstain prob and o the payoff (the --o flag).
    r=0 reduces it to cross-entropy + const, so larger o => predict more / abstain
    less. class_weight is accepted for train()-API compatibility but IGNORED -- the
    abstain mechanism, not reweighting, handles the imbalance.
    """

    ABSTAIN = True

    def __init__(self, class_weight: torch.Tensor | None = None, o: float = 2.0,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__(class_weight=None, lam_d=lam_d, lam_m=lam_m, lam_r=lam_r)
        self.o = o

    def defect_term(self, out: dict, y_def: torch.Tensor) -> torch.Tensor:
        return gambler_term(out["defect"], y_def, self.o)

    def mech_term(self, out: dict, y_mech: torch.Tensor) -> torch.Tensor:
        return gambler_term(out["mech"], y_mech, self.o)


# loss registry exposed via --loss; cascade is the paper-style default
LOSSES = {"cascade": CascadeLoss, "abstention": CascadeAbstentionLoss}


# Train / evaluate

def _epoch_loss(model: CascadeMLP, loss_fn: CascadeLoss, dl: DataLoader,
                device: str) -> float:
    """Average loss over a loader (no gradients)."""

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yd, ym, yr in dl:
            xb, yd, ym, yr = xb.to(device), yd.to(device), ym.to(device), yr.to(device)
            total += float(loss_fn(model(xb), yd, ym, yr)) * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train(spec: dict, df: pd.DataFrame, device: str = "cpu",
          epochs: int = EPOCHS, class_weight_mode: str = "none",
          loss_cls: type[CascadeLoss] = CascadeLoss,
          hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT,
          seed: int = 0, o: float = 2.0, lr: float = LR,
          batch: int = BATCH) -> tuple[CascadeMLP, dict]:
    """Algorithm — Train the cascade with early stopping on validation loss.

    Input: spec, the v2 dataframe, device, epoch budget, defect-head weighting
           mode, the loss class, trunk hidden/dropout, a seed for the shuffle, the
           abstention payoff o, lr, batch size.
    Return: the best model and the encoded-data bundle (with stats + val curve).
    """

    enc = encode_v2(df, spec)

    # leak-free batch-grouped split indices come from the split column
    sp = df["split"].to_numpy()
    tr, va = np.where(sp == "train")[0], np.where(sp == "val")[0]

    # standardize on TRAIN stats only; store them so eval reuses them exactly
    stdz, enc["mu"], enc["sd"] = _standardize(enc["X"][tr], enc["X"][va])
    enc["X"][tr], enc["X"][va] = stdz

    # an abstention loss needs the +1-wide classification heads
    abstain = getattr(loss_cls, "ABSTAIN", False)

    # build the cascade with sizes from the spec
    model = CascadeMLP(
        n_features=enc["X"].shape[1], n_defects=len(enc["names"]),
        n_mech=len(enc["mech_vocab"]), n_params=enc["y_risk"].shape[1],
        hidden=hidden, dropout=dropout, abstain=abstain).to(device)

    # optional defect-head class weighting (the ~88% no_defect imbalance) + Adam.
    # pass o only to an abstention loss (the base loss has no o argument)
    cw = _class_weights(enc["y_def"][tr], len(enc["names"]), class_weight_mode)
    loss_kwargs = {"o": o} if abstain else {}
    loss_fn = loss_cls(class_weight=cw.to(device) if cw is not None else None,
                       **loss_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # seed the shuffle explicitly for run-by-its-own-seed reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    tr_dl = _loader(enc, tr, batch, shuffle=True, generator=g)
    va_dl = _loader(enc, va, batch, shuffle=False)

    # early stopping: keep the lowest-val-loss weights
    best_val, best_state, waited, val_hist = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        for xb, yd, ym, yr in tr_dl:
            xb, yd, ym, yr = xb.to(device), yd.to(device), ym.to(device), yr.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yd, ym, yr)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep}")
            loss.backward()
            opt.step()

        val = _epoch_loss(model, loss_fn, va_dl, device)
        val_hist.append(val)
        print(f"  epoch {ep:2d}  val_loss {val:.4f}")
        if val < best_val - 1e-4:
            best_val, best_state, waited = val, _clone(model), 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    enc["val_loss_history"] = val_hist
    enc["best_epoch"] = int(np.argmin(val_hist)) if val_hist else 0
    return model, enc


def evaluate(model: CascadeMLP, enc: dict, df: pd.DataFrame, spec: dict,
             device: str = "cpu", split: str = "test",
             model_version: str = MODEL_VERSION) -> dict:
    """Algorithm — Per-split metrics for the cascade.

    Input: trained model, encoded data, dataframe, spec, device, split.
    Return: defect accuracy/F1 (+ Bayes ceiling), joint-mechanism accuracy with a
            per-stage decomposition, and per-parameter risk MAE.
    """

    names, vocab, ids = enc["names"], enc["mech_vocab"], enc["ids"]
    te = np.where(df["split"].to_numpy() == split)[0]

    # pull RAW features for these rows and standardize with the STORED train stats
    Xraw = df.iloc[te][ids].to_numpy(np.float32)
    Xte = ((Xraw - enc["mu"]) / enc["sd"]).astype(np.float32)
    pred = model.predict(torch.from_numpy(Xte).to(device))

    # head1 defect: accuracy + weighted/macro F1
    y_def = enc["y_def"][te]
    d_pred = pred["defect_argmax"].cpu().numpy()
    defect = {
        "accuracy": float(accuracy_score(y_def, d_pred)),
        "weighted_f1": float(f1_score(y_def, d_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_def, d_pred, average="macro", zero_division=0)),
    }

    # Bayes-optimal ceiling: argmax of the KNOWN defect posterior on the same rows
    post = df.iloc[te][[f"p_{n}" for n in names]].to_numpy()
    bayes_acc = float(accuracy_score(y_def, post.argmax(1)))

    # head2 mechanism: joint 9-class accuracy + per-stage decomposition
    y_mech = enc["y_mech"][te]
    m_pred = pred["mech_argmax"].cpu().numpy()
    joint_acc = float(accuracy_score(y_mech, m_pred))
    # split each predicted/true joint label back into (printing, reflow)
    pred_pairs = np.array([vocab[i].split(JOINT_SEP) for i in m_pred])
    true_pairs = np.array([vocab[i].split(JOINT_SEP) for i in y_mech])
    stages = list(spec["mechanisms"].keys())
    mech = {
        "joint_accuracy": joint_acc,
        stages[0]: float((pred_pairs[:, 0] == true_pairs[:, 0]).mean()),
        stages[1]: float((pred_pairs[:, 1] == true_pairs[:, 1]).mean()),
    }

    # head3 risk: mean absolute error vs the graded targets
    risk_pred = pred["risk_prob"].cpu().numpy()
    risk_mae = float(np.abs(risk_pred - enc["y_risk"][te]).mean())

    result = {
        "model_version": model_version,
        "defect_head": defect,
        "bayes_optimal_accuracy": bayes_acc,
        "paper_reference": dict(PAPER_REF),
        "mechanism_accuracy": mech,
        "risk_mae": risk_mae,
    }

    # abstention metrics: only when the model has abstain outputs
    if getattr(model, "abstain", False):
        r = pred["abstain_prob"].cpu().numpy()      # P(abstain) on the defect head
        correct = d_pred == y_def
        by_h = {}
        for h in (0.3, 0.5, 0.7):
            keep = r < h                            # accept (predict) when r < h
            cov = float(keep.mean())
            sel = float(correct[keep].mean()) if keep.any() else None
            by_h[str(h)] = {"coverage": cov, "abstention_rate": 1.0 - cov,
                            "n_abstained": int((~keep).sum()),
                            "selective_accuracy": sel}
        result["abstention"] = {
            "mean_abstain_prob": float(r.mean()),
            "n_records": int(len(te)),
            "by_threshold": by_h,
            "mech_mean_abstain_prob": float(pred["mech_abstain_prob"].cpu().numpy().mean()),
        }
    return result


# Checkpoint

def load_model(path: str, device: str = "cpu") -> tuple[CascadeMLP, dict]:
    """Rebuild a trained cascade from a checkpoint written by main()."""

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CascadeMLP(
        n_features=cfg["n_features"], n_defects=cfg["n_defects"],
        n_mech=cfg["n_mech"], n_params=cfg["n_params"],
        hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
        abstain=cfg.get("abstain", False)).to(device)   # .get: pre-abstain ckpts
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# CLI

def main() -> None:
    """CLI entry: load data, train (or load a checkpoint), evaluate, save, print."""

    ap = argparse.ArgumentParser(
        description="Cascaded 3-head MLP (defect -> mechanism -> risk) on the v2 dataset")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic_v2.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--class-weight", default="none",
                    choices=["none", "sqrt", "inverse"],
                    help="defect-head class weighting (accuracy vs minority recall)")
    ap.add_argument("--loss", default="cascade", choices=list(LOSSES),
                    help="cascade = CE/CE/BCE (default); abstention = gambler loss "
                         "on the defect + mechanism heads")
    ap.add_argument("--o", type=float, default=2.0,
                    help="abstention payoff (only used by --loss abstention)")
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)),
                    help="comma-separated trunk widths, e.g. 256,256")
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mlp_v2_metrics.json")
    ap.add_argument("--model-out", default="results/mlp_v2_model.pt")
    ap.add_argument("--load", default=None,
                    help="evaluate a saved checkpoint instead of training")
    args = ap.parse_args()

    seed_everything(args.seed)
    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)

    summary = split_summary(df, spec)
    _print_split_summary(summary)

    if args.load:
        print(f"Loading checkpoint {args.load} (device={args.device})")
        model, ckpt = load_model(args.load, device=args.device)
        enc = encode_v2(df, spec)
        enc["mu"], enc["sd"] = ckpt["standardize"]["mu"], ckpt["standardize"]["sd"]
        hidden = tuple(ckpt["config"]["hidden"])
    else:
        print(f"Training {MODEL_VERSION} ({args.loss}) on {len(df):,} records (device={args.device})")
        hidden = tuple(int(w) for w in args.hidden.split(","))
        model, enc = train(spec, df, device=args.device, epochs=args.epochs,
                           class_weight_mode=args.class_weight,
                           loss_cls=LOSSES[args.loss], hidden=hidden,
                           dropout=args.dropout, seed=args.seed, o=args.o)

    metrics = evaluate(model, enc, df, spec, device=args.device,
                       model_version=MODEL_VERSION)
    metrics["data_summary"] = summary

    if not args.load:
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_version": MODEL_VERSION,
            "state_dict": model.state_dict(),
            "config": {
                "n_features": enc["X"].shape[1],
                "n_defects": len(enc["names"]),
                "n_mech": len(enc["mech_vocab"]),
                "n_params": enc["y_risk"].shape[1],
                "hidden": hidden,
                "dropout": args.dropout,
                "mech_vocab": enc["mech_vocab"],
                "abstain": getattr(model, "abstain", False),
            },
            "standardize": {"mu": enc["mu"], "sd": enc["sd"]},
            "val_loss_history": enc.get("val_loss_history", []),
            "best_epoch": enc.get("best_epoch", 0),
            "train_args": {"spec": args.spec, "data": args.data, "loss": args.loss,
                           "o": args.o, "seed": args.seed,
                           "class_weight": args.class_weight},
        }, args.model_out)
        print(f"Saved model checkpoint to {args.model_out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # console summary, lined up against the paper and the oracle ceiling
    d = metrics["defect_head"]
    print("\n=== Defect head (test split) ===")
    print(f"  {'Ours (cascade)':22s} acc {d['accuracy']:.4f}  wF1 {d['weighted_f1']:.4f}  macroF1 {d['macro_f1']:.4f}")
    print(f"  {'Bayes-optimal ceiling':22s} acc {metrics['bayes_optimal_accuracy']:.4f}")
    print(f"  {'Paper (reported)':22s} acc {PAPER_REF['accuracy']:.4f}  wF1 {PAPER_REF['weighted_f1']:.4f}")
    print("\n=== Mechanism head (joint 9-class) ===")
    m = metrics["mechanism_accuracy"]
    print(f"  {'joint accuracy':22s} {m['joint_accuracy']:.4f}")
    for stage in list(spec["mechanisms"].keys()):
        print(f"  {stage:22s} {m[stage]:.4f}  (decomposed from the joint prediction)")
    print(f"\nRisk head MAE: {metrics['risk_mae']:.4f}")

    # abstention summary (only present when --loss abstention)
    if "abstention" in metrics:
        ab = metrics["abstention"]
        o = ckpt["train_args"].get("o") if args.load else args.o
        print(f"\n=== Abstention (o={o}) ===")
        print(f"  mean abstain prob: {ab['mean_abstain_prob']:.4f}  "
              f"(mechanism head {ab['mech_mean_abstain_prob']:.4f})")
        for h, v in ab["by_threshold"].items():
            sel = v["selective_accuracy"]
            sel_s = f"{sel:.4f}" if sel is not None else "n/a"
            print(f"  h={h}: coverage {v['coverage']:.4f}  "
                  f"abstained {v['n_abstained']:,}  selective_acc {sel_s}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
