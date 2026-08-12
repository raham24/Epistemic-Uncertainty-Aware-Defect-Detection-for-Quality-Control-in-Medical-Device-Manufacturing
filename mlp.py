"""Cascaded multi-head MLP for the SMT dataset (self-contained).

Reads data/smt_synthetic.csv (from generator.py) and predicts a 3-head RCA chain,
where each head sees what the previous heads predicted:

  x --trunk--> h
                |- head1 defect : Linear(h)                  -> 3-way softmax
                |- head2 mech   : Linear(h (+) p_defect)     -> 9-way softmax  (joint)
                |- head3 risk   : Linear(h (+) p_defect (+) p_mech) -> 6 sigmoids

True cascade: each downstream head receives the trunk latent concatenated with
the upstream heads' SOFTMAX distributions (soft, so it stays differentiable and
downstream losses also sharpen the upstream heads). Trained end-to-end on the
model's own predictions -- no teacher forcing, so no train/inference mismatch.

  head1 defect : 3 classes {no_defect, open_circuit, solder_bridging}.
  head2 mech   : ONE 9-way softmax over the JOINT mechanism label mechanism_joint
                 ("<printing>__<reflow>", the 3x3 Cartesian product; only 7 occur).
  head3 risk   : the GLOBAL, two-sided per-parameter risk risk_<param> -- a
                 function of how far each parameter drifted, NOT of the mechanism.
                 So it always reports a per-parameter risk, even when head2
                 predicts no_mechanism (a clean board sits at the ~p_L floor; a
                 sub-threshold-drifting board shows elevated risk on that param).

Two losses, selectable with --loss:
  cascade    (default) cross-entropy on defect + mechanism, BCE on risk. The
             paper's loss form.
  abstention the selective-classification ("abstention") term -log(o*p_y + r) on
             the DEFECT head only (mechanism uses CE, risk uses BCE, as in cascade).
             The defect head gains one "abstain" output so the model can flag an
             ambiguous board instead of guessing; --o sets the payoff. With r=0 the
             term reduces to cross-entropy, so larger o -> abstain less. (Mechanism-
             head abstention is commented out in the model + loss, not removed --
             flip it back on to restore two-head abstention.)

Loss: L = lam_d*CE(defect) + lam_m*CE(mech) + lam_r*BCE(risk)  (or the abstention
term in place of CE on the two classification heads).

Run: python mlp.py                          # train + eval (cascade loss)
     python mlp.py --loss abstention --o 2.0
     python mlp.py --epochs 60 --device mps
     python mlp.py --load results/mlp_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# defaults (single place to change; all overridable on the CLI)
MODEL_VERSION = "mlp-cascade-1.0"
HIDDEN = (256, 256)                  # shared-trunk widths
DROPOUT = 0.1
LAMBDA_D, LAMBDA_M, LAMBDA_R = 1.0, 1.0, 1.0   # per-head loss weights
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 128, 40, 6
JOINT_SEP = "__"

# paper's reported defect-head numbers (Section V-A), for comparison
PAPER_REF = {"accuracy": 0.9500, "weighted_f1": 0.9536}


# --------------------------------------------------------------------------- #
# Spec helpers (this file reads the spec directly -- self-contained)
# --------------------------------------------------------------------------- #

def load_spec(path: str) -> dict:
    """Load the YAML domain spec."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def param_ids(spec: dict) -> list[str]:
    """The ordered parameter ids (= the model features and risk targets)."""
    return [p["id"] for p in spec["parameters"]]


def defect_names(spec: dict) -> list[str]:
    """The ordered defect class names."""
    return [d["name"] for d in spec["defects"]]


def joint_mech_vocab(spec: dict) -> list[str]:
    """The 9 joint mechanism classes, in printing-major Cartesian order:
    ["<printing>__<reflow>", ...] -- the label set for head2."""
    stages = list(spec["mechanisms"].keys())
    printing = spec["mechanisms"][stages[0]]
    reflow = spec["mechanisms"][stages[1]]
    return [f"{p}{JOINT_SEP}{r}" for p in printing for r in reflow]


def seed_everything(seed: int) -> None:
    """Pin python / numpy / torch RNGs for a reproducible run."""
    import os
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Abstention (selective-classification) term
# --------------------------------------------------------------------------- #

def abstention_term(logits: torch.Tensor, y: torch.Tensor, o: float) -> torch.Tensor:
    """Selective-classification term on one softmax head.

    Input: (B, k+1) logits whose LAST column is the abstain output, the true class
           index y in [0, k-1], and the payoff o (> 0).
    Return: mean of -log(o*p_y + r), where p_y is the true-class prob and r the
            abstain prob. r=0 reduces this to cross-entropy + const.

    Computed in log-space so o*p_y + r stays stable when both are tiny:
        log(o*p_y + r) = logsumexp([log p_y + log o, log r])   -- no eps, no spike
    """
    log_probs = torch.log_softmax(logits, dim=1)             # (B, k+1)
    log_py = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)  # log p_y (true class)
    log_r = log_probs[:, -1]                                 # log r (abstain col)
    log_o = torch.log(torch.tensor(o, device=log_probs.device, dtype=log_probs.dtype))
    log_z = torch.logsumexp(torch.stack([log_py + log_o, log_r], 0), 0)
    return -log_z.mean()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class CascadeMLP(nn.Module):
    """Shared trunk h = f(x), then a 3-head cascade: defect -> mechanism -> risk.

    Each downstream head reads the trunk latent concatenated with the upstream
    heads' softmax distributions. The risk head is P independent sigmoids (one
    per parameter), not a softmax.

    abstain widens the DEFECT head by one "abstain" output (so the abstention loss
    has a column for reject mass); that widened distribution feeds the downstream
    heads. The mechanism-head abstain path is commented out (mechanism stays a plain
    softmax) and the risk head is never widened. abstain=False is byte-identical to
    the plain cascade.
    """

    def __init__(self, n_features: int, n_defects: int, n_mech: int,
                 n_params: int, hidden: tuple[int, ...] = HIDDEN,
                 dropout: float = DROPOUT, abstain: bool = False) -> None:
        super().__init__()
        self.abstain = abstain
        self.n_defects, self.n_mech, self.n_params = n_defects, n_mech, n_params

        layers: list[nn.Module] = []
        d = n_features
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(dropout)]
            d = w
        self.trunk = nn.Sequential(*layers)

        extra = 1 if abstain else 0
        self.defect_out = n_defects + extra
        # Only the defect head abstains now; the mechanism head stays a plain softmax.
        # self.mech_out = n_mech + extra
        self.mech_out = n_mech
        self.defect_head = nn.Linear(d, self.defect_out)
        self.mech_head = nn.Linear(d + self.defect_out, self.mech_out)
        self.risk_head = nn.Linear(d + self.defect_out + self.mech_out, n_params)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        defect = self.defect_head(h)                         # head1
        p_def = F.softmax(defect, dim=1)
        mech = self.mech_head(torch.cat([h, p_def], dim=1))  # head2 sees p(defect)
        p_mech = F.softmax(mech, dim=1)
        risk = self.risk_head(torch.cat([h, p_def, p_mech], dim=1))   # head3
        return {"defect": defect, "mech": mech, "risk": risk,
                "defect_prob": p_def, "mech_prob": p_mech}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-head probabilities + argmax picks (the evidence-layer payload)."""
        self.eval()
        out = self.forward(x)
        risk_prob = torch.sigmoid(out["risk"])

        # abstain: each classification head is one column wider; the LAST column
        # is the abstain prob. Argmax over the REAL columns only (never abstain).
        if self.abstain:
            # Only the defect head abstains: its LAST column is the abstain prob and
            # the argmax is over the REAL defect classes only. The mechanism head no
            # longer abstains, so it is a plain softmax (no abstain column to strip).
            m = self.n_defects
            d_full = out["defect_prob"]
            d_real = d_full[:, :m]
            # --- mechanism-head abstain handling (disabled) ---
            # k = self.n_mech
            # m_full = out["mech_prob"]
            # m_real = m_full[:, :k]
            return {
                "defect_prob": d_real,
                "defect_argmax": d_real.argmax(1),
                "abstain_prob": d_full[:, m],
                "mech_prob": out["mech_prob"],
                "mech_argmax": out["mech_prob"].argmax(1),
                # "mech_prob": m_real,
                # "mech_argmax": m_real.argmax(1),
                # "mech_abstain_prob": m_full[:, k],
                "risk_prob": risk_prob,
            }
        return {
            "defect_prob": out["defect_prob"],
            "defect_argmax": out["defect_prob"].argmax(1),
            "mech_prob": out["mech_prob"],
            "mech_argmax": out["mech_prob"].argmax(1),
            "risk_prob": risk_prob,
        }


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

class CascadeLoss(nn.Module):
    """Composite chain loss: lam_d*CE(defect) + lam_m*CE(mech) + lam_r*BCE(risk).

    Each per-head TERM is its own method so a subclass can override only the
    classification terms (the abstention variant) without rewriting the combine.
    Subclasses needing the +1-wide abstain heads flip ABSTAIN to True.
    """

    ABSTAIN = False

    def __init__(self, class_weight: torch.Tensor | None = None,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        self.lam_d, self.lam_m, self.lam_r = lam_d, lam_m, lam_r

    def defect_term(self, out: dict, y_def: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["defect"], y_def, weight=self.class_weight)

    def mech_term(self, out: dict, y_mech: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out["mech"], y_mech)

    def risk_term(self, out: dict, y_risk: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(out["risk"], y_risk)

    def forward(self, out: dict, y_def: torch.Tensor, y_mech: torch.Tensor,
                y_risk: torch.Tensor) -> torch.Tensor:
        return (self.lam_d * self.defect_term(out, y_def)
                + self.lam_m * self.mech_term(out, y_mech)
                + self.lam_r * self.risk_term(out, y_risk))


class CascadeAbstentionLoss(CascadeLoss):
    """The abstention variant: the selective-classification term on the DEFECT head
    only; the mechanism head uses plain CE and the sigmoid/BCE risk head is unchanged.
    (The mechanism-head abstention override below is commented out, not removed.)

    class_weight is accepted for train()-API compatibility but IGNORED -- the
    abstain mechanism, not reweighting, handles the class imbalance.
    """

    ABSTAIN = True

    def __init__(self, class_weight: torch.Tensor | None = None, o: float = 2.0,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__(class_weight=None, lam_d=lam_d, lam_m=lam_m, lam_r=lam_r)
        self.o = o

    def defect_term(self, out: dict, y_def: torch.Tensor) -> torch.Tensor:
        return abstention_term(out["defect"], y_def, self.o)

    # Mechanism head no longer abstains -> inherit the parent's plain CE on it
    # (do NOT override with the abstention term).
    # def mech_term(self, out: dict, y_mech: torch.Tensor) -> torch.Tensor:
    #     return abstention_term(out["mech"], y_mech, self.o)


# loss registry exposed via --loss; cascade is the paper-style default
LOSSES = {"cascade": CascadeLoss, "abstention": CascadeAbstentionLoss}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def encode(df: pd.DataFrame, spec: dict) -> dict:
    """Turn the dataframe into model arrays (features + per-head targets)."""

    ids = param_ids(spec)
    names = defect_names(spec)
    vocab = joint_mech_vocab(spec)

    X = df[ids].to_numpy(np.float32)                         # the 6 raw parameters

    d_idx = {n: i for i, n in enumerate(names)}
    y_def = df["defect_label"].map(d_idx).to_numpy(np.int64)

    m_idx = {m: i for i, m in enumerate(vocab)}
    mapped = df["mechanism_joint"].map(m_idx)
    if mapped.isna().any():
        bad = sorted(df.loc[mapped.isna(), "mechanism_joint"].unique())
        raise ValueError(f"mechanism_joint has values outside the 9-class vocab "
                         f"{vocab}: {bad}")
    y_mech = mapped.to_numpy(np.int64)

    y_risk = df[[f"risk_{p}" for p in ids]].to_numpy(np.float32)   # graded targets

    return {"X": X, "y_def": y_def, "y_mech": y_mech, "y_risk": y_risk,
            "names": names, "mech_vocab": vocab, "ids": ids}


def _standardize(Xtr: np.ndarray, *others: np.ndarray):
    """Fit mean/std on TRAIN only; apply to every split. Returns (arrays, mu, sd)
    so the caller can STORE the exact train stats and reuse them at eval."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    arrs = [((X - mu) / sd).astype(np.float32) for X in (Xtr, *others)]
    return arrs, mu, sd


def _class_weights(y: np.ndarray, n: int, mode: str) -> torch.Tensor | None:
    """Defect-head class weights, normalized to mean 1 (None if mode == 'none')."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n).astype(np.float64)
    w = 1.0 / np.clip(counts, 1, None)
    if mode == "sqrt":
        w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _loader(enc: dict, idx: np.ndarray, batch: int, shuffle: bool,
            generator: torch.Generator | None = None) -> DataLoader:
    """Wrap one split's tensors (X, y_def, y_mech, y_risk) in a DataLoader."""
    ds = TensorDataset(torch.from_numpy(enc["X"][idx]),
                       torch.from_numpy(enc["y_def"][idx]),
                       torch.from_numpy(enc["y_mech"][idx]),
                       torch.from_numpy(enc["y_risk"][idx]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      generator=generator)


def split_summary(df: pd.DataFrame, spec: dict) -> dict:
    """Per-split record counts and per-defect-class counts within each split."""
    names = defect_names(spec)
    out: dict = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        counts = sub["defect_label"].value_counts()
        out[split] = {"n": int(len(sub)),
                      "classes": {n: int(counts.get(n, 0)) for n in names}}
    return out


def _print_split_summary(summary: dict) -> None:
    """Console table: one row per defect class, one column per split."""
    splits = list(summary)
    classes = list(next(iter(summary.values()))["classes"])
    width = max(len(c) for c in classes + ["class"]) + 2
    header = f"  {'class':{width}s}" + "".join(f"{s:>10s}" for s in splits)
    print("\n=== Dataset split ===")
    print(header)
    for c in classes:
        row = "".join(f"{summary[s]['classes'][c]:>10,d}" for s in splits)
        print(f"  {c:{width}s}{row}")
    totals = "".join(f"{summary[s]['n']:>10,d}" for s in splits)
    print(f"  {'total':{width}s}{totals}")


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def _clone(model: nn.Module) -> dict:
    """A CPU copy of the model weights (for the early-stopping snapshot)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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
          batch: int = BATCH, warmup: int = 0) -> tuple[CascadeMLP, dict]:
    """Train the cascade with early stopping on validation loss.

    Return: the best model and the encoded-data bundle (stats + val curve).
    """

    enc = encode(df, spec)
    sp = df["split"].to_numpy()
    tr, va = np.where(sp == "train")[0], np.where(sp == "val")[0]

    # standardize on TRAIN stats only; store them so eval reuses them exactly
    stdz, enc["mu"], enc["sd"] = _standardize(enc["X"][tr], enc["X"][va])
    enc["X"][tr], enc["X"][va] = stdz

    abstain = getattr(loss_cls, "ABSTAIN", False)            # widen the heads?
    model = CascadeMLP(
        n_features=enc["X"].shape[1], n_defects=len(enc["names"]),
        n_mech=len(enc["mech_vocab"]), n_params=enc["y_risk"].shape[1],
        hidden=hidden, dropout=dropout, abstain=abstain).to(device)

    cw = _class_weights(enc["y_def"][tr], len(enc["names"]), class_weight_mode)
    loss_kwargs = {"o": o} if abstain else {}                # only abstention takes o
    loss_fn = loss_cls(class_weight=cw.to(device) if cw is not None else None,
                       **loss_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    g = torch.Generator()
    g.manual_seed(seed)                                      # reproducible shuffle
    tr_dl = _loader(enc, tr, batch, shuffle=True, generator=g)
    va_dl = _loader(enc, va, batch, shuffle=False)

    # classifier warmup (abstention only): for the first `warmup` epochs train at a HIGH
    # payoff (>= n_defects => full betting, no abstention, cf. the regime theorem) so the
    # classifier is competent BEFORE the reservation term can collapse it into abstaining
    # on everything; then drop to the target o. Avoids the degenerate-basin seeds.
    warmup = int(warmup) if abstain else 0
    warm_o = max(o, float(len(enc["names"])) + 5.0)

    best_val, best_state, waited, val_hist = float("inf"), None, 0, []
    for ep in range(epochs):
        if warmup:
            loss_fn.o = warm_o if ep < warmup else o     # classifier first, then abstention
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
        print(f"  epoch {ep:2d}  val_loss {val:.4f}" + ("  (warmup)" if warmup and ep < warmup else ""))
        if warmup and ep < warmup:
            continue                                     # don't select/early-stop at warm_o
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
    """Per-split metrics: defect accuracy/F1 (+ Bayes ceiling), joint-mechanism
    accuracy with a per-stage decomposition, per-parameter risk MAE, and (for an
    abstention model) coverage / selective accuracy."""

    names, vocab, ids = enc["names"], enc["mech_vocab"], enc["ids"]
    te = np.where(df["split"].to_numpy() == split)[0]

    # pull RAW features for these rows and standardize with the STORED train stats
    Xraw = df.iloc[te][ids].to_numpy(np.float32)
    Xte = ((Xraw - enc["mu"]) / enc["sd"]).astype(np.float32)
    pred = model.predict(torch.from_numpy(Xte).to(device))

    # head1 defect
    y_def = enc["y_def"][te]
    d_pred = pred["defect_argmax"].cpu().numpy()
    defect = {
        "accuracy": float(accuracy_score(y_def, d_pred)),
        "weighted_f1": float(f1_score(y_def, d_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_def, d_pred, average="macro", zero_division=0)),
    }
    post = df.iloc[te][[f"p_{n}" for n in names]].to_numpy()
    bayes_acc = float(accuracy_score(y_def, post.argmax(1)))

    # head2 mechanism: joint accuracy + per-stage decomposition
    y_mech = enc["y_mech"][te]
    m_pred = pred["mech_argmax"].cpu().numpy()
    joint_acc = float(accuracy_score(y_mech, m_pred))
    pred_pairs = np.array([vocab[i].split(JOINT_SEP) for i in m_pred])
    true_pairs = np.array([vocab[i].split(JOINT_SEP) for i in y_mech])
    stages = list(spec["mechanisms"].keys())
    mech = {
        "joint_accuracy": joint_acc,
        stages[0]: float((pred_pairs[:, 0] == true_pairs[:, 0]).mean()),
        stages[1]: float((pred_pairs[:, 1] == true_pairs[:, 1]).mean()),
    }

    # head3 risk
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
        r = pred["abstain_prob"].cpu().numpy()
        correct = d_pred == y_def
        by_h = {}
        for h in (0.3, 0.5, 0.7):
            keep = r < h                                     # accept (predict) when r < h
            cov = float(keep.mean())
            sel = float(correct[keep].mean()) if keep.any() else None
            by_h[str(h)] = {"coverage": cov, "abstention_rate": 1.0 - cov,
                            "n_abstained": int((~keep).sum()),
                            "selective_accuracy": sel}
        result["abstention"] = {
            "mean_abstain_prob": float(r.mean()),
            "n_records": int(len(te)),
            "by_threshold": by_h,
            # mechanism head no longer abstains -> no mech abstain prob to report.
            # "mech_mean_abstain_prob": float(pred["mech_abstain_prob"].cpu().numpy().mean()),
        }
    return result


def load_model(path: str, device: str = "cpu") -> tuple[CascadeMLP, dict]:
    """Rebuild a trained cascade from a checkpoint written by main()."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CascadeMLP(
        n_features=cfg["n_features"], n_defects=cfg["n_defects"],
        n_mech=cfg["n_mech"], n_params=cfg["n_params"],
        hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
        abstain=cfg.get("abstain", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    """CLI: load data, train (or load a checkpoint), evaluate, save, print."""

    ap = argparse.ArgumentParser(
        description="Cascaded 3-head MLP (defect -> mechanism -> risk)")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--class-weight", default="none",
                    choices=["none", "sqrt", "inverse"],
                    help="defect-head class weighting (accuracy vs minority recall)")
    ap.add_argument("--loss", default="cascade", choices=list(LOSSES),
                    help="cascade = CE/CE/BCE (default); abstention = the "
                         "selective-classification term on the defect + mech heads")
    ap.add_argument("--warmup", type=int, default=0,
                    help="abstention only: epochs of classifier-only warmup (high payoff) "
                         "before the reservation term is applied at --o")
    ap.add_argument("--o", type=float, default=2.0,
                    help="abstention payoff (only used by --loss abstention)")
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)),
                    help="comma-separated trunk widths, e.g. 256,256")
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--lr", type=float, default=LR, help="Adam learning rate")
    ap.add_argument("--batch", type=int, default=BATCH, help="minibatch size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mlp_metrics.json")
    ap.add_argument("--model-out", default="results/mlp_model.pt")
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
        enc = encode(df, spec)
        enc["mu"], enc["sd"] = ckpt["standardize"]["mu"], ckpt["standardize"]["sd"]
        hidden = tuple(ckpt["config"]["hidden"])
    else:
        print(f"Training {MODEL_VERSION} ({args.loss}) on {len(df):,} records (device={args.device})")
        hidden = tuple(int(w) for w in args.hidden.split(","))
        model, enc = train(spec, df, device=args.device, epochs=args.epochs,
                           class_weight_mode=args.class_weight,
                           loss_cls=LOSSES[args.loss], hidden=hidden,
                           dropout=args.dropout, seed=args.seed, o=args.o,
                           lr=args.lr, batch=args.batch, warmup=args.warmup)

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
                           "o": args.o, "seed": args.seed, "lr": args.lr,
                           "batch": args.batch, "epochs": args.epochs,
                           "dropout": args.dropout, "class_weight": args.class_weight,
                           "warmup": args.warmup},
        }, args.model_out)
        print(f"Saved model checkpoint to {args.model_out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)

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

    if "abstention" in metrics:
        ab = metrics["abstention"]
        o = ckpt["train_args"].get("o") if args.load else args.o
        print(f"\n=== Abstention (o={o}) ===")
        print(f"  mean abstain prob: {ab['mean_abstain_prob']:.4f}")
        # mechanism head no longer abstains; nothing to report for it:
        #     print(f"  (mechanism head {ab['mech_mean_abstain_prob']:.4f})")
        for h, v in ab["by_threshold"].items():
            sel = v["selective_accuracy"]
            sel_s = f"{sel:.4f}" if sel is not None else "n/a"
            print(f"  h={h}: coverage {v['coverage']:.4f}  "
                  f"abstained {v['n_abstained']:,}  selective_acc {sel_s}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
