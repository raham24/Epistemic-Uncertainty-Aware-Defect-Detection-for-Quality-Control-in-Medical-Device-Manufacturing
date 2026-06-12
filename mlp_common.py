"""Shared multi-head MLP machinery for the SMT replication.

Both entry points build on this one module:
  - mlp_paper.py : the paper's exact model -- 2 softmax/CE heads (defect,
    mechanism) + 1 sigmoid/BCE risk head. The safety-net baseline.
  - mlp.py       : our version -- the abstention ("gambler") loss on BOTH
    classification heads (defect AND each mechanism head), same sigmoid/BCE
    risk head.

Everything that is identical between the two -- the shared trunk, the data
encoding, standardization, the train loop, evaluate, the CLI -- lives here.
Each entry file only adds its own loss (and registry) and calls run_cli().

The single switch is MultiHeadMLP(abstain=...): when True the defect head and
every mechanism head gain ONE extra "abstain" output (so the gambler loss has a
column to put reject mass on). The risk head is never widened. When abstain is
False every code path is byte-identical to the original single-file model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from generator import defect_names, load_spec, param_ids, seed_everything

# Customization knobs (single place to change; all overridable on the CLI)

DEFAULT_VERSION = "mlp-v0.1"         # stamped into outputs (provenance) by default
HIDDEN = (256, 256)                  # shared-trunk widths
DROPOUT = 0.1                        # trunk dropout
LAMBDA_D, LAMBDA_M, LAMBDA_R = 1.0, 1.0, 1.0   # loss weights per head type
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 128, 40, 6  # optimizer + early-stop
CLASS_WEIGHT_MODE = "none"

# Paper's reported defect-head numbers (Section V-A), for comparison.
PAPER_REF = {"accuracy": 0.9500, "weighted_f1": 0.9536}


# Gambler / selective-classification term (shared by both classification heads)

def gambler_term(logits: torch.Tensor, y: torch.Tensor, o: float) -> torch.Tensor:
    """Algorithm — Selective-classification "gambler" loss on one softmax head.

    Input: (B, k+1) logits whose LAST column is the abstain output, the true
           class index y in [0, k-1], and the payoff o (> 0).
    Return: mean of -log(o*p_y + r), where p_y is the true-class prob and r the
            abstain prob. r=0 reduces this to cross-entropy + const.
    """

    # work in log-space so o*p_y + r is stable even when both are tiny.
    # log(o*p_y + r) = logsumexp([log p_y + log o, log r]) -- no eps, no spike
    log_probs = torch.log_softmax(logits, dim=1)             # (B, k+1)
    log_py = log_probs.gather(1, y.unsqueeze(1)).squeeze(1)  # log p_y (true class)
    log_r = log_probs[:, -1]                                 # log r (abstain col)
    log_o = torch.log(torch.tensor(o, device=log_probs.device,
                                   dtype=log_probs.dtype))
    log_z = torch.logsumexp(torch.stack([log_py + log_o, log_r], 0), 0)
    return -log_z.mean()


# Model

class MultiHeadMLP(nn.Module):
    """Shared trunk h = f(x) with a defect head, per-stage mechanism heads, and
    P independent per-parameter sigmoid risk heads (Eq. 3) -- implemented as one
    fused nn.Linear(d, n_params) + elementwise sigmoid, equivalent to P separate
    sigmoid heads, NOT a softmax across parameters. Sizes come from the spec.

    abstain widens the CLASSIFICATION heads only: the defect head and EVERY
    mechanism head get one extra "abstain" output (m+1 / k+1 wide). The risk head
    is never touched."""

    def __init__(self, n_features: int, n_defects: int, mech_sizes: list[int],
                 n_params: int, hidden: tuple[int, ...] = HIDDEN,
                 dropout: float = DROPOUT, abstain: bool = False) -> None:
        super().__init__()

        # remember the abstain setting and the REAL (pre-abstain) head sizes
        self.abstain = abstain
        self.n_defects = n_defects
        self.mech_sizes = list(mech_sizes)

        # build the shared trunk: Linear -> ReLU -> Dropout, stacked
        layers: list[nn.Module] = []
        d = n_features
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(dropout)]
            d = w
        self.trunk = nn.Sequential(*layers)

        # one head per task; mechanism heads are one-per-stage.
        # abstain adds ONE extra output to each classification head (the
        # "abstain" class) -> defect is m+1 wide, each mechanism head is k+1 wide
        extra = 1 if abstain else 0
        self.defect_head = nn.Linear(d, n_defects + extra)
        self.mech_heads = nn.ModuleList(nn.Linear(d, k + extra) for k in mech_sizes)
        self.risk_head = nn.Linear(d, n_params)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        # shared latent, then each head reads from it
        h = self.trunk(x)
        return {
            "defect": self.defect_head(h),                      # (B, n_defects[+1]) logits
            "mech": [head(h) for head in self.mech_heads],      # list of (B, k_s[+1]) logits
            "risk": self.risk_head(h),                          # (B, n_params) logits
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, object]:
        """Algorithm — Calibrated per-head evidence for the symbolic layer.

        Input: feature batch x.
        Return: per-head probabilities (the hasProbability payload) + argmax picks.
        """

        # turn the raw logits into the probabilities the evidence layer needs
        self.eval()
        out = self.forward(x)
        risk_prob = torch.sigmoid(out["risk"])                   # independent sigmoid

        # abstain: every classification head is one column wider; the LAST column
        # is the abstain prob. Pick the class over the REAL columns only (never
        # the abstain column), and expose the abstain prob alongside it.
        if self.abstain:
            # defect head
            full = torch.softmax(out["defect"], dim=1)           # (B, m+1)
            m = self.n_defects
            defect_prob = full[:, :m]                            # real-class probs

            # each mechanism head, same treatment
            mech_full = [torch.softmax(mm, dim=1) for mm in out["mech"]]
            mech_prob = [p[:, :k] for p, k in zip(mech_full, self.mech_sizes)]

            return {
                "defect_prob": defect_prob,
                "defect_argmax": defect_prob.argmax(1),         # over the m real classes
                "abstain_prob": full[:, m],                     # P(abstain) = r(x)
                "mech_prob": mech_prob,
                "mech_argmax": [p.argmax(1) for p in mech_prob],  # over real classes
                "mech_abstain_prob": [p[:, -1] for p in mech_full],
                "risk_prob": risk_prob,
            }

        # default (no abstain) path -- unchanged
        mech_prob = [torch.softmax(m, dim=1) for m in out["mech"]]  # p(m_s | x)
        defect_prob = torch.softmax(out["defect"], dim=1)        # p(y | x)
        return {
            "defect_prob": defect_prob,
            "defect_argmax": defect_prob.argmax(1),             # selected defect (chain root)
            "mech_prob": mech_prob,
            "mech_argmax": [p.argmax(1) for p in mech_prob],    # selected mechanism per stage
            "risk_prob": risk_prob,
        }


class MultiHeadLoss(nn.Module):
    """Composite loss (Eq. 1-3): lam_d*CE(defect) + sum_s lam_m*CE(mech_s) +
    lam_r*BCE(risk). Each per-head TERM is its own method so a subclass can
    override just one (e.g. an abstention defect term) without rewriting the
    combine. train()'s optimizer already includes loss_fn.parameters(), so a
    subclass with nn.Parameter head weights trains with no plumbing change.

    This is the PAPER loss as written. mlp.py subclasses it to swap the two
    classification terms for the gambler term."""

    # subclasses that need the m+1-wide abstain heads flip this to True
    ABSTAIN = False

    def __init__(self, class_weight: torch.Tensor | None = None,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__()

        # class_weight is a fixed tensor (not learned) -> register as a buffer
        self.register_buffer("class_weight", class_weight)
        self.lam_d, self.lam_m, self.lam_r = lam_d, lam_m, lam_r

    # per-head TERMS (override one or more in a subclass to experiment)

    def defect_term(self, out: dict, y_defect: torch.Tensor) -> torch.Tensor:
        # weighted cross-entropy (handles the ~88% no_defect imbalance)
        return F.cross_entropy(out["defect"], y_defect, weight=self.class_weight)

    def mech_term(self, logits: torch.Tensor, y_mech_s: torch.Tensor) -> torch.Tensor:
        # plain cross-entropy for one stage's mechanism head
        return F.cross_entropy(logits, y_mech_s)

    def risk_term(self, out: dict, y_risk: torch.Tensor) -> torch.Tensor:
        # BCE with SOFT targets in (0,1) -> regress the graded risk.
        # default reduction='mean' averages over BOTH batch AND the P param
        # columns, so this term is (1/P)*sum_j BCE_j -- lam_r weights the AVERAGE
        # risk head, not the SUM written in CLAUDE.md. Kept as mean-of-heads on
        # purpose so every head term sits on a comparable per-head scale.
        return F.binary_cross_entropy_with_logits(out["risk"], y_risk)

    # COMBINE (subclasses override the terms above, not this)

    def forward(self, out: dict, y_defect: torch.Tensor, y_mech: torch.Tensor,
                y_risk: torch.Tensor) -> torch.Tensor:
        loss = self.lam_d * self.defect_term(out, y_defect)
        for s, logits in enumerate(out["mech"]):
            loss = loss + self.lam_m * self.mech_term(logits, y_mech[:, s])
        loss = loss + self.lam_r * self.risk_term(out, y_risk)
        return loss


# Data

def _encode(df: pd.DataFrame, spec: dict) -> dict:
    """Algorithm — Turn the dataframe into model arrays.

    Input: the generated dataframe and the spec.
    Return: features X and integer/float targets for every head, as numpy.
    """

    ids = param_ids(spec)
    names = defect_names(spec)
    stages = list(spec["mechanisms"].keys())

    # features: the 6 raw parameter values
    X = df[ids].to_numpy(np.float32)

    # defect target: class name -> index (order fixed by the spec)
    d_idx = {n: i for i, n in enumerate(names)}
    y_def = df["defect_label"].map(d_idx).to_numpy(np.int64)

    # mechanism targets: one column per stage, each name -> index in that stage
    y_mech = np.empty((len(df), len(stages)), np.int64)
    for s, stage in enumerate(stages):
        m_idx = {m: i for i, m in enumerate(spec["mechanisms"][stage])}
        col = f"{stage}_mechanism_label"
        mapped = df[col].map(m_idx)
        # a label not in the spec vocab maps to NaN -> fail loudly instead of
        # silently casting NaN to a garbage int index (caught a typo'd label)
        if mapped.isna().any():
            bad = sorted(df.loc[mapped.isna(), col].unique())
            raise ValueError(f"{col} has values outside the spec mechanism "
                             f"vocab {list(m_idx)}: {bad}")
        y_mech[:, s] = mapped.to_numpy()

    # risk targets: the graded risk columns, already in (0,1)
    y_risk = df[[f"risk_{p}" for p in ids]].to_numpy(np.float32)
    return {"X": X, "y_def": y_def, "y_mech": y_mech, "y_risk": y_risk,
            "names": names, "stages": stages}


def _standardize(Xtr: np.ndarray, *others: np.ndarray):
    """Fit mean/std on TRAIN only (paper Section IV-C), apply to every split.

    Return: (list of standardized arrays, mu, sd) so the caller can STORE the
    exact train stats and reuse them at eval instead of recomputing them.
    """

    # train statistics, with a floor on std to avoid divide-by-zero
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    arrs = [((X - mu) / sd).astype(np.float32) for X in (Xtr, *others)]
    return arrs, mu, sd


def _loader(enc: dict, idx: np.ndarray, batch: int, shuffle: bool,
            generator: torch.Generator | None = None) -> DataLoader:
    """Wrap one split's tensors in a DataLoader (seeded shuffle if given)."""

    # pack features + all targets into a single dataset
    ds = TensorDataset(torch.from_numpy(enc["X"][idx]),
                       torch.from_numpy(enc["y_def"][idx]),
                       torch.from_numpy(enc["y_mech"][idx]),
                       torch.from_numpy(enc["y_risk"][idx]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      generator=generator)


def split_summary(df: pd.DataFrame, spec: dict) -> dict:
    """Per-split record counts and per-defect-class counts within each split.

    Return: {split: {"n": total, "classes": {defect_name: count}}}, with splits
    in train/val/test order and classes in spec order (zeros included).
    """

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


def _class_weights(y: np.ndarray, n: int, mode: str) -> torch.Tensor | None:
    """Defect-head class weights, normalized to mean 1 (None if mode == 'none')."""

    # no weighting -> let cross-entropy treat every record equally
    if mode == "none":
        return None

    # count each class; weight is 1/count (inverse) or 1/sqrt(count) (sqrt)
    counts = np.bincount(y, minlength=n).astype(np.float64)
    w = 1.0 / np.clip(counts, 1, None)
    if mode == "sqrt":
        w = np.sqrt(w)

    # scale so the mean weight is 1 (keeps the loss magnitude comparable)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

# Train and evaluate

def train(spec: dict, df: pd.DataFrame, device: str = "cpu",
          epochs: int = EPOCHS,
          class_weight_mode: str = CLASS_WEIGHT_MODE,
          loss_cls: type[MultiHeadLoss] = MultiHeadLoss,
          hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT,
          seed: int = 0, o: float = 2.0, lr: float = LR,
          batch: int = BATCH) -> tuple[MultiHeadMLP, dict]:
    """Algorithm — Train the multi-head MLP with early stopping on val loss.

    Input: spec, the generated dataframe, device, epoch budget, weighting mode,
           loss class, trunk hidden/dropout, a seed for the shuffle, the
           abstention payoff o, the learning rate lr, and the batch size.
    Return: the best model and the encoded-data bundle (with stats + val history).
    """

    enc = _encode(df, spec)

    # split indices come straight from the leak-free batch-grouped split column
    sp = df["split"].to_numpy()
    tr, va = np.where(sp == "train")[0], np.where(sp == "val")[0]

    # standardize using train stats only; STORE the stats so eval reuses them
    # (test rows enc["X"][te] are left RAW on purpose)
    stdz, enc["mu"], enc["sd"] = _standardize(enc["X"][tr], enc["X"][va])
    enc["X"][tr], enc["X"][va] = stdz

    # an abstention loss needs the m+1 / k+1-wide classification heads
    abstain = getattr(loss_cls, "ABSTAIN", False)

    # build the model with sizes taken from the spec
    model = MultiHeadMLP(
        n_features=enc["X"].shape[1], n_defects=len(enc["names"]),
        mech_sizes=[len(spec["mechanisms"][s]) for s in enc["stages"]],
        n_params=enc["y_risk"].shape[1], hidden=hidden, dropout=dropout,
        abstain=abstain).to(device)

    # optional class weighting + Adam (the paper-default optimizer)
    cw = _class_weights(enc["y_def"][tr], len(enc["names"]), class_weight_mode)
    # pass o only to an abstention loss (the base loss has no o argument)
    loss_kwargs = {"o": o} if abstain else {}
    loss_fn = loss_cls(class_weight=cw.to(device) if cw is not None else None,
                       **loss_kwargs)
    loss_fn = loss_fn.to(device)
    # loss_fn.parameters() is included on purpose: a learnable-weight loss
    # subclass (nn.Parameter head weights) then trains with no change to train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr)

    # seed the shuffle explicitly so a run is reproducible by its own seed,
    # not by the global-RNG ordering of everything that ran before it
    g = torch.Generator()
    g.manual_seed(seed)
    tr_dl = _loader(enc, tr, batch, shuffle=True, generator=g)
    va_dl = _loader(enc, va, batch, shuffle=False)

    # early stopping: keep the weights with the lowest validation loss.
    # val_hist records the per-epoch val loss for the training-curve plot
    best_val, best_state, waited, val_hist = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        for xb, yd, ym, yr in tr_dl:
            xb, yd, ym, yr = xb.to(device), yd.to(device), ym.to(device), yr.to(device)
            opt.zero_grad()                 # clear last step's gradients
            out = model(xb)                 # forward pass
            loss = loss_fn(out, yd, ym, yr) # composite loss
            # stop loudly on a NaN/Inf loss instead of training on garbage
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {ep} "
                    f"(loss={loss_cls.__name__}, o={o})")
            loss.backward()                 # autograd fills every gradient
            opt.step()                      # update weights

        # validation loss (no gradients) drives early stopping
        val = _epoch_loss(model, loss_fn, va_dl, device)
        val_hist.append(val)            # keep the per-epoch curve
        print(f"  epoch {ep:2d}  val_loss {val:.4f}")
        if val < best_val - 1e-4:
            best_val, best_state, waited = val, _clone(model), 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"  early stop at epoch {ep}")
                break

    # restore the best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    # stash the val curve + best epoch for the training-curve plot
    enc["val_loss_history"] = val_hist
    enc["best_epoch"] = int(np.argmin(val_hist)) if val_hist else 0
    return model, enc


def _epoch_loss(model: MultiHeadMLP, loss_fn: MultiHeadLoss,
                dl: DataLoader, device: str) -> float:
    """Average loss over a loader (no gradients)."""

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yd, ym, yr in dl:
            xb, yd, ym, yr = xb.to(device), yd.to(device), ym.to(device), yr.to(device)
            total += float(loss_fn(model(xb), yd, ym, yr)) * len(xb)
            n += len(xb)
    return total / max(n, 1)


def _clone(model: nn.Module) -> dict:
    """A CPU copy of the model weights (for the early-stopping snapshot)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model(path: str, device: str = "cpu") -> tuple[MultiHeadMLP, dict]:
    """Rebuild a trained model from a checkpoint written by run_cli().

    Return: (model in eval mode, the full checkpoint dict -- including the
    train-time standardization stats under "standardize").
    """

    # weights_only=False: the checkpoint carries numpy mu/sd + config dicts,
    # not just tensors (it's our own file, so this is safe)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultiHeadMLP(
        n_features=cfg["n_features"], n_defects=cfg["n_defects"],
        mech_sizes=cfg["mech_sizes"], n_params=cfg["n_params"],
        hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"],
        abstain=cfg["abstain"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def evaluate(model: MultiHeadMLP, enc: dict, df: pd.DataFrame,
             spec: dict, device: str = "cpu", split: str = "test",
             model_version: str = DEFAULT_VERSION) -> dict:
    """Algorithm — Per-split metrics, comparable to the paper and Bayes ceiling.

    Input: trained model, encoded data, dataframe, spec, device, split name
           (default 'test'; use 'val' for hyperparameter tuning -- never tune
           on 'test'), and the version string to stamp into the output.
    Return: defect accuracy/F1, per-stage mechanism accuracy, risk MAE, and an
            abstention section when the model has abstain outputs.
    """

    names, stages = enc["names"], enc["stages"]
    ids = param_ids(spec)
    te = np.where(df["split"].to_numpy() == split)[0]

    # pull RAW features for these rows straight from df and standardize with the
    # STORED train stats. We do NOT reuse enc["X"][te]: train() standardized the
    # train/val rows IN PLACE, so only test stays raw there -- recomputing raw
    # from df is correct for ANY split (and byte-identical to before for test).
    Xraw = df.iloc[te][ids].to_numpy(np.float32)
    Xte = ((Xraw - enc["mu"]) / enc["sd"]).astype(np.float32)

    # one predict pass -> calibrated per-head evidence (the symbolic payload)
    pred = model.predict(torch.from_numpy(Xte).to(device))

    # defect head: argmax of the posterior -> accuracy + weighted/macro F1
    y_def = enc["y_def"][te]
    d_pred = pred["defect_argmax"].cpu().numpy()
    defect = {
        "accuracy": float(accuracy_score(y_def, d_pred)),
        "weighted_f1": float(f1_score(y_def, d_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_def, d_pred, average="macro", zero_division=0)),
    }

    # Bayes-optimal ceiling: argmax of the KNOWN posterior on the same rows
    post = df.iloc[te][[f"p_{n}" for n in names]].to_numpy()
    bayes_acc = float(accuracy_score(y_def, post.argmax(1)))

    # mechanism heads: per-stage accuracy
    mech = {}
    for s, stage in enumerate(stages):
        m_pred = pred["mech_argmax"][s].cpu().numpy()
        mech[stage] = float(accuracy_score(enc["y_mech"][te][:, s], m_pred))

    # risk heads: mean absolute error vs the graded targets
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
        correct = d_pred == y_def                   # right on the m real classes
        # how often abstain outright wins the m+1-way argmax (defect head)
        n_argmax = int((r > pred["defect_prob"].cpu().numpy().max(1)).sum())
        by_h = {}
        for h in (0.3, 0.5, 0.7):
            keep = r < h                            # accept (predict) when r < h
            cov = float(keep.mean())                # fraction we predict on
            # accuracy on the accepted set; None if we abstained on everything
            sel = float(correct[keep].mean()) if keep.any() else None
            by_h[str(h)] = {"coverage": cov, "abstention_rate": 1.0 - cov,
                            "n_abstained": int((~keep).sum()),
                            "selective_accuracy": sel}
        # the mechanism heads now abstain too -- report their mean abstain prob
        mech_abstain = {stages[s]: float(p.cpu().numpy().mean())
                        for s, p in enumerate(pred["mech_abstain_prob"])}
        result["abstention"] = {"mean_abstain_prob": float(r.mean()),
                                "n_records": int(len(te)),
                                "argmax_abstentions": n_argmax,
                                "by_threshold": by_h,
                                "mech_mean_abstain_prob": mech_abstain}
    return result


# CLI (shared entry point; each file passes its own losses + version)

def run_cli(model_version: str, losses: dict, default_loss: str,
            description: str) -> None:
    """Algorithm — Parse args, train (or load), evaluate, save, and print.

    Input: the version string to stamp, the {name: loss_cls} registry to expose
           via --loss, the default loss name, and the argparse description.
    Return: nothing; writes the metrics JSON and (when training) a checkpoint.
    """

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--class-weight", default=CLASS_WEIGHT_MODE,
                    choices=["none", "sqrt", "inverse"],
                    help="defect-head class weighting (accuracy vs minority recall)")
    ap.add_argument("--loss", default=default_loss, choices=list(losses),
                    help="which loss to train with")
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)),
                    help="comma-separated trunk widths, e.g. 256,256")
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--o", type=float, default=2.0,
                    help="abstention payoff (only used by an abstention loss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mlp_metrics.json")
    ap.add_argument("--model-out", default="results/mlp_model.pt",
                    help="checkpoint path (state_dict + config + train stats)")
    ap.add_argument("--load", default=None,
                    help="evaluate a saved checkpoint instead of training")
    args = ap.parse_args()

    # pin every RNG source so a run is reproducible
    seed_everything(args.seed)

    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)

    summary = split_summary(df, spec)
    _print_split_summary(summary)

    if args.load:
        # skip training: rebuild the model and the train-time standardization
        # stats from the checkpoint, then evaluate on the test split as usual
        print(f"Loading checkpoint {args.load} (device={args.device})")
        model, ckpt = load_model(args.load, device=args.device)
        enc = _encode(df, spec)
        enc["mu"], enc["sd"] = ckpt["standardize"]["mu"], ckpt["standardize"]["sd"]
    else:
        print(f"Training {model_version} on {len(df):,} records (device={args.device})")
        hidden = tuple(int(w) for w in args.hidden.split(","))
        model, enc = train(spec, df, device=args.device, epochs=args.epochs,
                           class_weight_mode=args.class_weight,
                           loss_cls=losses[args.loss], hidden=hidden,
                           dropout=args.dropout, seed=args.seed, o=args.o)
    metrics = evaluate(model, enc, df, spec, device=args.device,
                       model_version=model_version)
    metrics["data_summary"] = summary

    if not args.load:
        # save the checkpoint: weights + everything needed to rebuild the model
        # and standardize new inputs exactly as at train time
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_version": model_version,
            "state_dict": model.state_dict(),
            "config": {
                "n_features": enc["X"].shape[1],
                "n_defects": len(enc["names"]),
                "mech_sizes": [len(spec["mechanisms"][s]) for s in enc["stages"]],
                "n_params": enc["y_risk"].shape[1],
                "hidden": hidden,
                "dropout": args.dropout,
                "abstain": getattr(model, "abstain", False),
            },
            "standardize": {"mu": enc["mu"], "sd": enc["sd"]},
            # the per-epoch val curve + best epoch, so the training-curve plot
            # can be drawn straight from the checkpoint (no retraining)
            "val_loss_history": enc.get("val_loss_history", []),
            "best_epoch": enc.get("best_epoch", 0),
            "train_args": {"spec": args.spec, "data": args.data, "loss": args.loss,
                           "o": args.o, "seed": args.seed,
                           "class_weight": args.class_weight},
        }, args.model_out)
        print(f"Saved model checkpoint to {args.model_out}")

    # write metrics for the record
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # console summary, lined up against the paper and the oracle ceiling
    d = metrics["defect_head"]
    print("\n=== Defect head (test split) ===")
    print(f"  {'Ours (torch MLP)':22s} acc {d['accuracy']:.4f}  wF1 {d['weighted_f1']:.4f}  macroF1 {d['macro_f1']:.4f}")
    print(f"  {'Bayes-optimal ceiling':22s} acc {metrics['bayes_optimal_accuracy']:.4f}")
    print(f"  {'Paper (reported)':22s} acc {PAPER_REF['accuracy']:.4f}  wF1 {PAPER_REF['weighted_f1']:.4f}")
    print("\n=== Mechanism heads (accuracy) ===")
    for stage, acc in metrics["mechanism_accuracy"].items():
        print(f"  {stage:22s} {acc:.4f}")
    print(f"\nRisk head MAE: {metrics['risk_mae']:.4f}")

    # abstention summary (only present when --loss is an abstention loss)
    if "abstention" in metrics:
        ab = metrics["abstention"]
        o = ckpt["train_args"]["o"] if args.load else args.o
        print(f"\n=== Abstention (o={o}) ===")
        print(f"  mean abstain prob: {ab['mean_abstain_prob']:.4f}")
        print(f"  abstain wins argmax: {ab['argmax_abstentions']:,}/{ab['n_records']:,} test records")
        for h, v in ab["by_threshold"].items():
            sel = v["selective_accuracy"]
            sel_s = f"{sel:.4f}" if sel is not None else "n/a"
            print(f"  h={h}: coverage {v['coverage']:.4f}  "
                  f"abstained {v['n_abstained']:,}  selective_acc {sel_s}")
        print("  mechanism mean abstain prob: " +
              "  ".join(f"{s} {v:.4f}" for s, v in ab["mech_mean_abstain_prob"].items()))

    print(f"Wrote {args.out}")
