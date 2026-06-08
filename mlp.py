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

MODEL_VERSION = "mlp-v0.1"           # stamped into outputs (provenance)
HIDDEN = (256, 256)                  # shared-trunk widths
DROPOUT = 0.1                        # trunk dropout
LAMBDA_D, LAMBDA_M, LAMBDA_R = 1.0, 1.0, 1.0   # loss weights per head type
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 128, 40, 6  # optimizer + early-stop
CLASS_WEIGHT_MODE = "none"

# Paper's reported defect-head numbers (Section V-A), for comparison.
PAPER_REF = {"accuracy": 0.9500, "weighted_f1": 0.9536}


# Model

class MultiHeadMLP(nn.Module):
    """Shared trunk h = f(x) with a defect head, per-stage mechanism heads, and
    P independent per-parameter sigmoid risk heads (Eq. 3) -- implemented as one
    fused nn.Linear(d, n_params) + elementwise sigmoid, equivalent to P separate
    sigmoid heads, NOT a softmax across parameters. Sizes come from the spec."""

    def __init__(self, n_features: int, n_defects: int, mech_sizes: list[int],
                 n_params: int, hidden: tuple[int, ...] = HIDDEN,
                 dropout: float = DROPOUT) -> None:
        super().__init__()

        # build the shared trunk: Linear -> ReLU -> Dropout, stacked
        layers: list[nn.Module] = []
        d = n_features
        for w in hidden:
            layers += [nn.Linear(d, w), nn.ReLU(), nn.Dropout(dropout)]
            d = w
        self.trunk = nn.Sequential(*layers)

        # one head per task; mechanism heads are one-per-stage
        self.defect_head = nn.Linear(d, n_defects)
        self.mech_heads = nn.ModuleList(nn.Linear(d, k) for k in mech_sizes)
        self.risk_head = nn.Linear(d, n_params)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        # shared latent, then each head reads from it
        h = self.trunk(x)
        return {
            "defect": self.defect_head(h),                      # (B, n_defects) logits
            "mech": [head(h) for head in self.mech_heads],      # list of (B, k_s) logits
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
        defect_prob = torch.softmax(out["defect"], dim=1)        # p(y | x)
        mech_prob = [torch.softmax(m, dim=1) for m in out["mech"]]  # p(m_s | x)
        risk_prob = torch.sigmoid(out["risk"])                   # independent sigmoid
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
    subclass with nn.Parameter head weights trains with no plumbing change."""

    def __init__(self, class_weight: torch.Tensor | None = None,
                 lam_d: float = LAMBDA_D, lam_m: float = LAMBDA_M,
                 lam_r: float = LAMBDA_R) -> None:
        super().__init__()

        # class_weight is a fixed tensor (not learned) -> register as a buffer
        self.register_buffer("class_weight", class_weight)
        self.lam_d, self.lam_m, self.lam_r = lam_d, lam_m, lam_r

    # per-head TERMS (override exactly one in a subclass to experiment)

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


# loss registry: add a subclass here and pick it with --loss
LOSSES = {"multihead": MultiHeadLoss}


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
        y_mech[:, s] = df[f"{stage}_mechanism_label"].map(m_idx).to_numpy()

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
          seed: int = 0) -> tuple[MultiHeadMLP, dict]:
    """Algorithm — Train the multi-head MLP with early stopping on val loss.

    Input: spec, the generated dataframe, device, epoch budget, weighting mode,
           loss class, trunk hidden/dropout, and a seed for the shuffle.
    Return: the best model and the encoded-data bundle (with the train stats).
    """

    enc = _encode(df, spec)

    # split indices come straight from the leak-free batch-grouped split column
    sp = df["split"].to_numpy()
    tr, va = np.where(sp == "train")[0], np.where(sp == "val")[0]

    # standardize using train stats only; STORE the stats so eval reuses them
    # (test rows enc["X"][te] are left RAW on purpose)
    stdz, enc["mu"], enc["sd"] = _standardize(enc["X"][tr], enc["X"][va])
    enc["X"][tr], enc["X"][va] = stdz

    # build the model with sizes taken from the spec
    model = MultiHeadMLP(
        n_features=enc["X"].shape[1], n_defects=len(enc["names"]),
        mech_sizes=[len(spec["mechanisms"][s]) for s in enc["stages"]],
        n_params=enc["y_risk"].shape[1], hidden=hidden, dropout=dropout).to(device)

    # optional class weighting + Adam (the paper-default optimizer)
    cw = _class_weights(enc["y_def"][tr], len(enc["names"]), class_weight_mode)
    loss_fn = loss_cls(class_weight=cw.to(device) if cw is not None else None)
    loss_fn = loss_fn.to(device)
    # loss_fn.parameters() is included on purpose: a learnable-weight loss
    # subclass (nn.Parameter head weights) then trains with no change to train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=LR)

    # seed the shuffle explicitly so a run is reproducible by its own seed,
    # not by the global-RNG ordering of everything that ran before it
    g = torch.Generator()
    g.manual_seed(seed)
    tr_dl = _loader(enc, tr, BATCH, shuffle=True, generator=g)
    va_dl = _loader(enc, va, BATCH, shuffle=False)

    # early stopping: keep the weights with the lowest validation loss
    best_val, best_state, waited = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yd, ym, yr in tr_dl:
            xb, yd, ym, yr = xb.to(device), yd.to(device), ym.to(device), yr.to(device)
            opt.zero_grad()                 # clear last step's gradients
            out = model(xb)                 # forward pass
            loss = loss_fn(out, yd, ym, yr) # composite loss
            loss.backward()                 # autograd fills every gradient
            opt.step()                      # update weights

        # validation loss (no gradients) drives early stopping
        val = _epoch_loss(model, loss_fn, va_dl, device)
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


def evaluate(model: MultiHeadMLP, enc: dict, df: pd.DataFrame,
             spec: dict, device: str = "cpu") -> dict:
    """Algorithm — Test-split metrics, comparable to the paper and Bayes ceiling.

    Input: trained model, encoded data, dataframe, spec, device.
    Return: defect accuracy/F1, per-stage mechanism accuracy, risk MAE.
    """

    names, stages = enc["names"], enc["stages"]
    te = np.where(df["split"].to_numpy() == "test")[0]

    # standardize the still-raw test rows with the STORED train stats
    # (no leakage, no recompute -- enc["X"][te] was never touched in train())
    Xte = ((enc["X"][te] - enc["mu"]) / enc["sd"]).astype(np.float32)

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

    return {
        "model_version": MODEL_VERSION,
        "defect_head": defect,
        "bayes_optimal_accuracy": bayes_acc,
        "paper_reference": dict(PAPER_REF),
        "mechanism_accuracy": mech,
        "risk_mae": risk_mae,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Basic multi-head MLP (SMT)")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple GPU) | cuda")
    ap.add_argument("--class-weight", default=CLASS_WEIGHT_MODE,
                    choices=["none", "sqrt", "inverse"],
                    help="defect-head class weighting (accuracy vs minority recall)")
    ap.add_argument("--loss", default="multihead", choices=list(LOSSES),
                    help="which loss to train with")
    ap.add_argument("--hidden", default=",".join(map(str, HIDDEN)),
                    help="comma-separated trunk widths, e.g. 256,256")
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mlp_metrics.json")
    args = ap.parse_args()

    # pin every RNG source so a run is reproducible
    seed_everything(args.seed)

    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)

    print(f"Training {MODEL_VERSION} on {len(df):,} records (device={args.device})")
    hidden = tuple(int(w) for w in args.hidden.split(","))
    model, enc = train(spec, df, device=args.device, epochs=args.epochs,
                       class_weight_mode=args.class_weight,
                       loss_cls=LOSSES[args.loss], hidden=hidden,
                       dropout=args.dropout, seed=args.seed)
    metrics = evaluate(model, enc, df, spec, device=args.device)

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
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
