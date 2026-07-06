"""Selective-classification analysis on the cluster sweep -- the toy_example plots,
made for the real results, off the SAVED checkpoints (no retraining).

Answers the advisor's question head-on: for a single abstention model, as you lower
coverage (reject more), does accepted error drop meaningfully below the plain-CE
baseline -- and does that improvement grow with data complexity? If not, abstaining
is not helping.

For each dataset it draws, sweeping the REJECTION THRESHOLD finely on one model:
  1. Selective-risk curve  -- accepted error vs coverage, for
       (a) the abstention model (reject by its reservation prob r), and
       (b) the cascade model with confidence thresholding (reject low max-prob) --
           the honest baseline: does the LEARNED reject head beat plain confidence?
     against the CE full-coverage error and the Bayes-error floor.
  2. Margin histogram -- accepted vs rejected top1-top2 softmax margin (the toy's
     "rejected points are more ambiguous" check).
It also prints every value, and a gain-vs-difficulty summary.

Run ON THE CLUSTER (needs results/cluster/*.pt + data/cluster/*.csv):
    python cluster/selective_analysis.py
    python cluster/selective_analysis.py --o 2 --seed 0 --coverage 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import mlp   # load_model, predict, spec helpers (self-contained)

ROOT = Path(__file__).resolve().parent.parent
SPEC = "domain/smt_paper.yaml"
FIGDIR = ROOT / "figs"


# --------------------------------------------------------------------------- #
# Checkpoint lookup via the manifest
# --------------------------------------------------------------------------- #

def load_manifest() -> dict:
    return json.loads((ROOT / "cluster" / "manifest.json").read_text())


def find_run(manifest: dict, dataset: str, loss: str, o, seed: int) -> dict | None:
    """The manifest run matching (dataset, loss, o, seed), or None."""
    for r in manifest["runs"]:
        if (r["dataset"] == dataset and r["loss"] == loss and r["seed"] == seed
                and (r["o"] == o if loss == "abstention" else True)):
            return r
    return None


# --------------------------------------------------------------------------- #
# Inference on a saved checkpoint (mirrors mlp.evaluate's standardization)
# --------------------------------------------------------------------------- #

def infer(run: dict, df: pd.DataFrame, spec: dict, device: str = "cpu") -> dict:
    """Test-split predictions for one checkpoint: true labels, argmax over the REAL
    defect classes, per-class real probs, and (abstention only) the reservation r."""
    model, ckpt = mlp.load_model(str(ROOT / run["model"]), device=device)
    ids, names = mlp.param_ids(spec), mlp.defect_names(spec)
    te = np.where(df["split"].to_numpy() == "test")[0]
    mu, sd = ckpt["standardize"]["mu"], ckpt["standardize"]["sd"]
    Xte = ((df.iloc[te][ids].to_numpy(np.float32) - mu) / sd).astype(np.float32)
    y = df.iloc[te]["defect_label"].map({n: i for i, n in enumerate(names)}).to_numpy()
    pred = model.predict(torch.from_numpy(Xte).to(device))
    real = pred["defect_prob"].cpu().numpy()                 # real classes (sum<=1)
    out = {"y": y, "argmax": pred["defect_argmax"].cpu().numpy(),
           "real_probs": real, "abstain": bool(getattr(model, "abstain", False))}
    if out["abstain"]:
        out["r"] = pred["abstain_prob"].cpu().numpy()        # reservation prob
    return out


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #

COVERAGES = np.linspace(0.05, 1.0, 40)


def risk_coverage(score: np.ndarray, correct: np.ndarray,
                  covs: np.ndarray = COVERAGES) -> np.ndarray:
    """Rank-based selective risk: accept the most-confident `cov` fraction (highest
    score) and return accepted error at each coverage. Higher score = keep first."""
    order = np.argsort(-score)
    n = len(score)
    errs = []
    for c in covs:
        k = max(1, int(round(c * n)))
        errs.append(1.0 - correct[order[:k]].mean())
    return np.array(errs)


def margin(real_probs: np.ndarray) -> np.ndarray:
    """top1 - top2 over the real-class probabilities (the toy's confidence proxy)."""
    top2 = np.sort(real_probs, axis=1)[:, -2:]
    return top2[:, -1] - top2[:, -2]


def bayes_error(df: pd.DataFrame, names: list[str]) -> float:
    """Irreducible error on the test split = E[1 - max_y p(y|x)] from the posterior."""
    te = df["split"].to_numpy() == "test"
    post = df.loc[te, [f"p_{n}" for n in names]].to_numpy()
    return float((1.0 - post.max(1)).mean())


def err_at(covs: np.ndarray, errs: np.ndarray, target: float) -> float:
    """Accepted error at the coverage closest to `target`."""
    return float(errs[int(np.argmin(np.abs(covs - target)))])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="toy-style selective analysis on the sweep")
    ap.add_argument("--o", type=float, default=2.0, help="abstention payoff to analyse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coverage", type=float, default=0.8,
                    help="coverage at which to report the abstention gain")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    spec = mlp.load_spec(SPEC)
    names = mlp.defect_names(spec)
    manifest = load_manifest()
    datasets = list(manifest["datasets"])
    FIGDIR.mkdir(exist_ok=True)

    rows, per_ds = [], {}
    for ds in datasets:
        cas = find_run(manifest, ds, "cascade", None, args.seed)
        abst = find_run(manifest, ds, "abstention", args.o, args.seed)
        csv = ROOT / "data" / "cluster" / f"{ds}.csv"
        if not (cas and abst and csv.exists()
                and (ROOT / cas["model"]).exists() and (ROOT / abst["model"]).exists()):
            print(f"[skip] {ds}: missing checkpoint or data"); continue
        df = pd.read_csv(csv)
        bayes = bayes_error(df, names)

        ic = infer(cas, df, spec, args.device)
        ia = infer(abst, df, spec, args.device)
        ce_err = 1.0 - (ic["argmax"] == ic["y"]).mean()          # CE full-coverage error

        correct_a = (ia["argmax"] == ia["y"]).astype(float)
        # abstention: keep lowest reservation first  ->  score = -r
        abst_err = risk_coverage(-ia["r"], correct_a)
        # cascade confidence baseline: keep highest max-prob first
        correct_c = (ic["argmax"] == ic["y"]).astype(float)
        conf_err = risk_coverage(ic["real_probs"].max(1), correct_c)

        per_ds[ds] = dict(bayes=bayes, ce_err=float(ce_err),
                          abst_err=abst_err, conf_err=conf_err, ia=ia)
        a_at = err_at(COVERAGES, abst_err, args.coverage)
        c_at = err_at(COVERAGES, conf_err, args.coverage)
        rows.append(dict(dataset=ds, bayes_err=bayes, ce_full_err=float(ce_err),
                         abst_err_at=a_at, conf_err_at=c_at,
                         gain_vs_ce=float(ce_err) - a_at,
                         gain_vs_conf=c_at - a_at))

    if not rows:
        print("no datasets analysable -- run this on the cluster where the .pt + .csv live")
        return

    table = pd.DataFrame(rows).sort_values("bayes_err").reset_index(drop=True)
    order = list(table["dataset"])

    # ---- print all the values ----
    pd.set_option("display.width", 160)
    print(f"\n=== Selective classification @ coverage~{args.coverage:.2f}  "
          f"(abstention o={args.o:g}, seed {args.seed}); errors, ordered by difficulty ===")
    print(table.round(4).to_string(index=False))
    print("\ngain_vs_ce   = CE full-coverage error  - abstention accepted error   (>0: abstaining helps vs never rejecting)")
    print("gain_vs_conf = cascade confidence error - abstention accepted error   (>0: LEARNED reject beats plain confidence thresholding)")

    # ---- figure 1: selective-risk small multiples, ordered by difficulty ----
    ncol = min(5, len(order)); nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             squeeze=False, constrained_layout=True)
    for k, ds in enumerate(order):
        ax = axes[k // ncol][k % ncol]; d = per_ds[ds]
        ax.plot(COVERAGES, d["abst_err"], "-", color="#e6550d", lw=2, label="abstention")
        ax.plot(COVERAGES, d["conf_err"], "--", color="#2c7fb8", lw=1.8, label="cascade confidence")
        ax.axhline(d["ce_err"], color="#999", ls=":", lw=1.4, label="CE full coverage")
        ax.axhline(d["bayes"], color="k", ls="-", lw=1.0, alpha=0.6, label="Bayes floor")
        ax.set_title(f"{ds}\\n(Bayes err {d['bayes']:.3f})", fontsize=10)
        ax.set_xlabel("coverage"); ax.set_ylabel("accepted error")
        ax.grid(alpha=0.25)
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Selective-risk curves (reject-threshold swept), ordered easy -> hard",
                 fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_selective_risk.{ext}", dpi=200)
    print(f"\nsaved figs/fig_selective_risk.png (+ .pdf)")

    # ---- figure 2: does abstention help MORE as difficulty rises? ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(table["bayes_err"], table["gain_vs_ce"], "o-", color="#e6550d",
            label="vs CE full coverage")
    ax.plot(table["bayes_err"], table["gain_vs_conf"], "s--", color="#2c7fb8",
            label="vs cascade confidence thresholding")
    for _, r in table.iterrows():
        ax.annotate(r["dataset"], (r["bayes_err"], r["gain_vs_ce"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Bayes error  (data complexity ->)")
    ax.set_ylabel(f"accuracy gained by abstaining @ coverage~{args.coverage:.2f}")
    ax.set_title("Does abstaining help more as the problem gets harder?", fontweight="bold")
    ax.grid(alpha=0.25); ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_abstention_gain_vs_difficulty.{ext}", dpi=200)
    print("saved figs/fig_abstention_gain_vs_difficulty.png (+ .pdf)")

    # ---- figure 3: margin histogram (accepted vs rejected) for easy vs hard ----
    probe = [order[0], order[len(order) // 2], order[-1]]
    fig, axes = plt.subplots(1, len(probe), figsize=(5 * len(probe), 4.2),
                             squeeze=False, constrained_layout=True)
    for j, ds in enumerate(probe):
        ia = per_ds[ds]["ia"]; ax = axes[0][j]
        marg = margin(ia["real_probs"])
        rej = ia["r"] >= 0.5                                  # default reject rule
        acc = ~rej
        if acc.sum() and rej.sum():
            bins = np.linspace(0, max(marg.max(), 1e-6), 26)
            ax.hist(marg[acc], bins=bins, density=True, alpha=0.6, color="#2c7fb8", label="accepted")
            ax.hist(marg[rej], bins=bins, density=True, alpha=0.6, color="#e6550d", label="rejected")
            ax.legend(fontsize=9)
        ax.set_title(f"{ds}", fontsize=11)
        ax.set_xlabel("top1 - top2 margin"); ax.set_ylabel("density"); ax.grid(alpha=0.25)
    fig.suptitle("Rejected boards have smaller margin (more ambiguous)", fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_margin_hist.{ext}", dpi=200)
    print("saved figs/fig_margin_hist.png (+ .pdf)")


if __name__ == "__main__":
    main()
