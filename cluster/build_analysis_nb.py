"""Build cluster_analysis.ipynb (the sweep comparison notebook) from cell sources.

Run from the repo root:  python3 cluster/build_analysis_nb.py
Emits cluster_analysis.ipynb in the repo root. Stdlib only (json).

The notebook loads cluster/manifest.json + every results/cluster/<run>.json that
training produced, builds one comparison table across both losses and all dataset
variants, plots the trade-offs, and reports the optimal configs per metric.
"""

import json
from pathlib import Path

MD_TITLE = """# Cluster sweep analysis

Compares every model trained by the cluster sweep (`cluster/`) across both loss
functions (`cascade`, `abstention`) and all dataset variants (different class
balances + difficulties). Loads the trained checkpoints' metrics and finds the
optimal configurations.

Prereq: run the sweep first (`bash cluster/submit.sh` on SLURM, or
`bash cluster/run_local.sh` locally), then run this notebook from the repo root.
"""

LOAD = '''from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd

# find the repo root by walking up until cluster/manifest.json appears
root = Path.cwd()
while not (root / "cluster" / "manifest.json").exists() and root != root.parent:
    root = root.parent
manifest = json.loads((root / "cluster" / "manifest.json").read_text())
print("runs in matrix:", len(manifest["runs"]))


def load_rows():
    """One flat row per run that actually produced a metrics file."""
    rows, missing = [], []
    for r in manifest["runs"]:
        p = root / r["metrics"]
        if not p.exists():
            missing.append(r["run_id"]); continue
        m = json.loads(p.read_text())
        d, mech = m["defect_head"], m["mechanism_accuracy"]
        row = {
            "run_id": r["run_id"], "dataset": r["dataset"], "loss": r["loss"],
            "o": r["o"], "seed": r["seed"],
            "defect_acc": d["accuracy"], "defect_wf1": d["weighted_f1"],
            "defect_macrof1": d["macro_f1"],
            "bayes": m["bayes_optimal_accuracy"],
            "gap_to_bayes": m["bayes_optimal_accuracy"] - d["accuracy"],
            "mech_joint": mech["joint_accuracy"],
            "mech_printing": mech.get("stage_printing"),
            "mech_reflow": mech.get("stage_reflow"),
            "risk_mae": m["risk_mae"],
        }
        ab = m.get("abstention")
        if ab:
            row["mean_abstain"] = ab["mean_abstain_prob"]
            half = ab["by_threshold"].get("0.5", {})
            row["coverage@0.5"] = half.get("coverage")
            row["selective_acc@0.5"] = half.get("selective_accuracy")
        rows.append(row)
    if missing:
        print(f"WARNING: {len(missing)} runs have no metrics yet (still training?):")
        print("  " + ", ".join(missing))
    return pd.DataFrame(rows)


df = load_rows()
print(f"loaded {len(df)} runs")
df.sort_values(["dataset", "loss", "o", "seed"]).reset_index(drop=True)
'''

TABLE = '''# Full comparison table, rounded for reading.
cols = ["run_id", "dataset", "loss", "o", "seed", "defect_acc", "defect_macrof1",
        "gap_to_bayes", "mech_joint", "risk_mae", "selective_acc@0.5", "coverage@0.5"]
view = df[[c for c in cols if c in df.columns]].copy()
num = view.select_dtypes("number").columns
view[num] = view[num].round(4)
view.sort_values(["dataset", "loss", "o", "seed"]).reset_index(drop=True)
'''

AGG = '''# Average over seeds -> one row per (dataset, loss, o) config.
keys = ["dataset", "loss", "o"]
metrics = ["defect_acc", "defect_macrof1", "gap_to_bayes", "mech_joint",
           "risk_mae"] + [c for c in ("selective_acc@0.5", "coverage@0.5") if c in df.columns]
agg = (df.groupby(keys, dropna=False)[metrics]
         .agg(["mean", "std"]).round(4))
agg
'''

DEFECT_PLOT = '''# Defect accuracy by dataset, grouped by loss (mean +/- std over seeds), with the
# Bayes ceiling per dataset for reference.
datasets = list(manifest["datasets"])
losses = sorted(df["loss"].unique())
x = np.arange(len(datasets)); w = 0.8 / max(len(losses), 1)
fig, ax = plt.subplots(figsize=(10, 5))
for i, loss in enumerate(losses):
    means, stds = [], []
    for ds in datasets:
        sub = df[(df.dataset == ds) & (df.loss == loss)]
        means.append(sub["defect_acc"].mean()); stds.append(sub["defect_acc"].std())
    ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=loss)
# Bayes ceiling (same data regardless of loss) as a marker per dataset
bayes = [df[df.dataset == ds]["bayes"].mean() for ds in datasets]
ax.plot(x + w * (len(losses) - 1) / 2, bayes, "kD", ms=7, label="Bayes ceiling")
ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=20, ha="right")
ax.set_ylabel("defect accuracy"); ax.set_ylim(0.7, 1.0)
ax.set_title("Defect accuracy across dataset variants x loss")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); plt.show()
'''

GAP_PLOT = '''# How far each config sits below its Bayes ceiling (smaller = better).
fig, ax = plt.subplots(figsize=(10, 5))
for i, loss in enumerate(losses):
    means = [df[(df.dataset == ds) & (df.loss == loss)]["gap_to_bayes"].mean()
             for ds in datasets]
    ax.bar(x + i * w, means, w, label=loss)
ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=20, ha="right")
ax.set_ylabel("Bayes ceiling - defect accuracy"); ax.set_title("Gap to the Bayes ceiling (lower is better)")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); plt.show()
'''

RISK_MECH_PLOT = '''# Mechanism (joint) accuracy and risk MAE side by side, by dataset x loss.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, title, lo, hi in [
        (axes[0], "mech_joint", "Joint mechanism accuracy", 0.7, 1.0),
        (axes[1], "risk_mae", "Risk head MAE (lower better)", 0.0, None)]:
    for i, loss in enumerate(losses):
        means = [df[(df.dataset == ds) & (df.loss == loss)][metric].mean() for ds in datasets]
        ax.bar(x + i * w, means, w, label=loss)
    ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    if lo is not None:
        ax.set_ylim(lo, hi)
axes[0].set_ylabel("accuracy"); axes[1].set_ylabel("MAE"); axes[0].legend(frameon=False)
fig.tight_layout(); plt.show()
'''

ABST_PLOT = '''# Abstention only: selective accuracy @0.5 vs coverage @0.5, payoff o encoded by
# marker; one point per (dataset, o, seed). Up-and-right is better.
ab = df[df.loss == "abstention"].dropna(subset=["selective_acc@0.5", "coverage@0.5"])
if len(ab):
    fig, ax = plt.subplots(figsize=(8, 6))
    for o_val, mk in zip(sorted(ab["o"].unique()), ["o", "s", "^", "D"]):
        sub = ab[ab.o == o_val]
        ax.scatter(sub["coverage@0.5"], sub["selective_acc@0.5"], s=70, marker=mk,
                   label=f"o={o_val:g}", alpha=0.8)
        for _, rr in sub.iterrows():
            ax.annotate(rr["dataset"], (rr["coverage@0.5"], rr["selective_acc@0.5"]),
                        fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("coverage @ r<0.5 (fraction predicted)")
    ax.set_ylabel("selective accuracy @ r<0.5")
    ax.set_title("Abstention operating points (higher payoff o -> more coverage)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); plt.show()
else:
    print("no abstention runs with metrics yet")
'''

OPTIMAL = '''# The optimal configs. "Best" = averaged over seeds so we do not chase noise.
seed_keys = ["dataset", "loss", "o"]
mean_over_seeds = df.groupby(seed_keys, dropna=False).mean(numeric_only=True).reset_index()

def best(metric, maximize=True):
    s = mean_over_seeds.sort_values(metric, ascending=not maximize).iloc[0]
    o = "-" if pd.isna(s["o"]) else f"{s['o']:g}"
    return f"{s['dataset']} / {s['loss']} / o={o}  ->  {metric}={s[metric]:.4f}"

print("=== Most optimal (mean over seeds) ===")
print("highest defect accuracy :", best("defect_acc"))
print("smallest gap to Bayes   :", best("gap_to_bayes", maximize=False))
print("best minority macro-F1  :", best("defect_macrof1"))
print("highest joint mechanism :", best("mech_joint"))
print("lowest risk MAE         :", best("risk_mae", maximize=False))
if "selective_acc@0.5" in mean_over_seeds.columns and mean_over_seeds["selective_acc@0.5"].notna().any():
    ab = mean_over_seeds.dropna(subset=["selective_acc@0.5"])
    s = ab.sort_values("selective_acc@0.5", ascending=False).iloc[0]
    print(f"best selective acc@0.5  : {s['dataset']} / abstention / o={s['o']:g}  ->  "
          f"{s['selective_acc@0.5']:.4f} at coverage {s['coverage@0.5']:.3f}")

print("\\n=== Best config per dataset (by gap to Bayes) ===")
for ds in manifest["datasets"]:
    sub = mean_over_seeds[mean_over_seeds.dataset == ds]
    if not len(sub):
        continue
    win = sub.sort_values("gap_to_bayes").iloc[0]
    o = "-" if pd.isna(win["o"]) else f"{win['o']:g}"
    print(f"  {ds:14s} -> {win['loss']:10s} o={o:3s}  "
          f"defect {win['defect_acc']:.4f}  gap {win['gap_to_bayes']:.4f}  "
          f"mech {win['mech_joint']:.4f}  riskMAE {win['risk_mae']:.4f}")
'''

CODE_CELLS = [LOAD, TABLE, AGG, DEFECT_PLOT, GAP_PLOT, RISK_MECH_PLOT, ABST_PLOT, OPTIMAL]


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def main():
    nb = {
        "cells": [md(MD_TITLE)] + [code(c) for c in CODE_CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "cluster_analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}  ({len(CODE_CELLS)} code cells)")


if __name__ == "__main__":
    main()
