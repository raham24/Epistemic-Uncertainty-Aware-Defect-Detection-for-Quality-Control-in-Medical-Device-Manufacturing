"""Build cluster_analysis.ipynb (the sweep comparison notebook) from cell sources.

Run from the repo root:  python3 cluster/build_analysis_nb.py
Emits cluster_analysis.ipynb in the repo root. Stdlib only (json).

The notebook loads cluster/manifest.json + every results/cluster/<run>.json that
training produced, then analyses the sweep in two layers:

  * core grid   -- both losses x every dataset variant (the headline comparison),
                   each measured against its own Bayes ceiling.
  * studies     -- one-factor-at-a-time sweeps (payoff o, trunk capacity, dropout,
                   learning rate, class weighting), each isolating one axis.

Every run carries its study tags + full hyperparameters in the manifest, so the
notebook can slice by study and report the optimal setting per axis.
"""

import json
from pathlib import Path

TITLE = """# Cluster sweep analysis

Compares every model trained by the cluster sweep (`cluster/`). The sweep is a
**core grid** (both losses x all dataset variants, each vs its Bayes ceiling) plus
focused **studies** that vary one hyperparameter at a time:

| study | axis | on |
|---|---|---|
| `core` | loss x dataset | all datasets |
| `payoff` | abstention `o` | baseline / hard / imbalanced |
| `capacity` | trunk width+depth | baseline / hard |
| `dropout` | dropout | baseline / hard |
| `lr` | Adam learning rate | baseline / hard |
| `class_weight` | defect reweighting | imbalanced / baseline |

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
    """One flat row per run that actually produced a metrics file. Hyperparameters
    + study tags come from the manifest; scores from the run's metrics JSON."""
    rows, missing = [], []
    for r in manifest["runs"]:
        p = root / r["metrics"]
        if not p.exists():
            missing.append(r["run_id"]); continue
        m = json.loads(p.read_text())
        d, mech = m["defect_head"], m["mechanism_accuracy"]
        row = {
            "run_id": r["run_id"], "dataset": r["dataset"], "loss": r["loss"],
            "o": r["o"], "seed": r["seed"], "hidden": r.get("hidden"),
            "dropout": r.get("dropout"), "class_weight": r.get("class_weight"),
            "lr": r.get("lr"), "epochs": r.get("epochs"),
            "studies": r.get("studies", []),
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
        print("  " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    return pd.DataFrame(rows)


df = load_rows()
print(f"loaded {len(df)} of {len(manifest['runs'])} runs")


def study(name):
    """Rows tagged with a study (core, payoff, capacity, dropout, lr, class_weight)."""
    if not len(df) or "studies" not in df.columns:
        return df.iloc[0:0]
    return df[df["studies"].apply(lambda s: name in (s or []))]


def seed_mean(frame, keys, metrics):
    """Average over seeds -> one row per config (keeps NaN-keyed groups, e.g. o=None)."""
    metrics = [m for m in metrics if m in frame.columns]
    return frame.groupby(keys, dropna=False)[metrics].mean().reset_index()


def core_grid():
    """(core rows, dataset order, losses, bar x-positions, bar width)."""
    core = study("core")
    datasets = list(manifest["datasets"])
    losses = sorted(core["loss"].unique()) if len(core) else []
    x = np.arange(len(datasets)); w = 0.8 / max(len(losses), 1)
    return core, datasets, losses, x, w


print("studies present:", sorted({s for ss in df.get("studies", []) for s in ss})
      if len(df) else [])
'''

CORE_TABLE = '''# core grid: full per-run comparison table (rounded for reading).
core = study("core")
cols = ["run_id", "dataset", "loss", "o", "seed", "defect_acc", "defect_macrof1",
        "gap_to_bayes", "mech_joint", "risk_mae", "selective_acc@0.5", "coverage@0.5"]
view = core[[c for c in cols if c in core.columns]].copy()
num = view.select_dtypes("number").columns
view[num] = view[num].round(4)
view.sort_values(["dataset", "loss", "o", "seed"]).reset_index(drop=True)
'''

CORE_AGG = '''# core grid averaged over seeds -> one row per (dataset, loss, o), mean +/- std.
core = study("core")
keys = ["dataset", "loss", "o"]
metrics = ["defect_acc", "defect_macrof1", "gap_to_bayes", "mech_joint", "risk_mae",
           "selective_acc@0.5", "coverage@0.5"]
(core.groupby(keys, dropna=False)[[m for m in metrics if m in core.columns]]
     .agg(["mean", "std"]).round(4))
'''

DEFECT_PLOT = '''# core: defect accuracy by dataset, grouped by loss (mean +/- std over seeds),
# with the Bayes ceiling per dataset for reference.
core, datasets, losses, x, w = core_grid()
fig, ax = plt.subplots(figsize=(11, 5))
for i, loss in enumerate(losses):
    means = [core[(core.dataset == ds) & (core.loss == loss)]["defect_acc"].mean() for ds in datasets]
    stds  = [core[(core.dataset == ds) & (core.loss == loss)]["defect_acc"].std()  for ds in datasets]
    ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=loss)
bayes = [core[core.dataset == ds]["bayes"].mean() for ds in datasets]
ax.plot(x + w * (len(losses) - 1) / 2, bayes, "kD", ms=7, label="Bayes ceiling")
ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=25, ha="right")
ax.set_ylabel("defect accuracy"); ax.set_ylim(0.5, 1.0)
ax.set_title("core: defect accuracy across dataset variants x loss")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); plt.show()
'''

GAP_PLOT = '''# core: how far each loss sits below its Bayes ceiling per dataset (lower = better).
core, datasets, losses, x, w = core_grid()
fig, ax = plt.subplots(figsize=(11, 5))
for i, loss in enumerate(losses):
    means = [core[(core.dataset == ds) & (core.loss == loss)]["gap_to_bayes"].mean()
             for ds in datasets]
    ax.bar(x + i * w, means, w, label=loss)
ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=25, ha="right")
ax.set_ylabel("Bayes ceiling - defect accuracy")
ax.set_title("core: gap to the Bayes ceiling (lower is better)")
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); plt.show()
'''

RISK_MECH_PLOT = '''# core: joint-mechanism accuracy and risk MAE side by side, by dataset x loss.
core, datasets, losses, x, w = core_grid()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, title, lo, hi in [
        (axes[0], "mech_joint", "joint mechanism accuracy", 0.6, 1.0),
        (axes[1], "risk_mae", "risk head MAE (lower better)", 0.0, None)]:
    for i, loss in enumerate(losses):
        means = [core[(core.dataset == ds) & (core.loss == loss)][metric].mean() for ds in datasets]
        ax.bar(x + i * w, means, w, label=loss)
    ax.set_xticks(x + w * (len(losses) - 1) / 2); ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    if lo is not None:
        ax.set_ylim(lo, hi)
axes[0].set_ylabel("accuracy"); axes[1].set_ylabel("MAE"); axes[0].legend(frameon=False)
fig.tight_layout(); plt.show()
'''

PAYOFF_PLOT = '''# payoff study: how the abstention payoff o trades coverage for selective accuracy.
pay = study("payoff")
if len(pay):
    sm = seed_mean(pay, ["dataset", "o"], ["selective_acc@0.5", "coverage@0.5"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ds in sorted(pay["dataset"].unique()):
        s = sm[sm.dataset == ds].sort_values("o")
        axes[0].plot(s["o"], s["selective_acc@0.5"], "o-", label=ds)
        axes[1].plot(s["o"], s["coverage@0.5"], "o-", label=ds)
    axes[0].set_title("selective accuracy @ r<0.5 vs payoff o")
    axes[0].set_ylabel("selective accuracy")
    axes[1].set_title("coverage @ r<0.5 vs payoff o (higher o -> predict more)")
    axes[1].set_ylabel("coverage")
    for a in axes:
        a.set_xlabel("payoff o")
        a.grid(alpha=0.25); a.legend(frameon=False)
    fig.tight_layout(); plt.show()
else:
    print("no payoff-study runs with metrics yet")
'''

CAPACITY_PLOT = '''# capacity study: defect accuracy and gap-to-Bayes vs trunk width/depth.
cap = study("capacity")
if len(cap):
    order = [h for h in ["128,128", "256,256", "512,512", "256,256,256", "512,512,256"]
             if h in set(cap["hidden"])]
    pos = {h: i for i, h in enumerate(order)}
    sm = seed_mean(cap, ["dataset", "hidden"], ["defect_acc", "gap_to_bayes"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ds in sorted(cap["dataset"].unique()):
        s = sm[sm.dataset == ds].copy()
        s["xi"] = s["hidden"].map(pos); s = s.sort_values("xi")
        axes[0].plot(s["xi"], s["defect_acc"], "o-", label=ds)
        axes[1].plot(s["xi"], s["gap_to_bayes"], "o-", label=ds)
    for a, ttl, yl in [(axes[0], "defect accuracy vs trunk size", "defect accuracy"),
                       (axes[1], "gap to Bayes vs trunk size (lower better)", "gap to Bayes")]:
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=20, ha="right")
        a.set_title(ttl); a.set_ylabel(yl); a.grid(alpha=0.25); a.legend(frameon=False)
    fig.tight_layout(); plt.show()
else:
    print("no capacity-study runs with metrics yet")
'''

REG_OPT_PLOT = '''# dropout + learning-rate studies: gap-to-Bayes vs each axis (lower is better).
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
dro = study("dropout")
if len(dro):
    sm = seed_mean(dro, ["dataset", "dropout"], ["gap_to_bayes"])
    for ds in sorted(dro["dataset"].unique()):
        s = sm[sm.dataset == ds].sort_values("dropout")
        axes[0].plot(s["dropout"], s["gap_to_bayes"], "o-", label=ds)
    axes[0].set_xlabel("dropout"); axes[0].set_ylabel("gap to Bayes")
    axes[0].set_title("dropout sweep")
lr = study("lr")
if len(lr):
    sm = seed_mean(lr, ["dataset", "lr"], ["gap_to_bayes"])
    for ds in sorted(lr["dataset"].unique()):
        s = sm[sm.dataset == ds].sort_values("lr")
        axes[1].plot(s["lr"], s["gap_to_bayes"], "o-", label=ds)
    axes[1].set_xscale("log"); axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("gap to Bayes"); axes[1].set_title("learning-rate sweep")
for a in axes:
    a.grid(alpha=0.25)
    if a.get_legend_handles_labels()[0]:
        a.legend(frameon=False)
fig.tight_layout(); plt.show()
'''

CW_PLOT = '''# class_weight study: minority-sensitive macro-F1 and overall accuracy by mode.
cw = study("class_weight")
if len(cw):
    modes = [m for m in ["none", "sqrt", "inverse"] if m in set(cw["class_weight"])]
    dss = sorted(cw["dataset"].unique())
    sm = seed_mean(cw, ["dataset", "class_weight"], ["defect_acc", "defect_macrof1"])
    x = np.arange(len(modes)); w = 0.8 / max(len(dss), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for j, metric, ttl in [(0, "defect_macrof1", "macro-F1 (minority-sensitive)"),
                           (1, "defect_acc", "defect accuracy")]:
        for i, ds in enumerate(dss):
            vals = [sm[(sm.dataset == ds) & (sm.class_weight == mode)][metric].mean()
                    for mode in modes]
            axes[j].bar(x + i * w, vals, w, label=ds)
        axes[j].set_xticks(x + w * (len(dss) - 1) / 2); axes[j].set_xticklabels(modes)
        axes[j].set_title(ttl); axes[j].grid(axis="y", alpha=0.25); axes[j].legend(frameon=False)
    fig.tight_layout(); plt.show()
else:
    print("no class_weight-study runs with metrics yet")
'''

OPTIMAL = '''# Optimal configs. core picks are averaged over seeds so we do not chase noise;
# each study reports the best setting of its own axis per probe dataset.
core = study("core")
mean_over_seeds = core.groupby(["dataset", "loss", "o"], dropna=False).mean(numeric_only=True).reset_index()


def best(metric, maximize=True):
    s = mean_over_seeds.sort_values(metric, ascending=not maximize).iloc[0]
    o = "-" if pd.isna(s["o"]) else f"{s['o']:g}"
    return f"{s['dataset']} / {s['loss']} / o={o}  ->  {metric}={s[metric]:.4f}"


print("=== Optimal core configs (mean over seeds) ===")
print("highest defect accuracy :", best("defect_acc"))
print("smallest gap to Bayes   :", best("gap_to_bayes", maximize=False))
print("best minority macro-F1  :", best("defect_macrof1"))
print("highest joint mechanism :", best("mech_joint"))
print("lowest risk MAE         :", best("risk_mae", maximize=False))

print("\\n=== Best setting per study (probe datasets, mean over seeds) ===")
for name, axis, metric, maximize in [
        ("capacity", "hidden", "gap_to_bayes", False),
        ("dropout", "dropout", "gap_to_bayes", False),
        ("lr", "lr", "gap_to_bayes", False),
        ("class_weight", "class_weight", "defect_macrof1", True),
        ("payoff", "o", "selective_acc@0.5", True)]:
    s = study(name)
    if not len(s) or metric not in s.columns or not s[metric].notna().any():
        print(f"  {name:12s}: (no runs yet)"); continue
    sm = seed_mean(s, ["dataset", axis], [metric]).dropna(subset=[metric])
    for ds in sorted(sm["dataset"].unique()):
        win = sm[sm.dataset == ds].sort_values(metric, ascending=not maximize).iloc[0]
        print(f"  {name:12s} [{ds:11s}] best {axis}={win[axis]!s:11s} -> {metric}={win[metric]:.4f}")
'''

# (markdown header, code) in notebook order
SECTIONS = [
    ("## Load every run\\n\\nLoads the manifest + each run's metrics into one dataframe, with helpers to "
     "slice by study and average over seeds.", LOAD),
    ("## Core grid\\n\\nBoth losses across every dataset variant. First the full table, then averaged "
     "over seeds.", CORE_TABLE),
    (None, CORE_AGG),
    ("### Defect accuracy vs the Bayes ceiling", DEFECT_PLOT),
    ("### Gap to the Bayes ceiling", GAP_PLOT),
    ("### Mechanism accuracy and risk MAE", RISK_MECH_PLOT),
    ("## Payoff study\\n\\nAbstention only: the payoff `o` trades coverage against selective accuracy.",
     PAYOFF_PLOT),
    ("## Capacity study\\n\\nTrunk width/depth (cascade) on the probe datasets.", CAPACITY_PLOT),
    ("## Dropout and learning-rate studies", REG_OPT_PLOT),
    ("## Class-weight study\\n\\nDefect-head reweighting, where minority recall matters most.", CW_PLOT),
    ("## Optimal configurations", OPTIMAL),
]


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def main():
    cells = [md(TITLE)]
    for header, src in SECTIONS:
        if header:
            cells.append(md(header.replace("\\n", "\n")))
        cells.append(code(src))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "cluster_analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    n_code = sum(1 for _, s in SECTIONS)
    print(f"wrote {out}  ({n_code} code cells)")


if __name__ == "__main__":
    main()
