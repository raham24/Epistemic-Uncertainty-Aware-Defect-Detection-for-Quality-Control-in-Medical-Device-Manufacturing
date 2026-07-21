"""Build real_data_analysis.ipynb (the MAUDE abstention-sweep notebook) from cell
sources. Mirrors cluster/build_analysis_nb.py for the synthetic pipeline.

Run from the repo root:  python3 real-data/cluster/build_analysis_nb.py
Emits real-data/real_data_analysis.ipynb. Stdlib only (json).

The notebook loads real-data/cluster/manifest.json + every
real-data/results/cluster/<run>.json that training produced, then analyses the
learned-abstention sweep: forced accuracy, the payoff (o) study, and -- the
headline -- the learned reject vs the confidence-threshold baseline at matched
coverage (does the learned reject add anything). Key figures save to figs/.
"""

import json
from pathlib import Path

TITLE = """# MAUDE learned-abstention sweep analysis

Analyses every model trained by the real-data sweep (`real-data/cluster/`). One
scraped MAUDE dataset (product-problem detection from redacted narratives), a
single-head MLP trained with the learned-abstention term `-log(o*p_y + r)`, swept
over the payoff **o** (plus capacity / dropout / lr studies), replicated across
seeds.

Every run reports two selective-classification operating points on the SAME model:
the **learned reject** (accept when abstain prob `r < h`) and the **confidence
baseline** (accept when `max class prob >= threshold`, the professor's original
rule), both at a matched target coverage. `gain_vs_conf` is their difference -- the
question is whether the learned reject beats plain confidence thresholding.

Paper-ready figures write to `figs/fig_maude_*.{png,pdf}`:

| figure | what it shows |
|---|---|
| `fig_maude_gain_vs_o` | learned-reject minus confidence selective accuracy, vs `o` (headline) |
| `fig_maude_risk_coverage` | selective accuracy vs coverage: learned-reject curve vs confidence baseline |
| `fig_maude_metrics_vs_o` | forced accuracy, selective accuracy, mean abstain prob vs `o` |
| `fig_maude_studies` | capacity / dropout / lr studies |

Prereq: run the sweep first (`bash real-data/cluster/submit.sh`, or `run_local.sh`),
then run this notebook from the repo root.
"""

LOAD = '''from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd

# find the repo root by walking up until the manifest appears
root = Path.cwd()
while not (root / "real-data" / "cluster" / "manifest.json").exists() and root != root.parent:
    root = root.parent
manifest = json.loads((root / "real-data" / "cluster" / "manifest.json").read_text())
print("runs in matrix:", len(manifest["runs"]))
print("base config    :", manifest["base"])
print("seeds          :", manifest["seeds"])


def load_rows():
    """One flat row per run that produced a metrics file. Hyperparameters + study
    tags from the manifest; scores from the run's metrics JSON."""
    rows, missing = [], []
    for r in manifest["runs"]:
        p = root / r["metrics"]
        if not p.exists():
            missing.append(r["run_id"]); continue
        m = json.loads(p.read_text())
        lr_, cb = m["learned_reject"], m["confidence_baseline"]
        rows.append({
            "run_id": r["run_id"], "o": r["o"], "seed": r["seed"],
            "hidden": r.get("hidden"), "dropout": r.get("dropout"), "lr": r.get("lr"),
            "studies": r.get("studies", []),
            "forced_acc": m["forced"]["accuracy"], "forced_f1": m["forced"]["f1"],
            "forced_auc": m["forced"]["roc_auc"], "mean_abstain": m["mean_abstain_prob"],
            "learned_cov": lr_["coverage"], "learned_sel_acc": lr_["selective_accuracy"],
            "conf_cov": cb["coverage"], "conf_sel_acc": cb["selective_accuracy"],
            "gain_vs_conf": m["gain_vs_conf"],
            "_metrics_path": str(p),
        })
    if missing:
        print(f"WARNING: {len(missing)} runs have no metrics yet (still training?):")
        print("  " + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    return pd.DataFrame(rows)


df = load_rows()
print(f"loaded {len(df)} of {len(manifest['runs'])} runs")


def study(name):
    if not len(df) or "studies" not in df.columns:
        return df.iloc[0:0]
    return df[df["studies"].apply(lambda s: name in (s or []))]


def seed_mean(frame, keys, metrics):
    """Average over seeds -> one row per config (+ std for error bars)."""
    metrics = [m for m in metrics if m in frame.columns]
    g = frame.groupby(keys, dropna=False)[metrics]
    out = g.mean().reset_index()
    for m in metrics:
        out[m + "_std"] = g.std().reset_index()[m].values
    return out


def load_curve(metrics_path, key):
    """Return (coverage, selective_accuracy) arrays from a run's swept curve."""
    m = json.loads(Path(metrics_path).read_text())
    c = pd.DataFrame(m[key])
    c = c.dropna(subset=["coverage", "selective_accuracy"])
    return c["coverage"].to_numpy(), c["selective_accuracy"].to_numpy()
'''

STYLE = '''# Shared paper-quality plotting style + a figs/ saver.
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
C_LEARNED, C_CONF = "#1f77b4", "#d62728"   # learned reject vs confidence baseline
FIGS = root / "figs"; FIGS.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    print("saved", FIGS / f"{name}.png")
'''

FORCED = '''# Sanity: forced (full-coverage) test accuracy across the whole sweep. This is the
# plain classifier number -- how well the redacted narrative predicts the product-
# problem flag at all. The abstention only changes what we DECLINE to predict.
if len(df):
    print("forced accuracy  : %.4f +/- %.4f" % (df.forced_acc.mean(), df.forced_acc.std()))
    print("forced ROC-AUC   : %.4f" % df.forced_auc.dropna().mean())
    print("mean abstain prob: %.4f" % df.mean_abstain.mean())
    display(df.sort_values(["o", "seed"]).head(12))
else:
    print("no runs loaded yet -- run the sweep first")
'''

GAIN_VS_O = '''# HEADLINE: does the learned reject beat confidence thresholding, as a function of
# the payoff o? gain_vs_conf = selective_acc(learned) - selective_acc(confidence) at
# matched coverage. > 0 means the learned reject head adds something; <= 0 means the
# model's own class confidence is already as good a reject signal (the SMT finding).
pay = study("payoff")
if len(pay):
    agg = seed_mean(pay, ["o"], ["gain_vs_conf", "learned_sel_acc", "conf_sel_acc"]).sort_values("o")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.errorbar(agg["o"], agg["gain_vs_conf"], yerr=agg["gain_vs_conf_std"],
                marker="o", color=C_LEARNED, capsize=3)
    ax.set_xlabel("abstention payoff  o"); ax.set_ylabel("selective-acc gain (learned - confidence)")
    ax.set_title("Learned reject vs confidence threshold, across o")
    save(fig, "fig_maude_gain_vs_o"); plt.show()
    print(agg[["o", "gain_vs_conf", "gain_vs_conf_std"]].to_string(index=False))
else:
    print("no payoff-study runs loaded")
'''

RISK_COVERAGE = '''# Selective accuracy vs coverage: the learned-reject curve (sweep h) against the
# confidence baseline curve (sweep threshold), for the BASE payoff o, averaged over
# seeds by interpolating each seed onto a common coverage grid. This is the
# risk-coverage / selective-risk view -- the closer to the top-right, the better.
base_o = manifest["base"]["o"]
sub = study("payoff")
sub = sub[sub["o"] == base_o] if len(sub) else sub
if len(sub):
    grid = np.linspace(0.3, 1.0, 40)
    def mean_curve(key):
        ys = []
        for _, r in sub.iterrows():
            cov, acc = load_curve(r["_metrics_path"], key)
            if len(cov) < 2:
                continue
            order = np.argsort(cov)
            ys.append(np.interp(grid, cov[order], acc[order], left=np.nan, right=np.nan))
        return np.nanmean(np.vstack(ys), axis=0) if ys else np.full_like(grid, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, mean_curve("reject_curve"), color=C_LEARNED, lw=2, label="learned reject (r < h)")
    ax.plot(grid, mean_curve("confidence_curve"), color=C_CONF, lw=2, ls="--", label="confidence (max prob >= t)")
    ax.set_xlabel("coverage"); ax.set_ylabel("selective accuracy")
    ax.set_title(f"Risk-coverage at base payoff o={base_o}")
    ax.legend(); save(fig, "fig_maude_risk_coverage"); plt.show()
else:
    print("no base-o runs loaded")
'''

METRICS_VS_O = '''# Forced accuracy, selective accuracy (learned reject @ target coverage), and mean
# abstain prob as o sweeps. Larger o -> the payoff rewards predicting -> the model
# abstains LESS (mean abstain prob falls, coverage rises).
pay = study("payoff")
if len(pay):
    agg = seed_mean(pay, ["o"], ["forced_acc", "learned_sel_acc", "conf_sel_acc",
                                 "mean_abstain", "learned_cov"]).sort_values("o")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(agg["o"], agg["forced_acc"], marker="o", color="0.4", label="forced (full coverage)")
    ax.plot(agg["o"], agg["learned_sel_acc"], marker="o", color=C_LEARNED, label="learned reject (selective)")
    ax.plot(agg["o"], agg["conf_sel_acc"], marker="s", color=C_CONF, ls="--", label="confidence (selective)")
    ax.set_xlabel("payoff o"); ax.set_ylabel("accuracy"); ax.set_title("Accuracy vs o"); ax.legend()
    ax = axes[1]
    ax.plot(agg["o"], agg["mean_abstain"], marker="o", color="#2ca02c", label="mean abstain prob")
    ax.plot(agg["o"], agg["learned_cov"], marker="s", color="#9467bd", label="coverage @ target")
    ax.set_xlabel("payoff o"); ax.set_ylabel("value"); ax.set_title("Abstention behavior vs o"); ax.legend()
    fig.tight_layout(); save(fig, "fig_maude_metrics_vs_o"); plt.show()
else:
    print("no payoff-study runs loaded")
'''

STUDIES = '''# Hyperparameter studies: gain_vs_conf and forced accuracy under capacity (trunk
# width/depth), dropout, and lr -- each varied one at a time around BASE.
studies = [("capacity", "hidden"), ("dropout", "dropout"), ("lr", "lr")]
present = [(s, k) for s, k in studies if len(study(s))]
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 4.2), squeeze=False)
    for ax, (s, key) in zip(axes[0], present):
        agg = seed_mean(study(s), [key], ["gain_vs_conf", "forced_acc"])
        agg = agg.sort_values(key)
        xs = agg[key].astype(str)
        ax.axhline(0, color="0.6", lw=1, ls="--")
        ax.errorbar(xs, agg["gain_vs_conf"], yerr=agg["gain_vs_conf_std"],
                    marker="o", color=C_LEARNED, capsize=3, label="gain vs conf")
        ax.set_title(f"{s} study"); ax.set_xlabel(key); ax.set_ylabel("gain vs conf")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); save(fig, "fig_maude_studies"); plt.show()
else:
    print("no study runs loaded")
'''

OPTIMAL = '''# Best operating points. "Best learned" = highest learned selective accuracy at the
# target coverage; also report where the learned reject most beats confidence.
if len(df):
    keep = df.dropna(subset=["learned_sel_acc"])
    if len(keep):
        best = keep.loc[keep["learned_sel_acc"].idxmax()]
        print("highest learned selective accuracy:")
        print(f"  {best.run_id}  o={best.o}  sel_acc={best.learned_sel_acc:.4f} "
              f"@ coverage={best.learned_cov:.3f}  (gain vs conf {best.gain_vs_conf:+.4f})")
    g = df.dropna(subset=["gain_vs_conf"])
    if len(g):
        bg = g.loc[g["gain_vs_conf"].idxmax()]
        print("largest gain of learned reject over confidence baseline:")
        print(f"  {bg.run_id}  o={bg.o}  gain={bg.gain_vs_conf:+.4f}")
        print(f"mean gain_vs_conf across all runs: {g.gain_vs_conf.mean():+.4f} "
              f"(negative => confidence thresholding is as good or better)")
else:
    print("no runs loaded")
'''

SECTIONS = [
    ("## Load the sweep", LOAD),
    (None, STYLE),
    ("## Forced (full-coverage) accuracy", FORCED),
    ("## Payoff study: learned reject vs confidence baseline\\n\\n### Gain vs o (headline)\\n\\n"
     "**Saved: `fig_maude_gain_vs_o`.**", GAIN_VS_O),
    ("### Risk-coverage curves\\n\\nSelective accuracy vs coverage at the base payoff, learned "
     "reject vs confidence baseline. **Saved: `fig_maude_risk_coverage`.**", RISK_COVERAGE),
    ("### Accuracy + abstention behavior across o\\n\\n**Saved: `fig_maude_metrics_vs_o`.**", METRICS_VS_O),
    ("## Hyperparameter studies\\n\\nCapacity / dropout / lr. **Saved: `fig_maude_studies`.**", STUDIES),
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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "real_data_analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}  ({sum(1 for _, s in SECTIONS)} code cells)")


if __name__ == "__main__":
    main()
