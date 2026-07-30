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

TITLE = """# MAUDE abstention sweep -- regular vs abstention model

One scraped MAUDE dataset (4-class severity from redacted narratives: Malfunction /
Basic injury / Serious injury / Death). Every hyperparameter config (class_weight /
capacity / dropout / lr) is trained with the learned-abstention term `-log(o*p_y + r)`
over the **full** payoff sweep **o** (1.0 .. 5.0), replicated across seeds.

The notebook then **auto-picks the best config** -- ranked by forced macro-F1 in the
plain-classifier regime (largest `o`) -- freezes it, and draws every figure from that
one config's o-sweep. So the whole analysis is **one model**, `o` the only thing that
moves; it never changes models midway. (Set `PICK_CONFIG` in the select cell to
override the auto-pick.)

The **regular baseline** is that same model at the largest `o`: there the reject
threshold `1/o` drops below chance (0.25 for 4 classes), so it never abstains -- a
plain classifier. Lowering `o` turns abstention on: the model declines the rows it is
unsure about and is more accurate on the rows it keeps.

Paper-ready figures write to `figs/fig_maude_*.{png,pdf}`:

| figure | what it shows |
|---|---|
| `fig_maude_accuracy_vs_o` | accuracy vs `o`: forced (monotone) + selective at each model's chosen operating point |
| `fig_maude_coverage_accuracy_by_o` | selective accuracy vs coverage, one line per `o` (pick an operating point) |
| `fig_maude_regular_vs_selective` | regular (baseline, all rows) vs selective (abstain on hard rows) |

Prereq: run the sweep first (`bash real-data/cluster/submit.sh`), then run this
notebook from the repo root.
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
        lr_ = m.get("learned_reject") or {}        # None if a run had no abstain column
        cb = m.get("confidence_baseline") or {}    # legacy field; unused in the main story
        f = m["forced"]
        rows.append({
            "run_id": r["run_id"], "loss": r.get("loss"), "o": r["o"],
            "class_weight": r.get("class_weight"), "seed": r["seed"],
            "hidden": r.get("hidden"), "dropout": r.get("dropout"), "lr": r.get("lr"),
            "studies": r.get("studies", []),
            "forced_acc": f["accuracy"], "forced_f1": f.get("f1"),
            "forced_macro_f1": f.get("macro_f1"), "forced_auc": f.get("roc_auc"),
            "mean_abstain": m["mean_abstain_prob"],
            "learned_cov": lr_.get("coverage"), "learned_sel_acc": lr_.get("selective_accuracy"),
            "conf_cov": cb.get("coverage"), "conf_sel_acc": cb.get("selective_accuracy"),
            "gain_vs_conf": m.get("gain_vs_conf"),
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

BASELINE = '''# The REGULAR BASELINE = the abstention model at the LARGEST o. There 1/o is below
# chance (0.25 for 4 classes), so the model effectively never abstains -- it is a
# plain classifier that predicts on EVERY row. Everything below is the SAME model at
# smaller o (abstention turned on). On the skewed 4 classes, MACRO-F1 / balanced
# accuracy matter more than plain accuracy, which Malfunction (~65% of rows) can carry
# on its own.
pay = PAY
if len(pay):
    o_max = pay["o"].max()
    b = seed_mean(pay[pay["o"] == o_max], ["o"],
                  ["forced_acc", "forced_macro_f1", "mean_abstain"]).iloc[0]
    print(f"regular baseline = abstention model at o={o_max:g}  (never abstains)")
    print(f"  forced accuracy   : {b.forced_acc:.4f}   (predicts on all rows)")
    print(f"  forced macro-F1   : {b.forced_macro_f1:.4f}")
    print(f"  mean abstain prob : {b.mean_abstain:.4f}   (~0 confirms it does not abstain)")
    display(seed_mean(pay, ["o"], ["forced_acc", "forced_macro_f1", "mean_abstain"])
            .sort_values("o").round(4))
else:
    print("no payoff runs loaded yet -- run the sweep first")
'''

ACC_VS_O = '''# ACCURACY vs PAYOFF o -- the headline. NO fixed coverage: each o is read at ITS OWN
# chosen operating point -- the learned-reject model (accept when abstain r < h, with h
# tuned on val to the target coverage), i.e. the same model the analysis picks. So both
# the accuracy AND the coverage vary with o; read the two panels together.
#   forced (grey)    : accuracy on ALL rows -- rises with o and saturates (each o is a
#                      different model; at low o the abstain column steals probability
#                      during training so the classifier under-trains).
#   selective (blue) : accuracy at the chosen learned-reject operating point. At low o
#                      the model keeps few rows of an under-trained classifier; at the
#                      largest o it cannot abstain (coverage -> 1) so it collapses onto
#                      forced. A mid o -- decent classifier AND real selectivity -- can
#                      PEAK. That peak is the best payoff o.
#   baseline (dashed): the largest-o model (never abstains), accuracy on all rows.
pay = PAY
if len(pay):
    agg = (seed_mean(pay, ["o"], ["forced_acc", "learned_sel_acc", "learned_cov",
                                  "mean_abstain"]).sort_values("o").reset_index(drop=True))
    o_arr = agg["o"].to_numpy()
    o_base = float(o_arr.max()); base_acc = float(agg.loc[agg["o"].idxmax(), "forced_acc"])
    sel = agg["learned_sel_acc"].to_numpy()
    peak_i = int(np.nanargmax(sel)); peak_o = float(o_arr[peak_i])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(o_arr, agg["forced_acc"], marker="o", color="0.6", lw=1.5,
            label="regular / forced (all rows)")
    ax.plot(o_arr, sel, marker="o", color=C_LEARNED, lw=2,
            label="selective (learned reject -- the chosen model)")
    ax.axhline(base_acc, color="0.35", ls="--", lw=1, label=f"baseline (o={o_base:g}) = {base_acc:.3f}")
    ax.axvline(peak_o, color=C_LEARNED, ls=":", lw=1)
    ax.annotate(f"best o={peak_o:g}", (peak_o, sel[peak_i]),
                textcoords="offset points", xytext=(6, 6), color=C_LEARNED)
    ax.set_xlabel("payoff o"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs o (selective read at each model's chosen operating point)")
    ax.legend()
    ax2 = axes[1]
    ax2.plot(o_arr, agg["learned_cov"], marker="o", color="#9467bd", label="coverage (kept fraction)")
    ax2.plot(o_arr, agg["mean_abstain"], marker="s", color="#2ca02c", label="mean abstain prob")
    ax2.set_xlabel("payoff o"); ax2.set_ylabel("value")
    ax2.set_title("Operating point vs o (coverage is NOT held fixed)"); ax2.legend()
    fig.tight_layout(); save(fig, "fig_maude_accuracy_vs_o"); plt.show()

    print(f"regular baseline (o={o_base:g}) forced accuracy : {base_acc:.4f}")
    print(f"best payoff o = {peak_o:g}: selective accuracy = {sel[peak_i]:.4f} "
          f"@ coverage {agg.loc[peak_i, 'learned_cov']:.3f}  (+{sel[peak_i] - base_acc:.4f} over baseline)")
    print(agg[["o", "forced_acc", "learned_sel_acc", "learned_cov"]].round(4).to_string(index=False))
else:
    print("no payoff-study runs loaded")
'''

SELECT = '''# AUTO-PICK the best config. Every hyperparameter config was swept over the FULL o
# range; here we rank the configs by forced macro-F1 in the PLAIN-CLASSIFIER regime --
# the largest o, where the reject threshold 1/o is below chance (0.25 for 4 classes) so
# abstention is off and forced macro-F1 is pure classifier quality. We FREEZE the winner
# and set `PAY` = that one config's o-sweep; every figure below draws from PAY only, so
# nothing changes models midway. Set PICK_CONFIG to override the auto-pick.
PICK_CONFIG = None    # e.g. {"class_weight": "sqrt", "dropout": 0.1}; None = auto-pick
CONFIG_COLS = ["class_weight", "hidden", "dropout", "lr"]

pay_all = study("payoff")
if len(pay_all):
    o_max = pay_all["o"].max()
    ranking = (seed_mean(pay_all[pay_all["o"] == o_max], CONFIG_COLS,
                         ["forced_macro_f1", "forced_acc"])
               .sort_values("forced_macro_f1", ascending=False).reset_index(drop=True))
    print(f"config ranking by forced macro-F1 at o={o_max:g} (plain-classifier regime, "
          f"mean over seeds):")
    print(ranking.round(4).to_string(index=False))
    if PICK_CONFIG is not None:
        CHOSEN = {c: PICK_CONFIG.get(c, manifest["base"][c]) for c in CONFIG_COLS}
    else:
        CHOSEN = {c: ranking.loc[0, c] for c in CONFIG_COLS}
    mask = np.ones(len(pay_all), bool)
    for c in CONFIG_COLS:
        mask &= (pay_all[c] == CHOSEN[c])
    PAY = pay_all[mask]
    print("\\nchosen config (frozen for every figure below):")
    print("  " + ", ".join(f"{c}={CHOSEN[c]}" for c in CONFIG_COLS))
    print(f"  o-sweep runs for this config: {len(PAY)}")
else:
    PAY = pay_all
    print("no payoff runs loaded yet -- run the sweep first")
'''

CHOOSER = '''# CHOOSE the operating point. Each o trains a different model; each model trades
# coverage for selective accuracy by how much it abstains. This plots selective
# accuracy vs coverage for EVERY o (color = o), so you can find an o that keeps GOOD
# COVERAGE with GOOD ACCURACY -- not the highest accuracy at a tiny coverage. The table
# gives selective accuracy at a few fixed coverages so you can pick a value numerically,
# then set PICK_O in the next cell.
import matplotlib.cm as cm
from matplotlib.colors import Normalize
pay = PAY
if len(pay):
    o_vals = sorted(pay["o"].unique())
    norm = Normalize(min(o_vals), max(o_vals)); cmap = cm.viridis
    grid = np.linspace(0.2, 1.0, 60)
    def cov_curve(o, field):
        ys = []
        for p in pay[pay["o"] == o]["_metrics_path"]:
            m = json.loads(Path(p).read_text())
            c = pd.DataFrame(m.get("reject_curve") or []).dropna(subset=["coverage", field])
            if len(c) < 2:
                continue
            c = c.sort_values("coverage")
            ys.append(np.interp(grid, c["coverage"], c[field], left=np.nan, right=np.nan))
        return np.nanmean(np.vstack(ys), axis=0) if ys else np.full_like(grid, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for o in o_vals:
        ax.plot(grid, cov_curve(o, "selective_accuracy"), color=cmap(norm(o)), lw=1.3)
    ax.set_xlabel("coverage (fraction predicted)"); ax.set_ylabel("selective accuracy")
    ax.set_title("Selective accuracy vs coverage, one line per payoff o")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="payoff o")
    save(fig, "fig_maude_coverage_accuracy_by_o"); plt.show()

    targets = [0.70, 0.80, 0.90, 0.95]
    tbl = []
    for o in o_vals:
        ca = cov_curve(o, "selective_accuracy")
        row = {"o": o}
        for tc in targets:
            row[f"acc@cov{tc:g}"] = round(float(ca[int(np.argmin(np.abs(grid - tc)))]), 3)
        tbl.append(row)
    print("selective accuracy at fixed coverages (pick the o that suits you):")
    print(pd.DataFrame(tbl).to_string(index=False))
else:
    print("no payoff-study runs loaded")
'''

REG_VS_SEL = '''# REGULAR vs SELECTIVE accuracy -- two models from the SAME family.
#   regular   = the BASELINE (largest o, never abstains): accuracy on EVERY test row.
#   selective = an abstention model (mid o): accuracy on the rows it KEEPS, at
#               coverage = TARGET_COV. Selection is COVERAGE-AWARE -- by default it
#               picks the abstaining o with the best selective accuracy AT the coverage
#               you care about (not the highest accuracy at a tiny coverage). Set PICK_O
#               to a specific o (from the chooser above) to override.
PICK_O = None        # set e.g. 1.6 to force that model; None = auto-pick at TARGET_COV
TARGET_COV = 0.80    # the coverage you want good accuracy at

pay = PAY
def at_cov(path, cov, field):
    m = json.loads(Path(path).read_text())
    c = pd.DataFrame(m.get("reject_curve") or []).dropna(subset=["coverage", field])
    if len(c) < 2:
        return np.nan
    c = c.sort_values("coverage")
    return float(np.interp(cov, c["coverage"], c[field]))

if len(pay):
    o_vals = sorted(pay["o"].unique())
    o_base = max(o_vals)                                          # the no-abstention baseline
    reg = float(pay[pay["o"] == o_base]["forced_acc"].mean())    # baseline, all rows
    cand = [o for o in o_vals if o != o_base]                    # abstaining models only
    if PICK_O is not None:
        chosen_o = PICK_O
    else:
        scores = {o: np.nanmean([at_cov(p, TARGET_COV, "selective_accuracy")
                                 for p in pay[pay["o"] == o]["_metrics_path"]]) for o in cand}
        chosen_o = max(cand, key=lambda o: (-1 if np.isnan(scores[o]) else scores[o]))
    rows = pay[pay["o"] == chosen_o]
    paths = rows["_metrics_path"].tolist()
    sel = float(np.nanmean([at_cov(p, TARGET_COV, "selective_accuracy") for p in paths]))
    sel_f1 = float(np.nanmean([at_cov(p, TARGET_COV, "selective_f1") for p in paths]))
    print(f"regular baseline : o={o_base:g}, accuracy on ALL rows = {reg:.4f}")
    print(f"abstention model : o={chosen_o} (auto = best selective acc at {TARGET_COV:.0%} "
          f"coverage; set PICK_O to override)")
    print(f"selective accuracy @ {TARGET_COV:.0%} coverage : {sel:.4f}   (+{sel - reg:.4f} over baseline)")
    print(f"selective macro-F1 @ {TARGET_COV:.0%} coverage : {sel_f1:.4f}")

    grid = np.linspace(0.3, 1.0, 40)
    ys = []
    for p in paths:
        m = json.loads(Path(p).read_text())
        c = pd.DataFrame(m.get("reject_curve") or []).dropna(subset=["coverage", "selective_accuracy"])
        if len(c) < 2:
            continue
        c = c.sort_values("coverage")
        ys.append(np.interp(grid, c["coverage"], c["selective_accuracy"], left=np.nan, right=np.nan))
    sel_vs_cov = np.nanmean(np.vstack(ys), axis=0) if ys else np.full_like(grid, np.nan)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.5))
    a0.bar(["regular\\n(all rows)", f"selective\\n({TARGET_COV:.0%} coverage)"], [reg, sel],
           color=["0.55", C_LEARNED])
    for i, v in enumerate([reg, sel]):
        a0.text(i, v + 0.01, f"{v:.3f}", ha="center")
    a0.set_ylabel("accuracy"); a0.set_ylim(0, 1)
    a0.set_title(f"Regular vs selective accuracy (o={chosen_o})")
    a1.axhline(reg, color="0.35", ls="--", label=f"regular = {reg:.3f}")
    a1.plot(grid, sel_vs_cov, color=C_LEARNED, lw=2, label="selective")
    a1.axvline(TARGET_COV, color="0.7", ls=":", label=f"chosen coverage = {TARGET_COV:.0%}")
    a1.set_xlabel("coverage (fraction predicted)"); a1.set_ylabel("accuracy")
    a1.set_title("Selective accuracy vs coverage"); a1.legend()
    fig.tight_layout(); save(fig, "fig_maude_regular_vs_selective"); plt.show()
else:
    print("no payoff-study runs loaded")
'''

OPTIMAL = '''# Best operating point for the CHOSEN config: the o with the highest learned selective
# accuracy at its target coverage, next to that config's regular baseline (largest o).
pay = PAY
if len(pay):
    keep = pay.dropna(subset=["learned_sel_acc"])
    if len(keep):
        agg = seed_mean(keep, ["o"], ["learned_sel_acc", "learned_cov"])
        best = agg.loc[agg["learned_sel_acc"].idxmax()]
        print(f"best payoff o={best.o:g}: learned selective accuracy={best.learned_sel_acc:.4f} "
              f"@ coverage={best.learned_cov:.3f}")
    o_base = pay["o"].max()
    b = float(pay[pay["o"] == o_base]["forced_acc"].mean())
    print(f"regular baseline (o={o_base:g}) forced accuracy: {b:.4f}  (predicts on all rows)")
else:
    print("no runs loaded")
'''

SECTIONS = [
    ("## Load the sweep", LOAD),
    (None, STYLE),
    ("## Select the best config\\n\\nRank every hyperparameter config (each swept over the full "
     "`o` range) by forced macro-F1 in the plain-classifier regime, freeze the winner, and draw "
     "every figure below from that one config's o-sweep -- so nothing changes models midway. Set "
     "`PICK_CONFIG` to override.", SELECT),
    ("## Regular baseline (no abstention)\\n\\nThe largest-`o` model, which never abstains -- the "
     "plain classifier the abstention models are compared against.", BASELINE),
    ("## Accuracy vs payoff o (headline)\\n\\nForced accuracy rises to the baseline and plateaus; "
     "selective accuracy is read at each model's own chosen operating point (learned reject -- "
     "coverage NOT held fixed), so it can peak in the middle. That peak is the best `o`. "
     "**Saved: `fig_maude_accuracy_vs_o`.**", ACC_VS_O),
    ("## Choose the operating point\\n\\nSelective accuracy vs coverage for every `o`, plus a table "
     "of accuracy at fixed coverages -- so you can pick an `o` with good coverage AND accuracy. "
     "**Saved: `fig_maude_coverage_accuracy_by_o`.**", CHOOSER),
    ("## Regular vs selective accuracy\\n\\nThe baseline (all rows) vs an abstention model that "
     "declines the hard rows (selective, at `TARGET_COV`). Coverage-aware selection: set "
     "`TARGET_COV` (and optionally `PICK_O`) at the top of the cell. **Saved: `fig_maude_regular_vs_selective`.**", REG_VS_SEL),
    ("## Optimal operating point", OPTIMAL),
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
