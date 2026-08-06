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
| `fig_maude_selective_vs_o` | selective accuracy (kept rows) vs `o` |
| `fig_maude_noabstain_vs_abstain` | accuracy vs coverage: no-abstention baseline vs abstention model |
| `fig_maude_confusion_baseline` | per-class confusion for the baseline -- where the accuracy really comes from |

Prereq: run the sweep first. Generated into the repo root next to
`synthetic_analysis.ipynb`; it finds the sweep data automatically.
"""

LOAD = '''from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd

# find the repo root: walk up from the cwd; if that fails (e.g. the notebook was moved
# to the home dir), fall back to the absolute repo path baked in at generation time.
root = Path.cwd()
while not (root / "real-data" / "cluster" / "manifest.json").exists() and root != root.parent:
    root = root.parent
if not (root / "real-data" / "cluster" / "manifest.json").exists():
    root = Path(r"__REPO_ROOT__")
manifest = json.loads((root / "real-data" / "cluster" / "manifest.json").read_text())
print("runs in matrix:", len(manifest["runs"]))
print("base config    :", manifest["base"])
print("seeds          :", manifest["seeds"])


def _load_one(r):
    """Read one run's metrics JSON into a flat row (or mark it missing)."""
    p = root / r["metrics"]
    try:
        m = json.loads(p.read_text())
    except FileNotFoundError:
        return ("missing", r["run_id"])
    lr_ = m.get("learned_reject") or {}        # None if a run had no abstain column
    cb = m.get("confidence_baseline") or {}    # legacy field; unused in the main story
    f = m["forced"]
    return ("ok", {
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


def load_rows():
    """One flat row per run. Reads the ~thousands of metrics JSONs CONCURRENTLY --
    reading them one-at-a-time is I/O-bound and crawls on a shared cluster filesystem."""
    rows, missing = [], []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for kind, val in ex.map(_load_one, manifest["runs"]):
            (rows if kind == "ok" else missing).append(val)
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

PERCLASS = '''# PER-CLASS breakdown of the chosen config's plain-classifier baseline (largest o).
# Accuracy is carried by the majority classes; this shows WHERE the model actually works.
# Reads per_class_f1 + confusion_matrix already in each run's JSON (F1 averaged over
# seeds, confusion counts summed). macro-F1 = mean of the per-class F1 -- the honest
# number on the skewed classes. Watch the recall column and the off-diagonal: rare
# classes (e.g. Death) crater and leak into the majority columns.
LABELS = {4: ["Malfunction", "Basic injury", "Serious injury", "Death"],
          3: ["Malfunction", "Injury", "Death"], 2: ["neg", "pos"]}
base = PAY[PAY["o"] == PAY["o"].max()]
mats, f1s, ncls = [], [], None
for p in base["_metrics_path"]:
    m = json.loads(Path(p).read_text()); ncls = m["n_classes"]
    mats.append(np.array(m["forced"]["confusion_matrix"], dtype=float))
    f1s.append(np.array(m["forced"]["per_class_f1"], dtype=float))
if mats:
    names = LABELS.get(ncls, [f"class {i}" for i in range(ncls)])
    C = np.sum(mats, axis=0); support = C.sum(1)
    recall = np.divide(np.diag(C), support, out=np.zeros(ncls), where=support > 0)
    precision = np.divide(np.diag(C), C.sum(0), out=np.zeros(ncls), where=C.sum(0) > 0)
    f1_mean, f1_std = np.mean(f1s, axis=0), np.std(f1s, axis=0)
    tbl = pd.DataFrame({"class": names, "support": support.astype(int),
                        "precision": precision.round(3), "recall": recall.round(3),
                        "F1": f1_mean.round(3), "F1_std": f1_std.round(3)})
    print(f"plain-classifier baseline (o={PAY['o'].max():g}), summed over {len(mats)} seeds:")
    print(tbl.to_string(index=False))
    print(f"\\nmacro-F1 = {f1_mean.mean():.3f}   accuracy = {np.diag(C).sum() / C.sum():.3f}")
    Cn = C / support[:, None].clip(min=1)
    fig, ax = plt.subplots(figsize=(1.6 * ncls + 1, 1.4 * ncls))
    im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(ncls)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(ncls)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion (row-normalized), o={PAY['o'].max():g} baseline")
    for i in range(ncls):
        for j in range(ncls):
            ax.text(j, i, f"{Cn[i, j]:.2f}\\n({int(C[i, j])})", ha="center", va="center",
                    color="white" if Cn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    save(fig, "fig_maude_confusion_baseline"); plt.show()
else:
    print("no baseline runs loaded")
'''

RISKCOV = '''# RISK-COVERAGE for ONE fixed model: as it abstains MORE (coverage drops), what happens
# to accuracy vs macro-F1? Accuracy is monotone -- abstaining keeps the confident rows,
# so it only ever rises. macro-F1 is the real test: if it FALLS, abstention is dropping
# the rare classes (Death) rather than majority-class errors. O_PICK must have an active
# reject head (the largest o never abstains; very low o is under-trained).
O_PICK = 2.0
sub = PAY[np.isclose(PAY["o"], O_PICK)]
grid = np.linspace(0.35, 1.0, 60)
acc, f1 = [], []
for p in sub["_metrics_path"]:
    c = pd.DataFrame(json.loads(Path(p).read_text()).get("reject_curve") or [])
    c = c.dropna(subset=["coverage", "selective_accuracy", "selective_f1"]).sort_values("coverage")
    if len(c) < 2:
        continue
    acc.append(np.interp(grid, c["coverage"], c["selective_accuracy"], left=np.nan, right=np.nan))
    f1.append(np.interp(grid, c["coverage"], c["selective_f1"], left=np.nan, right=np.nan))
if acc:
    acc, f1 = np.nanmean(acc, axis=0), np.nanmean(f1, axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, acc, color=C_LEARNED, lw=2, marker=".", label="selective accuracy")
    ax.plot(grid, f1, color=C_CONF, lw=2, marker=".", label="selective macro-F1")
    ax.set_xlabel("coverage   (right = no abstention, left = abstain more)")
    ax.set_ylabel("score"); ax.invert_xaxis()
    ax.set_title(f"Single model o={O_PICK}: accuracy vs macro-F1 as it abstains more")
    ax.legend(); ax.grid(alpha=0.3); save(fig, "fig_maude_riskcoverage"); plt.show()
else:
    print(f"no reject curve for o={O_PICK} -- pick an o with an active reject head")
'''

RUNACC = '''# The two headline diagrams: accuracy vs o, and accuracy (risk) vs coverage. Re-derives
# the chosen config (same rule as the Select cell). Uses `root`/`manifest` from the Load
# cell. MIN_COV drops degenerate low-coverage points where the model abstains on ~all rows.
MIN_COV = 0.60          # drop o where the model answers too few rows to be meaningful

accdf_rows = []
for r in manifest["runs"]:
    p = root / r["metrics"]
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    lr = m.get("learned_reject") or {}
    accdf_rows.append({"o": r["o"], "class_weight": r["class_weight"], "hidden": r["hidden"],
                       "dropout": r["dropout"], "lr": r["lr"], "seed": r["seed"],
                       "forced_acc": m["forced"]["accuracy"],
                       "sel_acc": lr.get("selective_accuracy"),
                       "coverage": lr.get("coverage"),
                       "macro_f1": m["forced"].get("macro_f1")})
accdf = pd.DataFrame(accdf_rows)

# auto-pick the best config (highest forced macro-F1 at the largest o), then average over seeds
COLS = ["class_weight", "hidden", "dropout", "lr"]; om = accdf["o"].max()
chosen = accdf[accdf["o"] == om].groupby(COLS)["macro_f1"].mean().idxmax()
mask = np.ones(len(accdf), bool)
for k, v in zip(COLS, chosen):
    mask &= (accdf[k] == v)
d = accdf[mask].groupby("o").mean(numeric_only=True).reset_index().sort_values("o")
print("chosen config:", dict(zip(COLS, chosen)))

# --- Diagram 1: o vs selective accuracy (only where coverage is meaningful) ---
d2 = d.dropna(subset=["sel_acc"])
d2 = d2[d2["coverage"] >= MIN_COV]
fig2, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(d2["o"], d2["sel_acc"], marker="o", color="#1f77b4", lw=2)
ax.set_xlabel("payoff  o"); ax.set_ylabel("selective accuracy (kept rows)")
ax.set_title("Selective accuracy vs o")
ax.grid(alpha=0.3); fig2.tight_layout(); save(fig2, "fig_maude_selective_vs_o"); plt.show()

# --- Diagram 2: no-abstention vs abstention, accuracy vs coverage ---
base_acc = float(d.loc[d["o"] == om, "forced_acc"].iloc[0])       # baseline: full coverage
frontier = d.dropna(subset=["sel_acc"])
frontier = frontier[frontier["coverage"] >= MIN_COV].sort_values("coverage")
fig3, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(frontier["coverage"], frontier["sel_acc"], marker="o", color="#1f77b4", lw=2,
        label="abstention model (operating points across o)")
ax.scatter([1.0], [base_acc], color="#d62728", s=160, marker="*", zorder=5,
           label=f"no abstention  (coverage 100%, acc {base_acc:.3f})")
ax.axhline(base_acc, color="#d62728", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("coverage (fraction of reports answered)")
ax.set_ylabel("accuracy on answered reports")
ax.set_title("No-abstention vs abstention: accuracy vs coverage")
ax.legend(); ax.grid(alpha=0.3); fig3.tight_layout(); save(fig3, "fig_maude_noabstain_vs_abstain"); plt.show()

print(d[["o", "forced_acc", "sel_acc", "coverage"]].round(4).to_string(index=False))
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
    ("## Risk / accuracy vs coverage and vs o\\n\\nThe two headline diagrams: **selective accuracy vs "
     "`o`** (kept rows) and **accuracy (risk) vs coverage** (no-abstention baseline vs the abstention "
     "model). `MIN_COV` drops degenerate low-coverage points. **Saved: `fig_maude_selective_vs_o`, "
     "`fig_maude_noabstain_vs_abstain`.**", RUNACC),
    ("## Per-class breakdown\\n\\nWhere the accuracy actually comes from: per-class "
     "precision/recall/F1 and the confusion matrix for the baseline. Accuracy hides the rare "
     "classes; this exposes them. **Saved: `fig_maude_confusion_baseline`.**", PERCLASS),
    ("## Optimal operating point", OPTIMAL),
]


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}


def main():
    # bake the repo root into the notebook as a fallback so it still finds the data if
    # opened from elsewhere. real-data/cluster/build_analysis_nb.py -> repo root.
    repo = Path(__file__).resolve().parents[2]
    repo_root = str(repo)
    cells = [md(TITLE)]
    for header, src in SECTIONS:
        if header:
            cells.append(md(header.replace("\\n", "\n")))
        cells.append(code(src.replace("__REPO_ROOT__", repo_root)))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = repo / "real_data_analysis.ipynb"          # repo root, next to synthetic_analysis.ipynb
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}  ({sum(1 for _, s in SECTIONS)} code cells)")


if __name__ == "__main__":
    main()
