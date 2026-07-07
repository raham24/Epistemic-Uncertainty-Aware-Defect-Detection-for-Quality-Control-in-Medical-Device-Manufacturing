"""Build cluster_analysis.ipynb (the sweep comparison notebook) from cell sources.

Run from the repo root:  python3 cluster/build_analysis_nb.py
Emits cluster_analysis.ipynb in the repo root. Stdlib only (json).

The notebook loads cluster/manifest.json + every results/cluster/<run>.json that
training produced, then analyses the sweep in two layers:

  * core grid   -- both losses x every dataset variant (the headline comparison),
                   each measured against its own Bayes ceiling.
  * studies     -- one-factor-at-a-time sweeps (payoff o, capacity, dropout, lr,
                   class weighting), each isolating one axis.

Figures use a shared paper-quality style and the key ones are saved to figs/ as
PNG + PDF (fig_*.{png,pdf}) so they can drop straight into the paper.
"""

import json
from pathlib import Path

TITLE = """# Cluster sweep analysis

Compares every model trained by the cluster sweep (`cluster/`). The sweep is a
**core grid** (both losses x all dataset variants, each vs its Bayes ceiling) plus
focused **studies** that vary one hyperparameter at a time.

Paper-ready figures are written to `figs/fig_*.{png,pdf}` as they render:

| figure | what it shows |
|---|---|
| `fig_accuracy_vs_bayes` | defect accuracy vs the Bayes ceiling, per dataset (headline) |
| `fig_gap_to_bayes` | how far each loss sits below optimal |
| `fig_risk_coverage` | abstention's coverage/accuracy trade-off across `o` (one operating point per model) |
| `fig_metrics_vs_o` | accuracy + macro-F1 across the payoff `o` sweep |
| `fig_selective_risk` | selective-risk curves (reject threshold swept) vs confidence baseline + Bayes floor |
| `fig_abstention_gain_vs_difficulty` | does abstaining help more as the data gets harder? |
| `fig_margin_hist` | rejected boards are more ambiguous (smaller margin) |
| `fig_mechanism_risk`, `fig_capacity`, `fig_dropout_lr`, `fig_class_weight` | mechanism/risk + hyperparameter studies |

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
    """One flat row per run that produced a metrics file. Hyperparameters + study
    tags come from the manifest; scores from the run's metrics JSON."""
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
    losses = [l for l in ["cascade", "abstention"] if l in set(core["loss"])] if len(core) else []
    x = np.arange(len(datasets)); w = 0.8 / max(len(losses), 1)
    return core, datasets, losses, x, w
'''

STYLE = '''# Shared paper-quality plotting style + a figs/ saver. Consistent colors for the two
# losses everywhere; horizontal layouts + zoomed axes so small differences are legible.
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
COLORS = {"cascade": "#2c7fb8", "abstention": "#e6550d", "bayes": "#333333"}

FIGDIR = root / "figs"
FIGDIR.mkdir(exist_ok=True)


def save_fig(fig, name):
    """Write a paper copy (PNG + PDF) to figs/ and print the path."""
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"saved figs/{name}.png (+ .pdf)")
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

ACC_DUMBBELL = '''# HEADLINE: defect accuracy vs the Bayes ceiling, one row per dataset. A horizontal
# dumbbell zoomed to the real range makes the ~1% gaps legible (a bar chart from zero
# crushes them against the top). Cascade should sit essentially on the ceiling;
# abstention is shown at FORCED accuracy here -- its fair view is the risk-coverage
# curve below.
core = study("core")
datasets = list(manifest["datasets"])
agg = seed_mean(core, ["dataset", "loss"], ["defect_acc"])
ceil = {d: core[core.dataset == d]["bayes"].mean() for d in datasets}
order = sorted(datasets, key=lambda d: ceil[d])          # highest ceiling on top


def _acc(d, loss):
    v = agg[(agg.dataset == d) & (agg.loss == loss)]["defect_acc"]
    return float(v.iloc[0]) if len(v) else np.nan


fig, ax = plt.subplots(figsize=(9, 6))
for i, d in enumerate(order):
    ca, ab, cl = _acc(d, "cascade"), _acc(d, "abstention"), ceil[d]
    pts = [p for p in (ca, ab, cl) if not np.isnan(p)]
    ax.plot([min(pts), max(pts)], [i, i], color="#d9d9d9", lw=2.5, zorder=1)
    ax.scatter(cl, i, marker="D", s=70, color=COLORS["bayes"], zorder=3,
               label="Bayes ceiling" if i == 0 else "")
    ax.scatter(ca, i, s=95, color=COLORS["cascade"], zorder=3,
               label="cascade" if i == 0 else "")
    if not np.isnan(ab):
        ax.scatter(ab, i, s=95, color=COLORS["abstention"], zorder=3,
                   label="abstention (forced)" if i == 0 else "")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.set_xlabel("defect accuracy (test split)")
ax.set_title("Defect accuracy vs the Bayes ceiling, by dataset")
ax.margins(x=0.04, y=0.03); ax.grid(axis="y", alpha=0)
ax.legend(loc="lower left", ncol=3)
save_fig(fig, "fig_accuracy_vs_bayes"); plt.show()
'''

GAP_PLOT = '''# Gap to the Bayes ceiling (ceiling - accuracy). Lower = closer to optimal. Cascade
# should be a hair above zero everywhere (it tracks the ceiling); abstention's forced
# gap is larger on the harder / imbalanced regimes.
core = study("core")
datasets = list(manifest["datasets"])
agg = seed_mean(core, ["dataset", "loss"], ["gap_to_bayes"])


def _gap(d, loss):
    v = agg[(agg.dataset == d) & (agg.loss == loss)]["gap_to_bayes"]
    return float(v.iloc[0]) if len(v) else np.nan


order = sorted(datasets, key=lambda d: _gap(d, "cascade"))
y = np.arange(len(order)); h = 0.38
fig, ax = plt.subplots(figsize=(9, 6))
for k, loss in enumerate(["cascade", "abstention"]):
    vals = [_gap(d, loss) for d in order]
    ax.barh(y + (0.5 - k) * h, vals, h, color=COLORS[loss], label=loss)
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlabel("Bayes ceiling  -  defect accuracy   (lower is better)")
ax.set_title("Gap to the Bayes ceiling")
ax.axvline(0, color="k", lw=0.8); ax.grid(axis="y", alpha=0)
ax.legend(loc="lower right")
save_fig(fig, "fig_gap_to_bayes"); plt.show()
'''

RISK_MECH_PLOT = '''# Joint mechanism (9-class) accuracy and risk-head MAE, by dataset x loss.
core, datasets, losses, x, w = core_grid()
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, metric, title, ylim in [
        (axes[0], "mech_joint", "Joint mechanism accuracy", (0.6, 1.0)),
        (axes[1], "risk_mae", "Risk-head MAE (lower is better)", None)]:
    for k, loss in enumerate(losses):
        vals = [core[(core.dataset == d) & (core.loss == loss)][metric].mean() for d in datasets]
        ax.bar(x + (k - (len(losses) - 1) / 2) * w, vals, w, color=COLORS.get(loss), label=loss)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_title(title); ax.grid(axis="x", alpha=0)
    if ylim:
        ax.set_ylim(*ylim)
axes[0].set_ylabel("accuracy"); axes[1].set_ylabel("MAE"); axes[0].legend()
save_fig(fig, "fig_mechanism_risk"); plt.show()
'''

RISK_COVERAGE = '''# PAPER FIGURE: risk-coverage curve. Each abstention model (one per payoff o) gives
# one (coverage, selective-accuracy) point at the h=0.5 accept threshold; sweeping
# o = 1 -> 4 traces the frontier. The star marks cascade at full coverage (it never
# abstains). Points up-and-left of a star => abstaining buys accuracy on the boards
# the model chooses to answer.
pay = study("payoff"); core = study("core")
if len(pay):
    sm = seed_mean(pay, ["dataset", "o"], ["coverage@0.5", "selective_acc@0.5"])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for j, d in enumerate(sorted(pay["dataset"].unique())):
        s = sm[sm.dataset == d].sort_values("coverage@0.5")
        ax.plot(s["coverage@0.5"], s["selective_acc@0.5"], "-o", ms=4,
                color=cmap(j), label=d)
        cba = core[(core.dataset == d) & (core.loss == "cascade")]["defect_acc"].mean()
        ax.scatter(1.0, cba, marker="*", s=190, color=cmap(j),
                   edgecolor="k", linewidth=0.6, zorder=5)
    ax.set_xlabel("coverage  (fraction of boards the model answers)")
    ax.set_ylabel("selective accuracy  (on the answered boards)")
    ax.set_title("Risk-coverage: abstention trades coverage for accuracy\\n"
                 "line = abstention swept over o;   star = cascade at full coverage")
    ax.legend(title="dataset", loc="lower left")
    save_fig(fig, "fig_risk_coverage"); plt.show()
else:
    print("no payoff-study runs with metrics yet")
'''

PAYOFF_CLS_PLOT = '''# The o-sweep on classification quality: FORCED defect accuracy and macro-F1 as the
# payoff o goes 1.0 -> 4.0. macro-F1 climbing toward o=4 (dashed line, where the
# abstention term reduces to cross-entropy for the 3-class head) shows the low-o
# "collapse" is an operating-point choice, not a broken loss.
pay = study("payoff")
if len(pay):
    sm = seed_mean(pay, ["dataset", "o"], ["defect_acc", "defect_macrof1"])
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for j, d in enumerate(sorted(pay["dataset"].unique())):
        s = sm[sm.dataset == d].sort_values("o")
        axes[0].plot(s["o"], s["defect_acc"], "-o", ms=4, color=cmap(j), label=d)
        axes[1].plot(s["o"], s["defect_macrof1"], "-o", ms=4, color=cmap(j), label=d)
    axes[0].set_title("forced defect accuracy vs payoff o"); axes[0].set_ylabel("defect accuracy")
    axes[1].set_title("macro-F1 (minority-sensitive) vs payoff o"); axes[1].set_ylabel("macro-F1")
    for a in axes:
        a.set_xlabel("payoff o"); a.axvline(4.0, ls="--", color="k", alpha=0.5)
        a.legend(title="dataset")
    save_fig(fig, "fig_metrics_vs_o"); plt.show()
else:
    print("no payoff-study runs with metrics yet")
'''

SEL_COMPUTE = '''# Selective-classification analysis (replicates the toy_example). Unlike the cells
# above (which read only the JSON metrics), this loads the SAVED checkpoints and
# sweeps the REJECTION THRESHOLD on each model -- the standard selective-risk view.
# Needs results/cluster/*.pt + data/cluster/*.csv, so run it on the cluster (or after
# rsync-ing them back). Edit SEL_O / SEL_SEED / SEL_COVERAGE to probe other settings.
import sys
sys.path.insert(0, str(root))            # so `import mlp` works regardless of cwd
import torch
import mlp

SEL_O, SEL_SEED, SEL_COVERAGE = 2.0, 0, 0.80
_spec = mlp.load_spec(str(root / "domain" / "smt_paper.yaml"))
_names = mlp.defect_names(_spec)
_ids = mlp.param_ids(_spec)
_covs = np.linspace(0.05, 1.0, 40)


def _find(dataset, loss, o, seed):
    for r in manifest["runs"]:
        if (r["dataset"] == dataset and r["loss"] == loss and r["seed"] == seed
                and (r["o"] == o if loss == "abstention" else True)):
            return r
    return None


def _infer(run, dfd):
    """Test-split predictions for one checkpoint (standardized with its stored stats)."""
    model, ckpt = mlp.load_model(str(root / run["model"]), device="cpu")
    te = np.where(dfd["split"].to_numpy() == "test")[0]
    mu, sd = ckpt["standardize"]["mu"], ckpt["standardize"]["sd"]
    X = ((dfd.iloc[te][_ids].to_numpy(np.float32) - mu) / sd).astype(np.float32)
    y = dfd.iloc[te]["defect_label"].map({n: i for i, n in enumerate(_names)}).to_numpy()
    p = model.predict(torch.from_numpy(X))
    d = {"y": y, "argmax": p["defect_argmax"].cpu().numpy(),
         "real": p["defect_prob"].cpu().numpy(), "te": te,
         "abstain": bool(getattr(model, "abstain", False))}
    if d["abstain"]:
        d["r"] = p["abstain_prob"].cpu().numpy()
    return d


def _risk_cov(score, correct, covs=_covs):
    """Rank-based selective risk: accept the most-confident `cov` fraction (highest
    score first) -> accepted error at each coverage."""
    order = np.argsort(-score); n = len(score)
    return np.array([1 - correct[order[:max(1, int(round(c * n)))]].mean() for c in covs])


def _bayes_err(dfd, te):
    post = dfd.iloc[te][[f"p_{n}" for n in _names]].to_numpy()
    return float((1 - post.max(1)).mean())


sel, _rows = {}, []
for ds in manifest["datasets"]:
    cas, abst = _find(ds, "cascade", None, SEL_SEED), _find(ds, "abstention", SEL_O, SEL_SEED)
    csv = root / "data" / "cluster" / f"{ds}.csv"
    if not (cas and abst and csv.exists()
            and (root / cas["model"]).exists() and (root / abst["model"]).exists()):
        continue
    dfd = pd.read_csv(csv)
    ic, ia = _infer(cas, dfd), _infer(abst, dfd)
    ce_err = 1 - (ic["argmax"] == ic["y"]).mean()
    abst_err = _risk_cov(-ia["r"], (ia["argmax"] == ia["y"]).astype(float))      # reject high r
    conf_err = _risk_cov(ic["real"].max(1), (ic["argmax"] == ic["y"]).astype(float))  # reject low conf
    bayes = _bayes_err(dfd, ia["te"])
    sel[ds] = dict(bayes=bayes, ce_err=float(ce_err), abst_err=abst_err,
                   conf_err=conf_err, ia=ia)
    _at = lambda e: float(e[int(np.argmin(np.abs(_covs - SEL_COVERAGE)))])
    _rows.append(dict(dataset=ds, bayes_err=bayes, ce_full_err=float(ce_err),
                      abst_err_at=_at(abst_err), conf_err_at=_at(conf_err),
                      gain_vs_ce=float(ce_err) - _at(abst_err),
                      gain_vs_conf=_at(conf_err) - _at(abst_err)))

if _rows:
    sel_table = pd.DataFrame(_rows).sort_values("bayes_err").reset_index(drop=True)
    print(f"selective analysis: {len(sel)} datasets  (abstention o={SEL_O:g}, seed {SEL_SEED}, "
          f"reported @ coverage~{SEL_COVERAGE:.2f})")
    print(sel_table.round(4).to_string(index=False))
    print("\\ngain_vs_ce   = CE full error - abstention accepted error   (>0: abstaining beats never rejecting)")
    print("gain_vs_conf = cascade confidence error - abstention error   (>0: LEARNED reject beats confidence thresholding)")
else:
    sel_table = pd.DataFrame()
    print("No checkpoints found -- run this on the cluster (needs results/cluster/*.pt + data/cluster/*.csv).")
'''

SEL_RISK_PLOT = '''# Selective-risk curves (reject threshold swept) per dataset, ordered easy -> hard.
# abstention (reject high reservation) vs cascade confidence thresholding vs the CE
# full-coverage error and the Bayes floor. Abstention dipping below CE as coverage
# drops = it is helping; below the blue line = it beats plain confidence.
if len(sel):
    order = list(sel_table["dataset"])
    ncol = min(5, len(order)); nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for k, ds in enumerate(order):
        ax = axes[k // ncol][k % ncol]; d = sel[ds]
        ax.plot(_covs, d["abst_err"], "-", color=COLORS["abstention"], lw=2, label="abstention")
        ax.plot(_covs, d["conf_err"], "--", color=COLORS["cascade"], lw=1.8, label="cascade confidence")
        ax.axhline(d["ce_err"], color="#999", ls=":", lw=1.4, label="CE full coverage")
        ax.axhline(d["bayes"], color="k", ls="-", lw=1.0, alpha=0.6, label="Bayes floor")
        ax.set_title(f"{ds} (Bayes {d['bayes']:.3f})", fontsize=10)
        ax.set_xlabel("coverage"); ax.set_ylabel("accepted error"); ax.grid(alpha=0.25)
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig_selective_risk"); plt.show()
else:
    print("no checkpoints loaded (see the compute cell above)")
'''

SEL_GAIN_PLOT = '''# The advisor's key question: does abstaining help MORE as the problem gets harder?
# Accuracy gained by abstaining (at fixed coverage) vs each dataset's Bayes error.
# Orange rising with difficulty => abstention earns its keep on complex data; blue
# above zero => the learned reject head beats plain confidence thresholding.
if len(sel_table):
    t = sel_table
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(t["bayes_err"], t["gain_vs_ce"], "o-", color=COLORS["abstention"], label="vs CE full coverage")
    ax.plot(t["bayes_err"], t["gain_vs_conf"], "s--", color=COLORS["cascade"], label="vs cascade confidence")
    for _, r in t.iterrows():
        ax.annotate(r["dataset"], (r["bayes_err"], r["gain_vs_ce"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Bayes error  (data complexity ->)")
    ax.set_ylabel(f"accuracy gained by abstaining @ coverage~{SEL_COVERAGE:.2f}")
    ax.set_title("Does abstaining help more as the problem gets harder?")
    ax.grid(alpha=0.25); ax.legend()
    save_fig(fig, "fig_abstention_gain_vs_difficulty"); plt.show()
else:
    print("no checkpoints loaded (see the compute cell above)")
'''

SEL_MARGIN_PLOT = '''# Rejected boards have a smaller top1-top2 softmax margin (more ambiguous) -- shown
# for an easy / mid / hard dataset. This is the toy example's "smart deferral" check.
if len(sel):
    order = list(sel_table["dataset"])
    probe = [order[0], order[len(order) // 2], order[-1]]
    fig, axes = plt.subplots(1, len(probe), figsize=(5 * len(probe), 4.2), squeeze=False)
    for j, ds in enumerate(probe):
        ia = sel[ds]["ia"]; ax = axes[0][j]
        top2 = np.sort(ia["real"], axis=1)[:, -2:]; marg = top2[:, -1] - top2[:, -2]
        rej = ia["r"] >= 0.5; acc = ~rej
        if acc.sum() and rej.sum():
            bins = np.linspace(0, max(marg.max(), 1e-6), 26)
            ax.hist(marg[acc], bins=bins, density=True, alpha=0.6, color=COLORS["cascade"], label="accepted")
            ax.hist(marg[rej], bins=bins, density=True, alpha=0.6, color=COLORS["abstention"], label="rejected")
            ax.legend(fontsize=9)
        ax.set_title(ds, fontsize=11); ax.set_xlabel("top1 - top2 margin")
        ax.set_ylabel("density"); ax.grid(alpha=0.25)
    fig.suptitle("Rejected boards are more ambiguous (smaller margin)", fontweight="bold")
    fig.tight_layout(); save_fig(fig, "fig_margin_hist"); plt.show()
else:
    print("no checkpoints loaded (see the compute cell above)")
'''

CAPACITY_PLOT = '''# Capacity study: defect accuracy and gap-to-Bayes vs trunk width/depth (cascade).
cap = study("capacity")
if len(cap):
    order = [h for h in ["128,128", "256,256", "512,512", "256,256,256", "512,512,256"]
             if h in set(cap["hidden"])]
    pos = {h: i for i, h in enumerate(order)}
    sm = seed_mean(cap, ["dataset", "hidden"], ["defect_acc", "gap_to_bayes"])
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for j, d in enumerate(sorted(cap["dataset"].unique())):
        s = sm[sm.dataset == d].copy()
        s["xi"] = s["hidden"].map(pos); s = s.sort_values("xi")
        axes[0].plot(s["xi"], s["defect_acc"], "-o", color=cmap(j), label=d)
        axes[1].plot(s["xi"], s["gap_to_bayes"], "-o", color=cmap(j), label=d)
    for a, ttl, yl in [(axes[0], "defect accuracy vs trunk size", "defect accuracy"),
                       (axes[1], "gap to Bayes vs trunk size (lower better)", "gap to Bayes")]:
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=20, ha="right")
        a.set_title(ttl); a.set_ylabel(yl); a.set_xlabel("trunk hidden layers")
        a.legend(title="dataset")
    save_fig(fig, "fig_capacity"); plt.show()
else:
    print("no capacity-study runs with metrics yet")
'''

REG_OPT_PLOT = '''# Dropout and learning-rate studies: gap-to-Bayes vs each axis (lower is better).
cmap = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
dro = study("dropout")
if len(dro):
    sm = seed_mean(dro, ["dataset", "dropout"], ["gap_to_bayes"])
    for j, d in enumerate(sorted(dro["dataset"].unique())):
        s = sm[sm.dataset == d].sort_values("dropout")
        axes[0].plot(s["dropout"], s["gap_to_bayes"], "-o", color=cmap(j), label=d)
    axes[0].set_xlabel("dropout"); axes[0].set_ylabel("gap to Bayes")
    axes[0].set_title("dropout sweep"); axes[0].legend(title="dataset")
lr = study("lr")
if len(lr):
    sm = seed_mean(lr, ["dataset", "lr"], ["gap_to_bayes"])
    for j, d in enumerate(sorted(lr["dataset"].unique())):
        s = sm[sm.dataset == d].sort_values("lr")
        axes[1].plot(s["lr"], s["gap_to_bayes"], "-o", color=cmap(j), label=d)
    axes[1].set_xscale("log"); axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("gap to Bayes"); axes[1].set_title("learning-rate sweep")
    axes[1].legend(title="dataset")
save_fig(fig, "fig_dropout_lr"); plt.show()
'''

CW_PLOT = '''# Class-weight study: minority-sensitive macro-F1 and overall accuracy by mode.
cw = study("class_weight")
if len(cw):
    modes = [m for m in ["none", "sqrt", "inverse"] if m in set(cw["class_weight"])]
    dss = sorted(cw["dataset"].unique())
    sm = seed_mean(cw, ["dataset", "class_weight"], ["defect_acc", "defect_macrof1"])
    x = np.arange(len(modes)); w = 0.8 / max(len(dss), 1)
    cmap = plt.get_cmap("Set2")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for col, metric, ttl in [(0, "defect_macrof1", "macro-F1 (minority-sensitive)"),
                             (1, "defect_acc", "defect accuracy")]:
        for i, d in enumerate(dss):
            vals = [sm[(sm.dataset == d) & (sm.class_weight == m)][metric].mean() for m in modes]
            axes[col].bar(x + (i - (len(dss) - 1) / 2) * w, vals, w, color=cmap(i), label=d)
        axes[col].set_xticks(x); axes[col].set_xticklabels(modes)
        axes[col].set_xlabel("class weighting"); axes[col].set_title(ttl)
        axes[col].grid(axis="x", alpha=0)
    axes[0].set_ylabel("macro-F1"); axes[1].set_ylabel("accuracy"); axes[0].legend(title="dataset")
    save_fig(fig, "fig_class_weight"); plt.show()
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
    ("## Plot style\\n\\nShared paper style + a `save_fig` helper that writes `figs/fig_*.{png,pdf}`.", STYLE),
    ("## Core grid\\n\\nBoth losses across every dataset variant: full table, then averaged over seeds.",
     CORE_TABLE),
    (None, CORE_AGG),
    ("### Defect accuracy vs the Bayes ceiling (headline)\\n\\nHorizontal dumbbell, zoomed to the real "
     "accuracy range. **Saved: `fig_accuracy_vs_bayes`.**", ACC_DUMBBELL),
    ("### Gap to the Bayes ceiling\\n\\n**Saved: `fig_gap_to_bayes`.**", GAP_PLOT),
    ("### Mechanism accuracy and risk MAE\\n\\n**Saved: `fig_mechanism_risk`.**", RISK_MECH_PLOT),
    ("## Payoff study (abstention)\\n\\n### Risk-coverage curve\\n\\nThe selective-classification figure: "
     "coverage vs selective accuracy as `o` sweeps. **Saved: `fig_risk_coverage`.**", RISK_COVERAGE),
    ("### Classification quality across o\\n\\nForced accuracy + macro-F1 vs `o` (the o-sweep). "
     "**Saved: `fig_metrics_vs_o`.**", PAYOFF_CLS_PLOT),
    ("## Selective-classification analysis (threshold-swept)\\n\\nReplicates the professor's "
     "`toy_example`: loads the saved checkpoints and sweeps the **rejection threshold** on each "
     "model (the standard selective-risk view), instead of fixing the threshold and sweeping `o`. "
     "Runs on the cluster (needs `results/cluster/*.pt` + `data/cluster/*.csv`).", SEL_COMPUTE),
    ("### Selective-risk curves\\n\\nAccepted error vs coverage, easy -> hard, vs the confidence "
     "baseline / CE / Bayes floor. **Saved: `fig_selective_risk`.**", SEL_RISK_PLOT),
    ("### Does abstaining help more as the problem gets harder?\\n\\n**Saved: "
     "`fig_abstention_gain_vs_difficulty`.**", SEL_GAIN_PLOT),
    ("### Rejected boards are more ambiguous\\n\\n**Saved: `fig_margin_hist`.**", SEL_MARGIN_PLOT),
    ("## Capacity study\\n\\nTrunk width/depth (cascade). **Saved: `fig_capacity`.**", CAPACITY_PLOT),
    ("## Dropout and learning-rate studies\\n\\n**Saved: `fig_dropout_lr`.**", REG_OPT_PLOT),
    ("## Class-weight study\\n\\nDefect-head reweighting, where minority recall matters most. "
     "**Saved: `fig_class_weight`.**", CW_PLOT),
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
