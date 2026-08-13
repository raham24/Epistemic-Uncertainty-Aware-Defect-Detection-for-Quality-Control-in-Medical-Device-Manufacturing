"""Build synthetic_analysis.ipynb (the SMT sweep analysis notebook) from cell sources.

Run from the repo root:  python3 cluster/build_analysis_nb.py
Emits synthetic_analysis.ipynb in the repo root. Stdlib only (json).

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

TITLE = """# Synthetic (SMT) sweep analysis

Analyses the synthetic sweep (`cluster/`): a multi-head MLP trained on datasets of
increasing difficulty, with and without the learned-abstention head. The focus is the
abstention story -- why harder datasets are harder, and how much abstaining buys back.

Paper-ready figures are written to `figs/fig_*.{png,pdf}` as they render:

| figure | what it shows |
|---|---|
| `fig_feature_separability_<dataset>` | per-feature class-conditional densities + overlap, baseline vs hardest |
| `fig_risk_coverage` | risk vs coverage: abstention's coverage/accuracy trade-off across `o` |
| `fig_selective_vs_o` | selective accuracy (kept rows) vs `o` on `balanced_hard` |
| `fig_cascade_vs_abstention_selective` | non-abstention vs abstention accuracy comparison, per dataset |
| `fig_selective_risk` | selective-risk curves (`baseline` + `harder`): abstention vs no-rejection |

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
# o = 1 -> 4 traces the frontier. The star marks the non-abstention model at full coverage (it never
# abstains). Points up-and-left of a star => abstaining buys accuracy on the boards
# the model chooses to answer.
RC_DATASETS = ["baseline", "harder", "imbalanced"]    # baseline + harder + imbalanced
pay = study("payoff"); core = study("core")
if len(pay):
    sm = seed_mean(pay, ["dataset", "o"], ["coverage@0.5", "selective_acc@0.5"])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for j, d in enumerate([d for d in RC_DATASETS if d in set(pay["dataset"])]):
        s = sm[sm.dataset == d].sort_values("coverage@0.5")
        ax.plot(s["coverage@0.5"], s["selective_acc@0.5"], "-o", ms=4,
                color=cmap(j), label=d)
        cba = core[(core.dataset == d) & (core.loss == "cascade")]["defect_acc"].mean()
        ax.scatter(1.0, cba, marker="*", s=190, color=cmap(j),
                   edgecolor="k", linewidth=0.6, zorder=5)
    ax.set_xlabel("coverage  (fraction of boards the model answers)")
    ax.set_ylabel("selective accuracy  (on the answered boards)")
    ax.set_title("Risk-coverage: abstention trades coverage for accuracy\\n"
                 "line = abstention swept over o;   star = non-abstention at full coverage")
    ax.legend(title="dataset", loc="lower left")
    save_fig(fig, "fig_risk_coverage"); plt.show()
else:
    print("no payoff-study runs with metrics yet")
'''

PAYOFF_CLS_PLOT = '''# SELECTIVE accuracy vs the payoff o, per dataset -- selective accuracy interpolated at a
# FIXED coverage TARGET_COV from each model's stored threshold sweep (by_threshold), so the
# curve is not confounded by how much each o abstains.
TARGET_COV = 0.65
pay_runs = [r for r in manifest["runs"] if "payoff" in (r.get("studies") or [])]
recs = []
for r in pay_runs:
    p = root / r["metrics"]
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    bt = (m.get("abstention") or {}).get("by_threshold") or {}
    pts = sorted((v["coverage"], v["selective_accuracy"]) for v in bt.values()
                 if v.get("coverage") is not None and v.get("selective_accuracy") is not None)
    sel = float(np.interp(TARGET_COV, [c for c, _ in pts], [a for _, a in pts])) if len(pts) >= 2 else np.nan
    recs.append({"dataset": r["dataset"], "o": r["o"], "seed": r["seed"], "sel_at_cov": sel})
pr = pd.DataFrame(recs)
if len(pr) and pr["sel_at_cov"].notna().any():
    sm = pr.groupby(["dataset", "o"], dropna=False)["sel_at_cov"].mean().reset_index()
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for j, dset in enumerate(sorted(pr["dataset"].dropna().unique())):
        s = sm[sm.dataset == dset].sort_values("o")
        ax.plot(s["o"], s["sel_at_cov"], "-o", ms=4, color=cmap(j), label=dset)
    ax.set_xlabel("payoff o"); ax.set_ylabel("selective accuracy (answered boards)")
    ax.set_title(f"selective accuracy vs payoff o  (@ coverage {TARGET_COV:.0%})")
    ax.axvline(4.0, ls="--", color="k", alpha=0.5); ax.legend(title="dataset")
    save_fig(fig, "fig_metrics_vs_o"); plt.show()
else:
    print("no payoff-study abstention runs with by_threshold metrics yet")
'''

SEL_COMPUTE = '''# Selective-classification analysis (replicates the toy_example). Unlike the cells
# above (which read only the JSON metrics), this loads the SAVED abstention checkpoints and
# sweeps coverage on ONE model -- the learned reject head vs the same model's softmax
# confidence, both referenced to that model's own full-coverage (CE) error.
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


SEL_SEEDS = sorted({r["seed"] for r in manifest["runs"]})   # every seed -> mean curve + seed band

sel, _rows = {}, []
for ds in manifest["datasets"]:
    csv = root / "data" / "cluster" / f"{ds}.csv"
    if not csv.exists():
        continue
    dfd = None
    abst_stack, conf_stack, ce_list = [], [], []      # one curve/value per seed
    bayes, ia_last = None, None
    for seed in SEL_SEEDS:
        abst = _find(ds, "abstention", SEL_O, seed)   # ONE model for all three curves
        if not (abst and (root / abst["model"]).exists()):
            continue
        if dfd is None:
            dfd = pd.read_csv(csv)
        ia = _infer(abst, dfd)
        correct = (ia["argmax"] == ia["y"]).astype(float)
        ce_list.append(1 - correct.mean())                            # THIS model's full-coverage error
        abst_stack.append(_risk_cov(-ia["r"], correct))               # rank by the learned reject head
        conf_stack.append(_risk_cov(ia["real"].max(1), correct))      # rank by the SAME model's softmax confidence
        if bayes is None:
            bayes = _bayes_err(dfd, ia["te"])         # dataset property -- same across seeds
        ia_last = ia
    if not abst_stack:
        continue
    abst_stack, conf_stack, ce_arr = np.array(abst_stack), np.array(conf_stack), np.array(ce_list)
    abst_err, conf_err, ce_err = abst_stack.mean(0), conf_stack.mean(0), float(ce_arr.mean())
    sel[ds] = dict(bayes=bayes, ce_err=ce_err, abst_err=abst_err, conf_err=conf_err,
                   abst_err_seeds=abst_stack, conf_err_seeds=conf_stack,   # [n_seeds, n_covs]
                   ce_err_seeds=ce_arr, n_seeds=len(abst_stack), ia=ia_last)
    _at = lambda e: float(e[int(np.argmin(np.abs(_covs - SEL_COVERAGE)))])
    _rows.append(dict(dataset=ds, bayes_err=bayes, ce_full_err=ce_err, n_seeds=len(abst_stack),
                      abst_err_at=_at(abst_err), conf_err_at=_at(conf_err),
                      gain_vs_ce=ce_err - _at(abst_err),
                      gain_vs_conf=_at(conf_err) - _at(abst_err)))

if _rows:
    sel_table = pd.DataFrame(_rows).sort_values("bayes_err").reset_index(drop=True)
    print(f"selective analysis: {len(sel)} datasets  (abstention o={SEL_O:g}, mean over seeds "
          f"{SEL_SEEDS}, reported @ coverage~{SEL_COVERAGE:.2f})")
    print(sel_table.round(4).to_string(index=False))
    print("\\ngain_vs_ce   = full-coverage error - abstention accepted error   (>0: abstaining helps vs never rejecting)")
    print("gain_vs_conf = softmax-confidence error - reject-head error   (>0: the LEARNED reject beats the model's own softmax confidence)")
else:
    sel_table = pd.DataFrame()
    print("No checkpoints found -- run this on the cluster (needs results/cluster/*.pt + data/cluster/*.csv).")
'''

SEL_RISK_PLOT = '''# Selective-risk curves per dataset (easy -> hard). Single model (the abstention model):
# accepted error vs coverage when the test boards are ranked by the learned reject head.
# Solid line = MEAN over seeds; shaded band = min..max over seeds (the full seed envelope).
# Dashed line = the model's own full-coverage (no-rejection) error; the curve meets it at
# coverage 1 and falls below it as coverage drops.
if len(sel):
    order = [d for d in sel_table["dataset"] if d in ("baseline", "harder")]   # paper datasets
    ncol = min(5, len(order)); nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for k, ds in enumerate(order):
        ax = axes[k // ncol][k % ncol]; d = sel[ds]
        ax.plot(_covs, d["abst_err"], "-", color=COLORS["abstention"], lw=2,
                label="abstention (selective)")
        aseeds = d.get("abst_err_seeds")
        if aseeds is not None and len(aseeds) > 1:
            ax.fill_between(_covs, aseeds.min(0), aseeds.max(0),
                            color=COLORS["abstention"], alpha=0.22, lw=0, label="min-max over seeds")
        ax.axhline(d["ce_err"], color="#333", ls="--", lw=1.4, label="no rejection (full coverage)")
        ax.set_title(f"{ds} (Bayes {d['bayes']:.3f}, {d.get('n_seeds', 1)} seeds)", fontsize=10)
        ax.set_xlabel("coverage"); ax.set_ylabel("accepted error"); ax.grid(alpha=0.25)
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig_selective_risk"); plt.show()
else:
    print("no checkpoints loaded (see the compute cell above)")
'''

SEL_ACC_PLOT = '''# Selective-ACCURACY curves (reject threshold swept) per dataset, ordered easy -> hard.
# Exactly the risk plot above with accuracy = 1 - accepted error on the y-axis. abstention
# (reject high reservation) vs the model's own softmax confidence vs the CE full-coverage accuracy
# and the Bayes ceiling. Both curves come from the SAME abstention model (reject head vs its
# own softmax confidence) and meet the full-coverage accuracy at coverage 1.
if len(sel):
    order = list(sel_table["dataset"])
    ncol = min(5, len(order)); nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for k, ds in enumerate(order):
        ax = axes[k // ncol][k % ncol]; d = sel[ds]
        ax.plot(_covs, 1 - d["abst_err"], "-", color=COLORS["abstention"], lw=2, label="abstention")
        ax.plot(_covs, 1 - d["conf_err"], "--", color=COLORS["cascade"], lw=1.8, label="softmax confidence")
        ax.axhline(1 - d["ce_err"], color="#999", ls=":", lw=1.4, label="CE full coverage")
        ax.axhline(1 - d["bayes"], color="k", ls="-", lw=1.0, alpha=0.6, label="Bayes ceiling")
        ax.set_title(f"{ds} (Bayes {1 - d['bayes']:.3f})", fontsize=10)
        ax.set_xlabel("coverage"); ax.set_ylabel("selective accuracy"); ax.grid(alpha=0.25)
    for k in range(len(order), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); save_fig(fig, "fig_selective_accuracy"); plt.show()
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

OVERLAP = '''# WHY the hardest problem is harder: the exact class posteriors p(y|x) overlap more on
# harder datasets, so even the Bayes-optimal classifier errs more. LEFT: the top class
# posterior (max_y p(y|x)) for the easiest vs hardest dataset -- the hard one shifts
# toward chance (1/K), i.e. the classes are less separable. RIGHT: two raw process
# features colored by defect class on the hardest dataset -- the class clouds overlap.
# Reads only the dataset CSVs (data/cluster/*.csv); no model needed.
csvs = {d: root / "data" / "cluster" / f"{d}.csv" for d in manifest["datasets"]}
csvs = {d: p for d, p in csvs.items() if p.exists()}
if csvs:
    def _bayes_err(path):
        dd = pd.read_csv(path, usecols=lambda c: c.startswith("p_"))
        post = dd.to_numpy()
        return float((1 - post.max(1)).mean())
    berr = {d: _bayes_err(p) for d, p in csvs.items()}
    easy, hard = min(berr, key=berr.get), max(berr, key=berr.get)
    de, dh = pd.read_csv(csvs[easy]), pd.read_csv(csvs[hard])
    pc = [c for c in de.columns if c.startswith("p_")]
    cols = de.columns.tolist(); feat = cols[:cols.index(pc[0])]   # raw features precede p_*
    K = len(pc)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    for dd, lab, col in [(de, f"easiest: {easy} (Bayes err {berr[easy]:.3f})", COLORS["cascade"]),
                         (dh, f"hardest: {hard} (Bayes err {berr[hard]:.3f})", COLORS["abstention"])]:
        top = dd[pc].to_numpy().max(1)
        a0.hist(top, bins=40, range=(1.0 / K, 1.0), density=True, alpha=0.6, color=col, label=lab)
    a0.axvline(1.0 / K, color="k", ls=":", lw=1, label=f"chance = 1/{K}")
    a0.set_xlabel("top class posterior  max_y p(y|x)"); a0.set_ylabel("density")
    a0.set_title("Class posteriors overlap more on harder data"); a0.legend(fontsize=9)

    sc = dh.sample(min(len(dh), 4000), random_state=0)
    for lab in sorted(sc["defect_label"].unique()):
        m = sc["defect_label"] == lab
        a1.scatter(sc.loc[m, feat[0]], sc.loc[m, feat[1]], s=7, alpha=0.4, label=lab)
    a1.set_xlabel(feat[0]); a1.set_ylabel(feat[1])
    a1.set_title(f"Features overlap by defect class ({hard})")
    a1.legend(fontsize=8, markerscale=2)
    fig.tight_layout(); save_fig(fig, "fig_posterior_overlap"); plt.show()
else:
    print("no dataset CSVs found (need data/cluster/*.csv on the cluster)")
'''

HARD_HIST = '''# TWO histograms tying difficulty together, datasets ordered easy -> hard by Bayes
# error. LEFT: forced accuracy FALLS as the task gets harder. RIGHT: the accuracy
# GAINED by abstaining RISES as the task gets harder -- so abstention earns its keep
# exactly where the plain classifier struggles. (Needs the selective-compute cell for
# `sel_table`.)
if len(sel_table):
    t = sel_table.sort_values("bayes_err").reset_index(drop=True)
    core = study("core")
    acc = {d: core[(core.dataset == d) & (core.loss == "cascade")]["defect_acc"].mean()
           for d in t["dataset"]}
    order = list(t["dataset"]); x = np.arange(len(order))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    a0.bar(x, [acc[d] for d in order], color=COLORS["cascade"])
    a0.set_xticks(x); a0.set_xticklabels(order, rotation=30, ha="right")
    a0.set_ylabel("forced defect accuracy"); a0.grid(axis="x", alpha=0)
    a0.set_ylim(max(0.0, min(acc.values()) - 0.05), 1.0)
    a0.set_title("Accuracy falls as the task gets harder")
    a1.bar(x, t["gain_vs_ce"], color=COLORS["abstention"])
    a1.set_xticks(x); a1.set_xticklabels(order, rotation=30, ha="right")
    a1.axhline(0, color="k", lw=0.8); a1.grid(axis="x", alpha=0)
    a1.set_ylabel(f"accuracy gained by abstaining @ cov~{SEL_COVERAGE:.2f}")
    a1.set_title("Abstention gains more as the task gets harder")
    fig.tight_layout(); save_fig(fig, "fig_hardness_hist"); plt.show()
else:
    print("run the selective-compute cell first (needs sel_table)")
'''

FEATSEP = '''# FEATURE SEPARABILITY: per-feature class-conditional densities, drawn for the BASELINE
# (well-separated) and the HARDEST dataset (more overlap) so you can VISUALLY compare the
# increased overlap. LOW open/bridge overlap => the feature separates the defects; HIGH
# overlap => it carries little signal, which is what makes the task hard. Dashed lines are
# the spec limits (lsl / usl). Reads only the dataset CSVs + the spec YAML.
csvs = {d: root / "data" / "cluster" / f"{d}.csv" for d in manifest["datasets"]}
csvs = {d: p for d, p in csvs.items() if p.exists()}
if csvs:
    def _berr(p):
        dd = pd.read_csv(p, usecols=lambda c: c.startswith("p_"))
        return float((1 - dd.to_numpy().max(1)).mean())

    # spec limits (lsl/usl) + the monitored feature ids from the YAML (no pyyaml dep)
    lims, cur, in_p = {}, None, False
    for line in (root / "domain" / "smt_paper.yaml").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            in_p = line.strip().startswith("parameters:"); cur = None; continue
        if not in_p:
            continue
        s = line.strip()
        if s.startswith("- id:"):
            cur = s.split("id:", 1)[1].split("#")[0].strip(); lims[cur] = []
        elif cur and (s.startswith("lsl:") or s.startswith("usl:")):
            lims[cur].append(float(s.split(":", 1)[1].split("#")[0]))
    CLS = [("no_defect", "no defect", "#9e9e9e"),
           ("open_circuit", "open circuit", COLORS["cascade"]),
           ("solder_bridging", "solder bridging", COLORS["abstention"])]

    def featsep(ds):
        dh = pd.read_csv(csvs[ds])
        feats = [f for f in lims if f in dh.columns]     # exactly the process features, spec order
        ncol = 3; nrow = int(np.ceil(len(feats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
        for k, f in enumerate(feats):
            ax = axes[k // ncol][k % ncol]
            bins = np.linspace(dh[f].min(), dh[f].max(), 60)
            hists = {}
            for key, lab, col in CLS:
                v = dh.loc[dh["defect_label"] == key, f].to_numpy()
                if len(v):
                    ax.hist(v, bins=bins, density=True, alpha=0.55, color=col, label=lab)
                    h, _ = np.histogram(v, bins=bins); hists[key] = h / max(h.sum(), 1)
            if "open_circuit" in hists and "solder_bridging" in hists:
                ov = float(np.minimum(hists["open_circuit"], hists["solder_bridging"]).sum())
            else:
                ov = float("nan")
            for lim in lims.get(f, []):
                ax.axvline(lim, ls="--", color="k", lw=1, alpha=0.7)
            ax.set_title(f"{f}\\nopen/bridge overlap = {ov:.2f}", fontsize=10); ax.grid(alpha=0.2)
        for k in range(len(feats), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        axes[0][0].legend(fontsize=9)
        fig.suptitle(f"Feature values by defect class (density) -- {ds} (Bayes err {_berr(csvs[ds]):.3f})"
                     "  --  low overlap = the class colors separate", fontweight="bold")
        fig.tight_layout(); save_fig(fig, f"fig_feature_separability_{ds}"); plt.show()

    hardest = max(csvs, key=lambda d: _berr(csvs[d]))
    baseline = "baseline" if "baseline" in csvs else min(csvs, key=lambda d: _berr(csvs[d]))
    for ds in dict.fromkeys([baseline, hardest]):        # baseline first, dedup if same
        featsep(ds)
else:
    print("no dataset CSVs found (need data/cluster/*.csv on the cluster)")
'''

CAS_VS_ABST = '''# Accuracy comparison: the non-abstention model at full coverage vs the abstention model
# at a PER-DATASET operating point that best trades coverage for accuracy. For each dataset we
# pick the coverage maximizing  net = (selective_acc - non_abstention_acc) - PENALTY*(1 - coverage);
# PENALTY tunes the mix (lower -> abstain more / lower coverage; higher -> keep coverage high).
# Selective curves come from `sel` (rank-based, from the compute cell); the non-abstention
# accuracy is the "cascade"-loss model at full coverage. Bars = mean, error bars = +/- 1 std.
PENALTY = 0.20
COV_MIN = 0.60
core = study("core")


def _ms(frame, ds, col):
    """(mean, std) over seeds for one dataset/column."""
    v = frame[frame.dataset == ds][col].dropna()
    return (float(v.mean()), float(v.std())) if len(v) else (np.nan, np.nan)


BAR_DATASETS = ["baseline", "harder"]                              # paper datasets
rows = []
for ds in [d for d in sel if d in set(core.dataset) and d in BAR_DATASETS]:
    d = sel[ds]
    sel_acc = 1 - d["abst_err"]                                     # selective accuracy at each coverage
    na_m, na_s = _ms(core[core.loss == "cascade"], ds, "defect_acc")   # non-abstention model
    net = (sel_acc - na_m) - PENALTY * (1 - _covs)
    net = np.where(_covs >= COV_MIN, net, -np.inf)                  # avoid degenerate low coverage
    j = int(np.argmax(net))
    seeds = d.get("abst_err_seeds")
    ab_s = float((1 - seeds[:, j]).std()) if seeds is not None and len(seeds) > 1 else 0.0
    rows.append((ds, na_m, na_s, float(sel_acc[j]), ab_s, float(_covs[j])))

rows.sort(key=lambda r: r[1])                                       # low -> high non-abstention accuracy
labels = [r[0] for r in rows]; x = np.arange(len(labels)); w = 0.38
na_m = [r[1] for r in rows]; na_s = [r[2] for r in rows]
ab_m = [r[3] for r in rows]; ab_s = [r[4] for r in rows]; cov = [r[5] for r in rows]

fig, ax = plt.subplots(figsize=(11, 6))
ekw = dict(ecolor="0.3", capsize=3, elinewidth=1)
ax.bar(x - w/2, na_m, w, yerr=na_s, color=COLORS["cascade"],
       label="non-abstention (full coverage)", error_kw=ekw)
ax.bar(x + w/2, ab_m, w, yerr=ab_s, color=COLORS["abstention"],
       label="abstention (selective)", error_kw=ekw)
for xi, a, s, c in zip(x, ab_m, ab_s, cov):
    ax.annotate(f"cov {c:.2f}", (xi + w/2, a + (0 if np.isnan(s) else s)),
                ha="center", va="bottom", fontsize=8)
allv = [v for v in na_m + ab_m if not np.isnan(v)]
ax.set_ylim(max(0.0, min(allv) - 0.04), 1.0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy comparison")
ax.legend(loc="upper left"); ax.grid(axis="x", alpha=0)
save_fig(fig, "fig_cascade_vs_abstention_selective"); plt.show()
print("\\n".join(f"{r[0]:>14}: non-abstention {r[1]:.4f}   abstention {r[3]:.4f} @ cov {r[5]:.2f}   gain {r[3]-r[1]:+.4f}" for r in rows))
'''

SELVO = '''# Selective accuracy vs the payoff o, on balanced_hard (the dataset that under-fits at low o).
# Each o's abstention model operates at its own reject decision but never below a coverage FLOOR;
# selective accuracy is the accuracy on the kept rows. As o rises the model abstains less and the
# curve settles onto the plain classifier -- so it rises to an interior peak, then comes back down.
# Needs results/cluster/*.pt + data/cluster/*.csv.
SEL_DS = "balanced_hard"
FLOOR = 0.60
SMOOTH_WIN = 3                # rolling-mean window for display smoothing (1 = raw, no smoothing)
pay_runs_all = [r for r in manifest["runs"] if "payoff" in (r.get("studies") or [])]
SELVO_SEEDS = sorted({r["seed"] for r in pay_runs_all})

pay_ds = sorted({r["dataset"] for r in pay_runs_all})
if SEL_DS not in pay_ds:      # fall back to the hardest-by-Bayes payoff dataset if not present
    def _bayes_acc(d):
        for r in pay_runs_all:
            if r["dataset"] == d and (root / r["metrics"]).exists():
                return json.loads((root / r["metrics"]).read_text()).get("bayes_optimal_accuracy", 1.0)
        return 1.0
    SEL_DS = min(pay_ds, key=_bayes_acc)
ds = SEL_DS
o_grid = sorted({r["o"] for r in pay_runs_all if r["loss"] == "abstention" and r["dataset"] == ds})

rows = []
csv = root / "data" / "cluster" / f"{ds}.csv"
if csv.exists():
    dfd = pd.read_csv(csv)
    for o in o_grid:
        for seed in SELVO_SEEDS:
            run = _find(ds, "abstention", o, seed)
            if not (run and (root / run["model"]).exists()):
                continue
            info = _infer(run, dfd)
            if not info["abstain"]:
                continue
            correct = (info["argmax"] == info["y"]).astype(float)
            acc = 1 - _risk_cov(-info["r"], correct)
            op_cov = max(float((info["r"] < 0.5).mean()), FLOOR)   # own coverage, never below the floor
            rows.append({"o": o, "sel": float(np.interp(op_cov, _covs, acc))})
selvo = pd.DataFrame(rows)
if len(selvo):
    sm = selvo.groupby("o")["sel"].mean().reset_index().sort_values("o").reset_index(drop=True)
    o_arr = sm["o"].to_numpy()
    sel_s = sm["sel"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()
    peak_i = int(np.nanargmax(sel_s)); peak_o = float(o_arr[peak_i])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(o_arr, sel_s, marker="o", color="#1f77b4", lw=2)
    ax.axvline(peak_o, color="#1f77b4", ls=":", lw=1)
    ax.annotate(f"best o={peak_o:g}", (peak_o, sel_s[peak_i]),
                textcoords="offset points", xytext=(6, 6), color="#1f77b4")
    ax.set_xlabel("payoff  o"); ax.set_ylabel("selective accuracy (kept rows)")
    ax.set_title("Selective accuracy vs o")
    ax.grid(alpha=0.3)
    save_fig(fig, "fig_selective_vs_o"); plt.show()
    print(sm.round(4).to_string(index=False))
else:
    print(f"no abstention checkpoints for {ds}")
'''

# (markdown header, code) in notebook order
SECTIONS = [
    ("## Load every run\\n\\nLoads the manifest + each run's metrics into one dataframe, with helpers to "
     "slice by study and average over seeds.", LOAD),
    ("## Plot style\\n\\nShared paper style + a `save_fig` helper that writes `figs/fig_*.{png,pdf}`.", STYLE),
    ("## Why the hardest problem is harder (feature separability)\\n\\nPer-feature class-conditional "
     "densities for the **baseline** vs the **hardest** dataset -- visually compare the increased "
     "overlap. Low open/bridge overlap = the feature separates the defects; high overlap = it carries "
     "little signal, which is what makes the task hard. **Saved: `fig_feature_separability_<dataset>`.**",
     FEATSEP),
    ("## Risk vs coverage\\n\\nThe selective-classification figure: coverage vs selective accuracy as the "
     "payoff `o` sweeps (one operating point per model); star = non-abstention at full coverage. "
     "**Saved: `fig_risk_coverage`.**", RISK_COVERAGE),
    ("## Selective-classification analysis (threshold-swept)\\n\\nLoads the saved checkpoints and sweeps "
     "the **rejection threshold** on each model (the standard selective-risk view). Runs on the cluster "
     "(needs `results/cluster/*.pt` + `data/cluster/*.csv`); builds `sel_table` + the `_find/_infer/"
     "_risk_cov` helpers used below.", SEL_COMPUTE),
    ("## Selective accuracy vs the payoff o\\n\\nSelective accuracy (kept rows) vs the payoff `o` on "
     "`balanced_hard`, from each `o`'s checkpoint. It rises to an interior peak, then comes back down as "
     "the model stops abstaining. **Saved: `fig_selective_vs_o`.**", SELVO),
    ("## Accuracy comparison (non-abstention vs abstention)\\n\\nThe non-abstention model at full coverage vs abstention's "
     "selective accuracy on the boards it answers (`r<0.5`), per dataset, coverage annotated. "
     "**Saved: `fig_cascade_vs_abstention_selective`.**", CAS_VS_ABST),
    ("## Comparing the datasets: selective-risk curves\\n\\nAccepted error vs coverage on `baseline` + "
     "`harder`: abstention (min-max seed band) vs the no-rejection line. **Saved: `fig_selective_risk`.**",
     SEL_RISK_PLOT),
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
    out = Path(__file__).resolve().parent.parent / "synthetic_analysis.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    n_code = sum(1 for _, s in SECTIONS)
    print(f"wrote {out}  ({n_code} code cells)")


if __name__ == "__main__":
    main()
