"""Distribution histograms and difficulty/comparison metrics for the SMT data.

Two jobs:
  (1) Histograms of how the classes/labels are distributed (defect classes,
      per-stage mechanisms) plus per-parameter distributions and class-conditional
      overlays.
  (2) Difficulty + paper-comparison metrics for the defect head:
        - exact Bayes error from the known posterior (the oracle ceiling),
        - a basic single-head MLP (same model family as the paper),
        - the Bayes-optimal classifier (argmax posterior),
      each reported as accuracy AND F1 (weighted + macro + per-class), so we can
      line up directly against the paper's Section V-A (95.00% accuracy,
      95.36% weighted-F1).
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
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from generator import defect_names, load_spec, param_ids

POST_PREFIX = "p_"

# Paper's reported defect-head numbers (Section V-A), for comparison.
PAPER_REF = {"accuracy": 0.9500, "weighted_f1": 0.9536}

DEFECT_COLORS = {
    "no_defect": "#8a8d91",
    "open_circuit": "#2f6db5",
    "solder_bridging": "#c1432e",
}
SPEC_RED = "#c1432e"


def _apply_style() -> None:
    """Consistent, presentation-grade matplotlib defaults."""

    # tighten fonts, drop top/right spines, add a faint grid
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    })


def _param_causal_note(spec: dict) -> dict[str, str]:
    """Map each parameter to a short note of its causal role."""

    # collect the (direction -> defect) edges that touch each parameter
    notes: dict[str, list[str]] = {p["id"]: [] for p in spec["parameters"]}
    short = {"solder_bridging": "bridging", "open_circuit": "open"}
    for e in spec["causal_edges"]:
        notes[e["parameter"]].append(f"{e['direction']}->{short.get(e['defect'], e['defect'])}")

    # parameters with no edge are nuisance variables
    return {p: (", ".join(sorted(set(v))) if v else "nuisance (no causal edge)")
            for p, v in notes.items()}


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #


def plot_class_balance(df: pd.DataFrame, spec: dict, out: Path) -> dict:
    """Bar chart of the realized defect distribution vs the spec priors."""

    # realized fraction per class and the spec prior beside it
    names = defect_names(spec)
    counts = df["defect_label"].value_counts().reindex(names).fillna(0).astype(int)
    fracs = counts / counts.sum()
    priors = {d["name"]: d["prior"] for d in spec["defects"]}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, fracs.values, width=0.4, label="realized", color="#2f6db5")
    ax.bar(x + 0.2, [priors[n] for n in names], width=0.4, label="spec prior",
           color="#c9ccd1")

    # number each bar
    for i, n in enumerate(names):
        ax.text(i - 0.2, fracs[n] + 0.008, f"{fracs[n]:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, priors[n] + 0.008, f"{priors[n]:.3f}", ha="center",
                fontsize=9, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names])
    ax.set_ylabel("fraction of records")
    ax.set_ylim(0, max(fracs.max(), max(priors.values())) * 1.15)
    ax.set_title("Defect class distribution — realized vs spec prior")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return {n: float(fracs[n]) for n in names}


def plot_mechanism_balance(df: pd.DataFrame, spec: dict, out: Path) -> dict:
    """Per-stage bar charts of the mechanism-label distribution."""

    stages = list(spec["mechanisms"].keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(7 * len(stages), 5),
                             sharey=True)
    result = {}
    for ax, stage in zip(np.atleast_1d(axes), stages):
        # fraction of each mechanism in this stage
        col = f"{stage}_mechanism_label"
        order = spec["mechanisms"][stage]
        fracs = df[col].value_counts(normalize=True).reindex(order).fillna(0)

        # grey for no_mechanism, blue for the real ones
        colors = ["#8a8d91"] + ["#2f6db5"] * (len(order) - 1)
        bars = ax.bar(range(len(order)), fracs.values, color=colors)
        for b, v in zip(bars, fracs.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([m.replace("_", "\n") for m in order], fontsize=9)
        ax.set_title(stage.replace("stage_", "").capitalize() + " stage mechanisms")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        result[stage] = {k: float(v) for k, v in fracs.items()}
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return result


def plot_parameter_hists(df: pd.DataFrame, spec: dict, out: Path) -> None:
    """Per-parameter value histograms with spec limits and out-of-spec rate."""

    params = spec["parameters"]
    notes = _param_causal_note(spec)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.ravel(), params):
        vals = df[p["id"]]

        # share of records outside the spec window
        oos = float(((vals < p["lsl"]) | (vals > p["usl"])).mean())
        ax.hist(vals, bins=80, color="#6aa84f", alpha=0.85)

        # shade the out-of-spec regions and mark nominal + limits
        lo, hi = ax.get_xlim()
        ax.axvspan(lo, p["lsl"], color=SPEC_RED, alpha=0.06)
        ax.axvspan(p["usl"], hi, color=SPEC_RED, alpha=0.06)
        ax.set_xlim(lo, hi)
        ax.axvline(p["nominal"], color="k", linewidth=1)
        for b in ("lsl", "usl"):
            ax.axvline(p[b], color=SPEC_RED, linestyle="--", linewidth=1.2)
        ax.set_title(f"{p['id']} ({p['unit']})")
        ax.set_xlabel(f"{notes[p['id']]}   |   out-of-spec: {oos*100:.2f}%",
                      fontsize=9, color="#444")
        ax.set_yticks([])
    fig.suptitle("Process parameter distributions  (black = nominal, red dashed = spec limits, shaded = out-of-spec)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_class_conditional(df: pd.DataFrame, spec: dict, out: Path) -> None:
    """Per-parameter distribution split by defect class (shows separability)."""

    params = spec["parameters"]
    names = defect_names(spec)
    notes = _param_causal_note(spec)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.ravel(), params):
        # one outline density per class
        for n in names:
            sub = df.loc[df["defect_label"] == n, p["id"]]
            ax.hist(sub, bins=60, density=True, histtype="step", linewidth=1.6,
                    color=DEFECT_COLORS.get(n), label=n.replace("_", " "))

        # grey out the title for nuisance parameters
        nuisance = "nuisance" in notes[p["id"]]
        ax.set_title(p["id"] + ("  [nuisance]" if nuisance else ""),
                     color="#999" if nuisance else "black")
        ax.set_xlabel(notes[p["id"]], fontsize=9, color="#444")
        ax.set_yticks([])
    axes.ravel()[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Class-conditional parameter distributions (density) — separation reveals which parameters carry defect signal",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_bayes_summary(exact: dict, metrics: dict, out: Path) -> None:
    """Headline error chart: our learner vs the paper vs the Bayes floor."""

    # test-set error for each reference point
    floor = exact["exact_mc_bayes_error_test_split"]
    paper_err = 1.0 - PAPER_REF["accuracy"]
    mlp_err = metrics["mlp_basic_test_error"]
    rows = [
        ("Majority baseline\n(always no_defect)", metrics["majority_baseline_error"], "#c9ccd1"),
        ("Basic MLP\n(ours)", mlp_err, "#2f6db5"),
        ("Paper\n(reported MLP)", paper_err, "#e08214"),
        ("Bayes floor\n(oracle, test split)", floor, "#6aa84f"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.002, yi, f"{v:.4f}", va="center", fontsize=10)

    # dashed line at the irreducible floor
    ax.axvline(floor, color="#6aa84f", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("test-set classification error")
    ax.set_title("Defect-head difficulty: our data vs the paper vs the Bayes floor", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_defect_metrics(metrics: dict, out: Path) -> None:
    """Grouped bars comparing accuracy and weighted-F1: ours, ceiling, paper."""

    # series across the two headline metrics
    series = [
        ("Ours (basic MLP)", metrics["mlp_basic"], "#2f6db5"),
        ("Bayes-optimal (ceiling)", metrics["bayes_optimal"], "#6aa84f"),
        ("Paper (reported)", {"accuracy": PAPER_REF["accuracy"],
                              "weighted_f1": PAPER_REF["weighted_f1"]}, "#e08214"),
    ]
    keys = [("accuracy", "Accuracy"), ("weighted_f1", "Weighted F1")]
    x = np.arange(len(keys))
    w = 0.25
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(8, 5))
    for s, (label, vals, color) in enumerate(series):
        heights = [vals[k] for k, _ in keys]
        bars = ax.bar(x + offsets[s], heights, width=w, label=label, color=color)
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.003, f"{h:.3f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Defect-head metrics: our data vs the paper")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Bayes error + defect-head metrics
# --------------------------------------------------------------------------- #


def bayes_error_exact(df: pd.DataFrame, spec: dict) -> dict:
    """Algorithm — Exact Bayes error from the known posterior.

    Input: df with posterior columns p_<class>, spec.
    Return: the irreducible error E_x[1 - max_y p(y|x)] (overall and on the test split).
    """

    # per-record irreducible error = 1 - probability of the most likely class
    names = defect_names(spec)
    post = df[[f"{POST_PREFIX}{n}" for n in names]].to_numpy()
    per_record = 1.0 - post.max(axis=1)
    overall = float(per_record.mean())

    # finite-sample cross-check: Bayes-optimal predictions vs the sampled labels
    pred = np.array(names)[post.argmax(axis=1)]
    y = df["defect_label"].to_numpy()
    realized_argmax_err = float((pred != y).mean())

    # the same floor restricted to the test split (for like-for-like comparison)
    test_mask = (df["split"] == "test").to_numpy()
    test_split = float(per_record[test_mask].mean()) if test_mask.any() else None
    return {
        "exact_mc_bayes_error": overall,
        "exact_mc_bayes_error_test_split": test_split,
        "bayes_optimal_vs_sampled_labels": realized_argmax_err,
    }


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    """Accuracy, weighted/macro F1, and per-class precision/recall/F1."""

    # overall accuracy and the two F1 averages
    acc = float(accuracy_score(y_true, y_pred))
    wf1 = float(f1_score(y_true, y_pred, labels=names, average="weighted", zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0))

    # precision / recall / F1 for each class
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=names, average=None, zero_division=0)
    per_class = {n: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(s[i])}
                 for i, n in enumerate(names)}
    return {"accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "per_class": per_class}


def defect_head_metrics(df: pd.DataFrame, spec: dict, model_seed: int = 0) -> dict:
    """Algorithm — Defect-head classification metrics (to compare with the paper).

    Input: dataset df, spec, seed.
    Return: majority baseline, a basic MLP (same model family as the paper), and
            the Bayes-optimal ceiling — each with accuracy and F1.
    """

    names = defect_names(spec)
    ids = param_ids(spec)

    # batch-grouped train/test split (no leakage; done upstream)
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    Xtr, ytr = tr[ids].to_numpy(), tr["defect_label"].to_numpy()
    Xte, yte = te[ids].to_numpy(), te["defect_label"].to_numpy()

    # trivial baseline: always predict the most common class
    majority = df["defect_label"].mode().iloc[0]
    majority_error = float((yte != majority).mean())

    # a basic MLP (same model family as the paper); MLPs need scaled features
    sc = StandardScaler().fit(Xtr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        max_iter=500, early_stopping=True, n_iter_no_change=12,
                        random_state=model_seed)
    mlp.fit(sc.transform(Xtr), ytr)
    mlp_metrics = _clf_metrics(yte, mlp.predict(sc.transform(Xte)), names)

    # Bayes-optimal classifier: argmax of the known posterior (the ceiling)
    post_te = te[[f"{POST_PREFIX}{n}" for n in names]].to_numpy()
    bo_pred = np.array(names)[post_te.argmax(axis=1)]
    bo_metrics = _clf_metrics(yte, bo_pred, names)

    return {
        "majority_baseline_error": majority_error,
        "majority_class": str(majority),
        "mlp_basic": mlp_metrics,
        "mlp_basic_test_error": 1.0 - mlp_metrics["accuracy"],
        "bayes_optimal": bo_metrics,
        "paper_reference": dict(PAPER_REF),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="SMT data analysis: histograms + metrics")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--out", default="results/bayes_error.json")
    args = ap.parse_args()

    # load spec + data, set the plot style
    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # distribution figures
    class_fracs = plot_class_balance(df, spec, figs / "class_balance.png")
    mech_fracs = plot_mechanism_balance(df, spec, figs / "mechanism_balance.png")
    plot_parameter_hists(df, spec, figs / "parameter_hists.png")
    plot_class_conditional(df, spec, figs / "class_conditional.png")

    # difficulty + paper-comparison metrics
    exact = bayes_error_exact(df, spec)
    metrics = defect_head_metrics(df, spec, model_seed=spec["generator"]["seed"])
    plot_bayes_summary(exact, metrics, figs / "bayes_summary.png")
    plot_defect_metrics(metrics, figs / "defect_metrics.png")

    # write the metrics file
    summary = {
        "n_records": int(len(df)),
        "class_fractions": class_fracs,
        "mechanism_fractions": mech_fracs,
        "bayes_error": exact,
        "defect_head_metrics": metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    # console summary
    mlp, bo, ref = metrics["mlp_basic"], metrics["bayes_optimal"], PAPER_REF
    print("=== Class distribution ===")
    for k, v in class_fracs.items():
        print(f"  {k:18s} {v:.4f}")
    print("\n=== Bayes error (oracle ceiling) ===")
    print(f"  exact E[1 - max_y p(y|x)]  (all) : {exact['exact_mc_bayes_error']:.4f}")
    print(f"  exact Bayes error    (test split): {exact['exact_mc_bayes_error_test_split']:.4f}")
    print("\n=== Defect-head metrics (test split) — compare to the paper ===")
    print(f"  {'':22s} {'accuracy':>9s} {'wF1':>8s} {'macroF1':>8s}")
    print(f"  {'Ours (basic MLP)':22s} {mlp['accuracy']:9.4f} {mlp['weighted_f1']:8.4f} {mlp['macro_f1']:8.4f}")
    print(f"  {'Bayes-optimal ceiling':22s} {bo['accuracy']:9.4f} {bo['weighted_f1']:8.4f} {bo['macro_f1']:8.4f}")
    print(f"  {'Paper (reported)':22s} {ref['accuracy']:9.4f} {ref['weighted_f1']:8.4f} {'-':>8s}")
    print(f"  majority baseline error: {metrics['majority_baseline_error']:.4f}")
    print("\n  per-class F1 (MLP / Bayes-optimal):")
    for n in defect_names(spec):
        print(f"    {n:18s} {mlp['per_class'][n]['f1']:.4f} / {bo['per_class'][n]['f1']:.4f}")
    print(f"\nWrote {args.out} and 6 figures to {figs}/")


if __name__ == "__main__":
    main()
