"""Distribution histograms and Bayes-error estimates for the SMT synthetic data.

Two deliverables:

  (1) Histograms of how the classes/labels are distributed (defect classes,
      per-stage mechanisms) and the per-parameter value distributions with spec
      limits, plus class-conditional overlays showing separability.

  (2) Bayes-error estimates — how hard the classification problem is, and the
      target error a well-trained model should approach. Three independent
      methods that should agree (or bracket):

        - EXACT Monte-Carlo: because the generator assigns labels by sampling
          y ~ Categorical(p(y|x)) with a KNOWN posterior p(y|x), the Bayes error
          is E_x[1 - max_y p(y|x)], computed directly from the posterior columns.
          This is exact up to MC variance (negligible at N=200k).
        - kNN test error: an empirical classifier error; converges to the Bayes
          error from above and brackets it via the asymptotic NN bound.
        - HistGradientBoosting test error: a strong learner; its error is an
          upper bound on the Bayes error and should sit just above the exact value.

The model sees exactly the features the posterior is built from (the 6 process
parameters), so there is no information mismatch: all three target the same
Bayes error.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from generator import defect_names, load_spec, param_ids

POST_PREFIX = "p_"

DEFECT_COLORS = {
    "no_defect": "#8a8d91",
    "open_circuit": "#2f6db5",
    "solder_bridging": "#c1432e",
}
SPEC_RED = "#c1432e"


def _apply_style() -> None:
    """Consistent, presentation-grade matplotlib defaults."""
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
    """Map each parameter -> a short human note of its causal role."""
    notes: dict[str, list[str]] = {p["id"]: [] for p in spec["parameters"]}
    short = {"solder_bridging": "bridging", "open_circuit": "open"}
    for e in spec["causal_edges"]:
        notes[e["parameter"]].append(f"{e['direction']}->{short.get(e['defect'], e['defect'])}")
    return {p: (", ".join(sorted(set(v))) if v else "nuisance (no causal edge)")
            for p, v in notes.items()}


def plot_class_balance(df: pd.DataFrame, spec: dict, out: Path) -> dict:
    names = defect_names(spec)
    counts = df["defect_label"].value_counts().reindex(names).fillna(0).astype(int)
    fracs = counts / counts.sum()
    priors = {d["name"]: d["prior"] for d in spec["defects"]}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, fracs.values, width=0.4, label="realized", color="#2f6db5")
    ax.bar(x + 0.2, [priors[n] for n in names], width=0.4, label="spec prior",
           color="#c9ccd1")
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
    stages = list(spec["mechanisms"].keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(7 * len(stages), 5),
                             sharey=True)
    result = {}
    for ax, stage in zip(np.atleast_1d(axes), stages):
        col = f"{stage}_mechanism_label"
        order = spec["mechanisms"][stage]
        fracs = df[col].value_counts(normalize=True).reindex(order).fillna(0)
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
    params = spec["parameters"]
    notes = _param_causal_note(spec)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.ravel(), params):
        vals = df[p["id"]]
        oos = float(((vals < p["lsl"]) | (vals > p["usl"])).mean())
        ax.hist(vals, bins=80, color="#6aa84f", alpha=0.85)
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
    """Per-parameter distribution split by defect class — visual separability."""
    params = spec["parameters"]
    names = defect_names(spec)
    notes = _param_causal_note(spec)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.ravel(), params):
        for n in names:
            sub = df.loc[df["defect_label"] == n, p["id"]]
            ax.hist(sub, bins=60, density=True, histtype="step", linewidth=1.6,
                    color=DEFECT_COLORS.get(n), label=n.replace("_", " "))
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


def plot_bayes_summary(exact: dict, bounds: dict, brackets: dict, out: Path) -> None:
    """Headline chart: learned-classifier test errors vs the Bayes floor."""
    floor = exact["exact_mc_bayes_error_test_split"]
    rows = [
        ("Majority baseline\n(always no_defect)", bounds["majority_baseline_error"], "#c9ccd1"),
        ("kNN (k=15)", bounds["knn15_test_error"], "#2f6db5"),
        ("HistGradientBoosting", bounds["hist_gradient_boosting_test_error"], "#2f6db5"),
        ("Bayes floor\n(irreducible, test split)", floor, "#6aa84f"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.002, yi, f"{v:.4f}", va="center", fontsize=10)
    ax.axvline(floor, color="#6aa84f", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("test-set classification error")
    ax.set_title("How hard is the problem? Strong learners reach the Bayes floor\n"
                 "(gap above the floor is irreducible label noise)", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Bayes error
# --------------------------------------------------------------------------- #


def bayes_error_exact(df: pd.DataFrame, spec: dict) -> dict:
    """Exact Bayes error from the known posterior: E_x[1 - max_y p(y|x)]."""
    names = defect_names(spec)
    post = df[[f"{POST_PREFIX}{n}" for n in names]].to_numpy()
    per_record = 1.0 - post.max(axis=1)
    overall = float(per_record.mean())

    # Bayes-optimal predictions vs sampled labels (finite-sample cross-check).
    pred = np.array(names)[post.argmax(axis=1)]
    y = df["defect_label"].to_numpy()
    realized_argmax_err = float((pred != y).mean())

    # Per-true-class irreducible error (mean 1 - p(true class | x) restricted).
    per_class = {}
    for n in names:
        mask = y == n
        per_class[n] = float(per_record[mask].mean()) if mask.any() else None

    # Same quantity restricted to the test split, for like-for-like comparison
    # with the learned-classifier test errors (the test split has a different
    # class balance than the full set).
    test_mask = (df["split"] == "test").to_numpy()
    test_split = float(per_record[test_mask].mean()) if test_mask.any() else None
    return {
        "exact_mc_bayes_error": overall,
        "exact_mc_bayes_error_test_split": test_split,
        "bayes_optimal_vs_sampled_labels": realized_argmax_err,
        "per_true_class_mean_irreducible": per_class,
    }


def classifier_upper_bounds(df: pd.DataFrame, spec: dict,
                            knn_train_cap: int = 50_000,
                            knn_test_cap: int = 20_000,
                            seed: int = 0) -> dict:
    """Empirical classifier errors (upper bounds on the Bayes error)."""
    ids = param_ids(spec)
    rng = np.random.default_rng(seed)
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    Xtr, ytr = tr[ids].to_numpy(), tr["defect_label"].to_numpy()
    Xte, yte = te[ids].to_numpy(), te["defect_label"].to_numpy()

    majority = df["defect_label"].mode().iloc[0]
    majority_err = float((yte != majority).mean())

    # HistGradientBoosting on the full training split.
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                        random_state=seed)
    gb.fit(Xtr, ytr)
    gb_err = float((gb.predict(Xte) != yte).mean())

    # kNN on standardized, subsampled data (asymptotic NN error brackets Bayes).
    sc = StandardScaler().fit(Xtr)
    itr = rng.choice(len(Xtr), min(knn_train_cap, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), min(knn_test_cap, len(Xte)), replace=False)
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(sc.transform(Xtr[itr]), ytr[itr])
    knn_err = float((knn.predict(sc.transform(Xte[ite])) != yte[ite]).mean())

    return {
        "majority_baseline_error": majority_err,
        "majority_class": str(majority),
        "hist_gradient_boosting_test_error": gb_err,
        "knn15_test_error": knn_err,
    }


def nn_bound_brackets(bayes: float, n_classes: int) -> dict:
    """Cover-Hart asymptotic 1-NN bound: R* <= R_NN <= R*(2 - c/(c-1) R*)."""
    c = n_classes
    upper = bayes * (2 - (c / (c - 1)) * bayes)
    return {"R_star": bayes, "asymptotic_1nn_error_upper": float(upper)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="SMT data analysis: histograms + Bayes error")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--out", default="results/bayes_error.json")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    _apply_style()

    class_fracs = plot_class_balance(df, spec, figs / "class_balance.png")
    mech_fracs = plot_mechanism_balance(df, spec, figs / "mechanism_balance.png")
    plot_parameter_hists(df, spec, figs / "parameter_hists.png")
    plot_class_conditional(df, spec, figs / "class_conditional.png")

    exact = bayes_error_exact(df, spec)
    bounds = classifier_upper_bounds(df, spec, seed=spec["generator"]["seed"])
    # Bracket the kNN/GB test errors against the TEST-split Bayes floor (same split).
    brackets = nn_bound_brackets(exact["exact_mc_bayes_error_test_split"],
                                 len(defect_names(spec)))
    plot_bayes_summary(exact, bounds, brackets, figs / "bayes_summary.png")

    summary = {
        "n_records": int(len(df)),
        "class_fractions": class_fracs,
        "mechanism_fractions": mech_fracs,
        "bayes_error": exact,
        "classifier_upper_bounds": bounds,
        "nn_asymptotic_bracket": brackets,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("=== Class distribution ===")
    for k, v in class_fracs.items():
        print(f"  {k:18s} {v:.4f}")
    print("\n=== Bayes error (how hard is the problem) ===")
    print(f"  exact MC  E[1 - max_y p(y|x)] (all): {exact['exact_mc_bayes_error']:.4f}")
    print(f"  exact MC  Bayes error (test split) : {exact['exact_mc_bayes_error_test_split']:.4f}")
    print(f"  Bayes-optimal vs sampled labels    : {exact['bayes_optimal_vs_sampled_labels']:.4f}")
    print(f"  --- learned classifiers, test split (upper bounds) ---")
    print(f"  majority baseline ({bounds['majority_class']}) error : {bounds['majority_baseline_error']:.4f}")
    print(f"  HistGradientBoosting test error    : {bounds['hist_gradient_boosting_test_error']:.4f}")
    print(f"  kNN(15) test error                 : {bounds['knn15_test_error']:.4f}")
    print(f"  Cover-Hart 1-NN upper bracket      : {brackets['asymptotic_1nn_error_upper']:.4f}")
    print(f"\nWrote {args.out} and 4 figures to {figs}/")


if __name__ == "__main__":
    main()
