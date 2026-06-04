"""Multi-seed robustness study: separate DATA variance from MODEL variance.

Randomness audit — the whole pipeline's entropy is exactly TWO seeds:

  - data_seed  : drives the generator. np.random.SeedSequence(data_seed) spawns
                 one stream per batch for the process draw, plus one stream for
                 label sampling. No global RNG is used, so nothing can overwrite it.
  - model_seed : drives the MLP only — weight init, mini-batch shuffling, and the
                 early-stopping validation split — via random_state (seed_everything
                 also pins global/torch state for good measure).

Everything else is deterministic: the batch-grouped split, the deviations, the
calibration (fixed-point), the Bayes-optimal and majority references.

So we can isolate each source:
  - Sweep A (model variance): fix the data, vary model_seed -> how much does the
    result depend on the MODEL's training randomness?
  - Sweep B (data variance): fix the model_seed, vary data_seed -> how much does
    it depend on WHICH dataset realization we drew?
We report mean ± std of accuracy / weighted-F1 / macro-F1 (and the Bayes floor)
across each sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import bayes_error_exact, defect_head_metrics
from generator import generate, load_spec, seed_everything

METRICS = ["accuracy", "weighted_f1", "macro_f1", "bayes_floor_test"]


def _eval(df, spec: dict, model_seed: int) -> dict:
    """Train the MLP at model_seed and return its metrics + the data's Bayes floor."""

    # pin global/torch state too (sklearn uses random_state; this is extra safety)
    seed_everything(model_seed)

    # the Bayes floor is a property of the data, not the model
    floor = bayes_error_exact(df, spec)["exact_mc_bayes_error_test_split"]

    # MLP metrics on the test split
    m = defect_head_metrics(df, spec, model_seed=model_seed)["mlp_basic"]
    return {"accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"],
            "macro_f1": m["macro_f1"], "bayes_floor_test": floor}


def _agg(runs: list[dict]) -> dict:
    """Mean and std of each metric across a list of runs."""
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in METRICS}


def _plot(model_var: dict, data_var: dict, n: int, out: Path) -> None:
    """Grouped bars with std error bars: model variance vs data variance."""

    metrics = ["accuracy", "weighted_f1", "macro_f1"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, agg, color) in enumerate([
            ("vary model (data fixed)", model_var, "#2f6db5"),
            ("vary data (model fixed)", data_var, "#e08214")]):
        means = [agg[m]["mean"] for m in metrics]
        stds = [agg[m]["std"] for m in metrics]
        ax.bar(x + (i - 0.5) * w, means, width=w, yerr=stds, capsize=5,
               label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Weighted F1", "Macro F1"])
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("score (mean ± std)")
    ax.set_title(f"Seed robustness (n={n} each): where does the variance come from?")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed data/model variance study")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--model_seed", type=int, default=0)
    ap.add_argument("--out", default="results/seed_study.json")
    ap.add_argument("--fig", default="figs/seed_study.png")
    args = ap.parse_args()
    spec = load_spec(args.spec)
    n = args.n_seeds

    # Sweep A — model variance: generate the data ONCE, vary the model seed
    df_fixed = generate(spec, seed=args.data_seed)
    model_runs = [_eval(df_fixed, spec, args.model_seed + s) for s in range(n)]
    model_var = _agg(model_runs)

    # Sweep B — data variance: fix the model seed, regenerate per data seed
    data_runs = [_eval(generate(spec, seed=args.data_seed + s), spec, args.model_seed)
                 for s in range(n)]
    data_var = _agg(data_runs)

    summary = {
        "n_seeds": n,
        "model_variance": {"data_seed": args.data_seed, "metrics": model_var},
        "data_variance": {"model_seed": args.model_seed, "metrics": data_var},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    _plot(model_var, data_var, n, Path(args.fig))

    def show(title, agg):
        print(f"\n=== {title} (n={n}) ===")
        for k in METRICS:
            print(f"  {k:18s} {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")

    show(f"MODEL variance  (data_seed={args.data_seed} fixed, model_seed varies)", model_var)
    show(f"DATA variance   (model_seed={args.model_seed} fixed, data_seed varies)", data_var)
    print(f"\nWrote {args.out} and {args.fig}")


if __name__ == "__main__":
    main()
