from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from generator import load_spec, seed_everything
from mlp import LOSSES, evaluate, train

# trunk architectures to search over (string keys -> tuples; categorical needs scalars)
ARCHS = ["128", "256", "128-128", "256-256", "512-256", "256-256-128", "512-256-128"]
BASE_SEED = 0          # per-trial model seed = BASE_SEED + trial.number
SAMPLER_SEED = 1337    # fixes the trial parameter sequence


def _hidden(arch: str) -> tuple[int, ...]:
    """Parse an arch key like '512-256' into a width tuple."""
    return tuple(int(x) for x in arch.split("-"))


def make_objective(spec: dict, df: pd.DataFrame, loss_cls, epochs: int):
    """Algorithm — Build the Optuna objective for one loss.

    Input: spec, data, the loss class, per-trial epoch budget.
    Return: an objective(trial) -> validation weighted-F1 (to maximize).
    """

    abstain = getattr(loss_cls, "ABSTAIN", False)

    def objective(trial: optuna.Trial) -> float:
        # sample the hyperparameters
        hidden = _hidden(trial.suggest_categorical("arch", ARCHS))
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical("batch", [64, 128, 256, 512])
        cw = trial.suggest_categorical("class_weight_mode", ["none", "sqrt", "inverse"])
        # the payoff o only exists for an abstention loss
        o = trial.suggest_float("o", 0.5, 8.0, log=True) if abstain else 2.0

        # pin THIS trial's randomness, then train (train() does not self-seed init)
        trial_seed = BASE_SEED + trial.number
        seed_everything(trial_seed)
        model, enc = train(spec, df, device="cpu", epochs=epochs,
                           class_weight_mode=cw, loss_cls=loss_cls, hidden=hidden,
                           dropout=dropout, seed=trial_seed, lr=lr, batch=batch, o=o)

        # SCORE ON VALIDATION ONLY -- test is held out for the final report
        res = evaluate(model, enc, df, spec, device="cpu", split="val")
        trial.set_user_attr("model_seed", trial_seed)
        trial.set_user_attr("val_accuracy", res["defect_head"]["accuracy"])
        return res["defect_head"]["weighted_f1"]

    return objective


def save_optuna_figs(study: optuna.Study, loss: str, figs: Path) -> None:
    """Save the standard Optuna diagnostic figures (matplotlib backend)."""

    from optuna.visualization.matplotlib import (plot_optimization_history,
                                                 plot_parallel_coordinate,
                                                 plot_param_importances,
                                                 plot_slice)

    # each returns a matplotlib Axes; grab its figure and save
    for fn, name in [(plot_optimization_history, "history"),
                     (plot_param_importances, "importances"),
                     (plot_slice, "slice"),
                     (plot_parallel_coordinate, "parallel")]:
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.tight_layout()
            fig.savefig(figs / f"optuna_{name}_{loss}.png", dpi=140)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001 - a missing fig should not kill the run
            print(f"  (skipped optuna_{name}: {e})")


def main() -> None:
    # command-line options
    ap = argparse.ArgumentParser(description="Optuna tuning for the multi-head MLP")
    ap.add_argument("--loss", default="abstention", choices=list(LOSSES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10, help="per-trial epoch budget")
    ap.add_argument("--final-epochs", type=int, default=40, help="best-config retrain")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--figs", default="figs")
    args = ap.parse_args()

    # load the spec + data and pick the loss class to tune
    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    loss_cls = LOSSES[args.loss]

    # reproducible study: fixed sampler seed -> same trial sequence every run
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SAMPLER_SEED, multivariate=True, group=True),
        study_name=f"smt_{args.loss}")
    print(f"tuning {args.loss}: {args.trials} trials x {args.epochs} epochs (val weighted-F1)")
    study.optimize(make_objective(spec, df, loss_cls, args.epochs),
                   n_trials=args.trials, show_progress_bar=False)

    # --- retrain the best config and report TEST exactly once ---
    best = study.best_trial
    bp = best.params
    best_seed = best.user_attrs["model_seed"]
    o = bp.get("o", 2.0)
    print(f"\nbest val weighted-F1 {best.value:.4f}  params {bp}")
    seed_everything(best_seed)
    model, enc = train(spec, df, device="cpu", epochs=args.final_epochs,
                       class_weight_mode=bp["class_weight_mode"], loss_cls=loss_cls,
                       hidden=_hidden(bp["arch"]), dropout=bp["dropout"],
                       seed=best_seed, lr=bp["lr"], batch=bp["batch"], o=o)
    test_metrics = evaluate(model, enc, df, spec, device="cpu", split="test")

    # --- persist everything (provenance: device, seeds, the single test report) ---
    out = {
        "loss": args.loss,
        "device": "cpu",
        "n_trials": args.trials,
        "per_trial_epochs": args.epochs,
        "sampler_seed": SAMPLER_SEED,
        "best_params": bp,
        "best_model_seed": best_seed,
        "best_val_weighted_f1": best.value,
        "test_metrics": test_metrics,
    }
    res_path = Path(f"results/best_params_{args.loss}.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(res_path, "w"), indent=2)
    save_optuna_figs(study, args.loss, figs)

    # --- console summary ---
    td = test_metrics["defect_head"]
    print(f"\n=== best config, TEST split ({args.loss}) ===")
    print(f"  defect acc {td['accuracy']:.4f}  wF1 {td['weighted_f1']:.4f}  "
          f"macroF1 {td['macro_f1']:.4f}")
    print(f"  Bayes ceiling {test_metrics['bayes_optimal_accuracy']:.4f}")
    print(f"  best params: {bp}")
    print(f"\nWrote {res_path} and optuna figures to {figs}/")


if __name__ == "__main__":
    main()
