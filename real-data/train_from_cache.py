"""Train ONE learned-abstention MLP off the cached MAUDE dataset -- the training
step of the sweep (mirrors mlp.py in the synthetic pipeline).

Reads the cache written by prep_dataset.py (fixed dataset + split + features), trains
the single-head abstention MLP with a given payoff --o (and capacity/dropout/lr),
then writes a metrics JSON with BOTH selective-classification curves so the notebook
can compare the learned reject against the confidence-threshold baseline:

  forced               argmax over the 2 real classes on ALL test rows (full coverage)
  learned_reject       accept when the learned abstain prob r < h; h chosen on val to
                       hit --target-coverage
  confidence_baseline  the professor's rule (max class prob >= threshold), threshold
                       chosen on val to hit the same coverage -- head-to-head control
  gain_vs_conf         selective_acc(learned) - selective_acc(confidence)
  reject_curve / confidence_curve   full swept operating points (for risk-coverage plots)

  python real-data/train_from_cache.py --cache real-data/data/maude --o 2.0 --seed 0 \
      --out real-data/results/cluster/0001_maude_o2_s0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maude_product_problem_abstention_mlp as pipe  # noqa: E402

MODEL_VERSION = "maude-abstention-mlp-1.0"


def _curve_rows(rows: list[dict]) -> list[dict]:
    """Keep only the JSON-serializable numeric fields from a sweep."""
    return [{k: (None if v is None else float(v)) for k, v in r.items()} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train abstention MLP off the MAUDE cache")
    ap.add_argument("--cache", type=str, default="real-data/data/maude")
    ap.add_argument("--o", type=float, default=pipe.DEFAULT_O, help="abstention payoff")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=str, default=",".join(map(str, pipe.HIDDEN)))
    ap.add_argument("--dropout", type=float, default=pipe.DROPOUT)
    ap.add_argument("--lr", type=float, default=pipe.LR)
    ap.add_argument("--weight-decay", type=float, default=pipe.WEIGHT_DECAY)
    ap.add_argument("--batch", type=int, default=pipe.BATCH)
    ap.add_argument("--epochs", type=int, default=pipe.EPOCHS)
    ap.add_argument("--patience", type=int, default=pipe.PATIENCE)
    ap.add_argument("--target-coverage", type=float, default=0.65)
    ap.add_argument("--sweep-step", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="real-data/results/cluster/run.json")
    ap.add_argument("--model-out", type=str, default=None,
                    help="optional .pt checkpoint path (skipped if unset)")
    args = ap.parse_args()

    pipe.set_seed(args.seed)
    cache = Path(args.cache)
    z = np.load(cache / "features.npz")
    X_train, X_val, X_test = z["X_train"], z["X_val"], z["X_test"]
    y_train, y_val, y_test = z["y_train"], z["y_val"], z["y_test"]

    hidden = tuple(int(w) for w in args.hidden.split(","))
    print(f"train_from_cache.py  o={args.o} seed={args.seed} hidden={hidden} "
          f"dropout={args.dropout} lr={args.lr}  (device={args.device})")

    model = pipe.train_abstention_mlp(
        X_train, y_train, X_val, y_val,
        o=args.o, seed=args.seed, device=args.device,
        hidden=hidden, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, batch=args.batch,
        epochs=args.epochs, patience=args.patience,
    )

    proba_val, abstain_val = pipe.model_outputs(model, X_val, args.device)
    proba_test, abstain_test = pipe.model_outputs(model, X_test, args.device)

    # forced (full-coverage) metrics
    yhat = (proba_test[:, 1] >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_test, proba_test[:, 1]))
    except Exception:
        auc = None
    forced = {
        "accuracy": float(accuracy_score(y_test, yhat)),
        "f1": float(f1_score(y_test, yhat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, yhat)),
        "roc_auc": auc,
    }

    # learned reject vs confidence baseline, both at the matched target coverage
    h = pipe.choose_reject_by_val(abstain_val, args.target_coverage)
    t = pipe.choose_threshold_by_val(proba_val, args.target_coverage)
    learned = pipe.selective_metrics(y_test, proba_test, pipe.reject_keep(abstain_test, h))
    baseline = pipe.selective_metrics(y_test, proba_test, pipe.confidence_keep(proba_test, t))
    gain = None
    if learned["selective_accuracy"] is not None and baseline["selective_accuracy"] is not None:
        gain = learned["selective_accuracy"] - baseline["selective_accuracy"]

    metrics = {
        "model_version": MODEL_VERSION,
        "o": args.o, "seed": args.seed, "hidden": args.hidden,
        "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
        "target_coverage": args.target_coverage,
        "n_test": int(len(y_test)),
        "mean_abstain_prob": float(abstain_test.mean()),
        "forced": forced,
        "learned_reject": {"abstain_threshold_h": float(h), **learned},
        "confidence_baseline": {"threshold": float(t), **baseline},
        "gain_vs_conf": gain,
        "reject_curve": _curve_rows(
            pipe.reject_sweep(y_test, proba_test, abstain_test, 0.01, 0.99, args.sweep_step)),
        "confidence_curve": _curve_rows(
            pipe.confidence_sweep(y_test, proba_test, 0.50, 0.99, args.sweep_step)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.model_out:
        mp = Path(args.model_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_features": X_train.shape[1], "n_classes": 2,
                               "hidden": hidden, "dropout": args.dropout, "o": args.o}}, mp)

    print(f"  forced acc {forced['accuracy']:.4f}  "
          f"learned sel {learned['selective_accuracy']}  "
          f"conf sel {baseline['selective_accuracy']}  gain {gain}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
