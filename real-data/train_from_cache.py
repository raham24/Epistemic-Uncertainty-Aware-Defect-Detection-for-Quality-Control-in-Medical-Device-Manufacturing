"""Train ONE single-head MLP off the cached MAUDE dataset -- the training step of
the sweep (mirrors mlp.py in the synthetic pipeline).

Reads the cache written by prep_dataset.py (fixed dataset + split + features), infers
the number of classes from the labels (binary product-problem OR the 4-class MAUDE
severity task: Malfunction / Basic injury / Serious injury / Death), trains the
single-head MLP with the chosen loss, then writes a metrics JSON with BOTH
selective-classification curves so the notebook can compare the learned reject
against the confidence-threshold baseline:

  forced               argmax over the real classes on ALL test rows (full coverage)
  learned_reject       accept when the learned abstain prob r < h (abstention loss
                       only); h chosen on val to hit --target-coverage
  confidence_baseline  max class prob >= threshold, threshold chosen on val to hit
                       the same coverage -- the head-to-head control
  gain_vs_conf         selective_acc(learned) - selective_acc(confidence)
  reject_curve / confidence_curve   full swept operating points (risk-coverage plots)

Two losses (like mlp.py): --loss abstention (learned reject, --o payoff) or
--loss ce (plain class-weighted cross-entropy, the baseline). --class-weight handles
the MAUDE class skew (Malfunction dominates, Death rare).

  python real-data/train_from_cache.py --cache real-data/data/maude --o 2.0 --seed 0 \
      --class-weight sqrt --out real-data/results/cluster/0001_maude_o2_s0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maude_product_problem_abstention_mlp as pipe  # noqa: E402

MODEL_VERSION = "maude-mlp-2.0"    # 2.0 = multi-class (was binary in 1.0)


def _curve_rows(rows: list[dict]) -> list[dict]:
    """Keep only the JSON-serializable numeric fields from a sweep."""
    return [{k: (None if v is None else float(v)) for k, v in r.items()} for r in rows]


def _forced_metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """Full-coverage multi-class metrics: argmax over the real classes on ALL rows.
    F1 is reported both macro (rare classes count equally) and weighted; ROC-AUC is
    one-vs-rest macro (guarded -- needs every class present in y_true)."""
    yhat = proba.argmax(axis=1)
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")) \
            if n_classes > 2 else float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        auc = None
    per_class = f1_score(y_true, yhat, average=None, labels=list(range(n_classes)),
                         zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, yhat)),
        "macro_f1": float(f1_score(y_true, yhat, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, yhat, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, yhat, average="weighted", zero_division=0)),  # notebook key
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "roc_auc": auc,
        "per_class_f1": [float(v) for v in per_class],
        "confusion_matrix": confusion_matrix(y_true, yhat,
                                             labels=list(range(n_classes))).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train single-head MLP off the MAUDE cache")
    ap.add_argument("--cache", type=str, default="real-data/data/maude")
    ap.add_argument("--loss", type=str, default="abstention",
                    choices=["abstention", "ce"],
                    help="abstention = learned reject (-log(o*p_y+r)); ce = plain CE")
    ap.add_argument("--o", type=float, default=pipe.DEFAULT_O, help="abstention payoff")
    ap.add_argument("--class-weight", type=str, default="none",
                    choices=["none", "sqrt", "inverse"],
                    help="class weighting for the skewed MAUDE classes")
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
    n_classes = int(max(int(y_train.max()), int(y_val.max()), int(y_test.max()))) + 1

    hidden = tuple(int(w) for w in args.hidden.split(","))
    print(f"train_from_cache.py  loss={args.loss} o={args.o} cw={args.class_weight} "
          f"n_classes={n_classes} seed={args.seed} hidden={hidden} "
          f"dropout={args.dropout} lr={args.lr}  (device={args.device})")

    model = pipe.train_abstention_mlp(
        X_train, y_train, X_val, y_val,
        o=args.o, seed=args.seed, device=args.device,
        hidden=hidden, dropout=args.dropout, lr=args.lr,
        weight_decay=args.weight_decay, batch=args.batch,
        epochs=args.epochs, patience=args.patience,
        n_classes=n_classes, loss=args.loss, class_weight_mode=args.class_weight,
    )

    proba_val, abstain_val = pipe.model_outputs(model, X_val, args.device)
    proba_test, abstain_test = pipe.model_outputs(model, X_test, args.device)

    forced = _forced_metrics(y_test, proba_test, n_classes)

    # confidence baseline (any loss): max class prob >= threshold, chosen on val
    t = pipe.choose_threshold_by_val(proba_val, args.target_coverage)
    baseline = pipe.selective_metrics(y_test, proba_test, pipe.confidence_keep(proba_test, t))
    # sweep from 0.20 (below 1/n_classes for up to 5 classes) so the Chow o-threshold
    # 1/o -- which can be as low as 0.25 at o=4 -- is covered by the curve.
    confidence_curve = _curve_rows(
        pipe.confidence_sweep(y_test, proba_test, 0.20, 0.99, args.sweep_step))

    # learned reject: only the abstention loss has an abstain column
    if args.loss == "abstention":
        h = pipe.choose_reject_by_val(abstain_val, args.target_coverage)
        learned = pipe.selective_metrics(y_test, proba_test, pipe.reject_keep(abstain_test, h))
        learned_block = {"abstain_threshold_h": float(h), **learned}
        reject_curve = _curve_rows(
            pipe.reject_sweep(y_test, proba_test, abstain_test, 0.01, 0.99, args.sweep_step))
        gain = None
        if learned["selective_accuracy"] is not None and baseline["selective_accuracy"] is not None:
            gain = learned["selective_accuracy"] - baseline["selective_accuracy"]
    else:
        learned_block, reject_curve, gain = None, [], None

    metrics = {
        "model_version": MODEL_VERSION,
        "loss": args.loss, "o": args.o, "class_weight": args.class_weight,
        "seed": args.seed, "hidden": args.hidden,
        "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
        "n_classes": n_classes,
        "class_counts_test": np.bincount(y_test, minlength=n_classes).tolist(),
        "target_coverage": args.target_coverage,
        "n_test": int(len(y_test)),
        "mean_abstain_prob": float(abstain_test.mean()),
        "forced": forced,
        "learned_reject": learned_block,
        "confidence_baseline": {"threshold": float(t), **baseline},
        "gain_vs_conf": gain,
        "reject_curve": reject_curve,
        "confidence_curve": confidence_curve,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.model_out:
        mp = Path(args.model_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_features": X_train.shape[1], "n_classes": n_classes,
                               "hidden": hidden, "dropout": args.dropout, "o": args.o,
                               "loss": args.loss, "abstain": args.loss == "abstention"}}, mp)

    learned_sel = learned_block["selective_accuracy"] if learned_block else None
    print(f"  forced acc {forced['accuracy']:.4f}  macroF1 {forced['macro_f1']:.4f}  "
          f"learned sel {learned_sel}  conf sel {baseline['selective_accuracy']}  gain {gain}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
