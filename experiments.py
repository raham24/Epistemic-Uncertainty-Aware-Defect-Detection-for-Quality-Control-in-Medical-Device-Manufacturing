from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlp
from generator import load_spec, seed_everything
from mlp import LOSSES, evaluate, train


def run_one(spec: dict, df: pd.DataFrame, cfg: dict, device: str) -> dict:
    """Algorithm — Train+evaluate one config, capturing any failure.

    Input: spec, data, a config dict, device.
    Return: {status, metrics} where status is ok / numerical_error / error.
    """

    # pin the seed, then train+eval inside a guard so one bad run can't kill the sweep
    seed_everything(cfg["seed"])
    try:
        model, enc = train(spec, df, device=device, epochs=cfg["epochs"],
                           class_weight_mode=cfg.get("class_weight", "none"),
                           loss_cls=LOSSES[cfg["loss"]], seed=cfg["seed"],
                           o=cfg.get("o", 2.0))
        metrics = evaluate(model, enc, df, spec, device=device)
        return {"status": "ok", "metrics": metrics}
    except FloatingPointError as e:
        # the finite-loss guard in train() raises this on a NaN/Inf loss
        return {"status": "numerical_error", "error": str(e)}
    except Exception as e:  # noqa: BLE001 - we want any failure recorded, not raised
        return {"status": "error", "error": repr(e)}


def check_guard_fires(spec: dict, df: pd.DataFrame, device: str) -> str:
    """Algorithm — Prove the finite-loss guard works by forcing a NaN loss.

    Input: spec, data, device.
    Return: the status run_one reports (want 'numerical_error').
    """

    # temporarily make the loss return NaN, run one short config, then restore
    original = mlp.MultiHeadLoss.forward
    import torch

    mlp.MultiHeadLoss.forward = lambda self, *a, **k: torch.tensor(float("nan"))
    try:
        res = run_one(spec, df,
                      {"name": "guard", "loss": "multihead", "seed": 0, "epochs": 1},
                      device)
    finally:
        mlp.MultiHeadLoss.forward = original
    return res["status"]


def _defect(metrics: dict) -> dict:
    """Pull the defect-head block (or zeros if the run failed)."""
    if not metrics or "defect_head" not in metrics:
        return {"accuracy": float("nan"), "weighted_f1": float("nan")}
    return metrics["defect_head"]


def _abst_rate(metrics: dict, h: str = "0.5") -> float | None:
    """Abstention rate at threshold h, or None if not an abstention run."""
    ab = (metrics or {}).get("abstention")
    return ab["by_threshold"][h]["abstention_rate"] if ab else None


def plot_loss_comparison(rows: list[dict], bayes: float, out: Path) -> None:
    """Grouped bars: defect accuracy + weighted-F1 for each loss vs the ceiling."""

    # one group per metric, one bar per loss, plus the Bayes ceiling line
    labels = [r["name"] for r in rows]
    accs = [_defect(r["metrics"])["accuracy"] for r in rows]
    wf1s = [_defect(r["metrics"])["weighted_f1"] for r in rows]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, accs, width=w, label="accuracy", color="#2f6db5")
    ax.bar(x + w / 2, wf1s, width=w, label="weighted-F1", color="#6aa84f")
    ax.axhline(bayes, color="#c1432e", linestyle="--", linewidth=1.2,
               label=f"Bayes ceiling {bayes:.3f}")
    for xi, a, f in zip(x, accs, wf1s):
        ax.text(xi - w / 2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, f + 0.002, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Defect head: multihead vs abstention loss")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_abstention_vs_o(sweep: list[dict], out: Path) -> None:
    """Abstention rate at h=0.5 vs the payoff o (should fall as o rises)."""

    # keep only the runs that finished, in increasing o order
    pts = [(c["o"], _abst_rate(c["metrics"])) for c in sweep
           if c["status"] == "ok" and _abst_rate(c["metrics"]) is not None]
    pts.sort()
    if not pts:
        return
    os_, rates = zip(*pts)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(os_, rates, "o-", color="#2f6db5", linewidth=2)
    for o, r in pts:
        ax.text(o, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("payoff o")
    ax.set_ylabel("abstention rate @ h=0.5")
    ax.set_title("Larger payoff o -> the model abstains less")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    # command-line options
    ap = argparse.ArgumentParser(description="Two-loss experiment matrix")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--data", default="data/smt_synthetic.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/experiments.json")
    ap.add_argument("--figs", default="figs")
    args = ap.parse_args()

    # load the spec and data once, then reuse them for every config
    spec = load_spec(args.spec)
    df = pd.read_csv(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)

    # --- core runs: the two losses head to head, plus the abstain edge cases ---
    core = [
        {"name": "A multihead", "loss": "multihead", "seed": 0, "epochs": 40},
        {"name": "B abstention", "loss": "abstention", "o": 2.0, "seed": 0, "epochs": 40},
        {"name": "E abstain+cw", "loss": "abstention", "o": 2.0, "seed": 0,
         "epochs": 20, "class_weight": "inverse"},
        {"name": "H abstain o=0.5", "loss": "abstention", "o": 0.5, "seed": 0,
         "epochs": 5},
    ]
    for c in core:
        print(f"[run] {c['name']} ...")
        c.update(run_one(spec, df, c, args.device))

    # --- o-sweep: abstention rate should fall as the payoff o rises ---
    sweep = []
    for o in (1.0, 2.0, 4.0, 8.0):
        c = {"name": f"C o={o}", "loss": "abstention", "o": o, "seed": 0, "epochs": 20}
        print(f"[run] {c['name']} ...")
        c.update(run_one(spec, df, c, args.device))
        sweep.append(c)

    # --- determinism: same config twice must give identical defect metrics ---
    det = {}
    for loss in ("multihead", "abstention"):
        cfg = {"name": f"det {loss}", "loss": loss, "o": 2.0, "seed": 0, "epochs": 10}
        r1 = run_one(spec, df, cfg, args.device)
        r2 = run_one(spec, df, cfg, args.device)
        det[loss] = (r1["status"] == "ok" and r2["status"] == "ok"
                     and _defect(r1["metrics"]) == _defect(r2["metrics"]))
        print(f"[det] {loss}: {'identical' if det[loss] else 'DIFFERS'}")

    # --- the finite-loss guard must fire on a NaN ---
    guard_status = check_guard_fires(spec, df, args.device)
    print(f"[guard] NaN -> status '{guard_status}' "
          f"({'OK' if guard_status == 'numerical_error' else 'FAILED'})")

    # --- figures ---
    bayes = next((c["metrics"]["bayes_optimal_accuracy"] for c in core
                  if c["status"] == "ok"), float("nan"))
    plot_loss_comparison([core[0], core[1]], bayes, figs / "loss_comparison.png")
    plot_abstention_vs_o(sweep, figs / "abstention_vs_o.png")

    # --- write everything + print a comparison table ---
    # collapse each run to the few fields worth keeping
    summary = {
        "core": [{k: c[k] for k in ("name", "loss", "status")} |
                 {"defect": _defect(c.get("metrics", {})),
                  "abstention_rate_0.5": _abst_rate(c.get("metrics", {}))}
                 for c in core],
        "o_sweep": [{"o": c["o"], "status": c["status"],
                     "abstention_rate_0.5": _abst_rate(c.get("metrics", {}))}
                    for c in sweep],
        "determinism": det,
        "guard_fires": guard_status == "numerical_error",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print("\n=== comparison (defect head, test split) ===")
    print(f"  {'config':18s} {'status':16s} {'acc':>8s} {'wF1':>8s} {'abst@0.5':>9s}")
    for c in core + sweep:
        d = _defect(c.get("metrics", {}))
        ar = _abst_rate(c.get("metrics", {}))
        ar_s = f"{ar:.3f}" if ar is not None else "-"
        print(f"  {c['name']:18s} {c['status']:16s} {d['accuracy']:8.4f} "
              f"{d['weighted_f1']:8.4f} {ar_s:>9s}")
    print(f"\n  Bayes ceiling: {bayes:.4f}")
    print(f"  determinism: {det}")
    print(f"  guard fires on NaN: {summary['guard_fires']}")
    print(f"\nWrote {args.out} and 2 figures to {figs}/")


if __name__ == "__main__":
    main()
