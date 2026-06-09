from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from generator import defect_names, load_spec, param_ids, seed_everything
from mlp import LOSSES, evaluate, train

DEFECT_COLORS = {0: "#8a8d91", 1: "#2f6db5", 2: "#c1432e"}
SHORT = {"no_defect": "none", "open_circuit": "open", "solder_bridging": "bridge"}


# --------------------------------------------------------------------------- #
# Shared: train both models once, then pack test-split predictions
# --------------------------------------------------------------------------- #


def _pack(model, enc: dict, df: pd.DataFrame, spec: dict, split: str = "test") -> dict:
    """Algorithm — Predict on one split and bundle what the plots need.

    Input: trained model, enc, df, spec, split name.
    Return: predictions, true labels, the known posterior, and the abstain prob.
    """

    # pull RAW features for the split and standardize with stored train stats
    ids, names = param_ids(spec), defect_names(spec)
    te = np.where(df["split"].to_numpy() == split)[0]
    Xraw = df.iloc[te][ids].to_numpy(np.float32)
    X = ((Xraw - enc["mu"]) / enc["sd"]).astype(np.float32)

    # per-head probabilities + argmax (the evidence payload)
    pred = model.predict(torch.from_numpy(X))
    y = enc["y_def"][te]
    post = df.iloc[te][[f"p_{n}" for n in names]].to_numpy()
    pack = {"te": te, "y": y, "post": post, "names": names,
            "d_pred": pred["defect_argmax"].cpu().numpy(),
            "defect_prob": pred["defect_prob"].cpu().numpy(),
            "risk_prob": pred["risk_prob"].cpu().numpy(),
            "y_risk": enc["y_risk"][te]}

    # abstain prob exists only for the abstention model
    if "abstain_prob" in pred:
        pack["r"] = pred["abstain_prob"].cpu().numpy()
    return pack


def _sel_curve(correct: np.ndarray, conf: np.ndarray):
    """Selective accuracy vs coverage: accept highest-confidence boards first."""

    # sort by confidence descending, then take a running accuracy of the top-k
    order = np.argsort(-conf)
    cov = np.arange(1, len(correct) + 1) / len(correct)
    sel = correct[order].cumsum() / np.arange(1, len(correct) + 1)
    return cov, sel


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_risk_coverage(mh: dict, ab: dict, bayes: float, out: Path) -> None:
    """Selective accuracy vs coverage: learned reject order vs the oracle order."""

    fig, ax = plt.subplots(figsize=(8, 5))

    # abstention model, rejecting by its LEARNED abstain prob (low r = keep)
    cor_ab = (ab["d_pred"] == ab["y"]).astype(float)
    cov, sel = _sel_curve(cor_ab, -ab["r"])
    ax.plot(cov, sel, color="#2f6db5", lw=2, label="abstention: learned reject (r)")

    # same predictions, but rejecting by the KNOWN posterior confidence (oracle order)
    cov_o, sel_o = _sel_curve(cor_ab, ab["post"].max(1))
    ax.plot(cov_o, sel_o, color="#6aa84f", lw=2, ls="--",
            label="abstention: oracle reject order")

    # multihead model, rejecting by its own max class prob (no abstain head)
    cor_mh = (mh["d_pred"] == mh["y"]).astype(float)
    cov_m, sel_m = _sel_curve(cor_mh, mh["defect_prob"].max(1))
    ax.plot(cov_m, sel_m, color="#e08214", lw=2, label="multihead: max-prob reject")

    # full-coverage accuracy and the Bayes ceiling for reference
    ax.axhline(cor_ab.mean(), color="#999", ls=":", lw=1,
               label=f"full coverage {cor_ab.mean():.3f}")
    ax.axhline(bayes, color="#c1432e", ls="--", lw=1, label=f"Bayes ceiling {bayes:.3f}")
    ax.set_xlabel("coverage (fraction predicted)")
    ax.set_ylabel("selective accuracy")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.94, 1.005)
    ax.set_title("Risk-coverage: abstain on the hardest boards -> accuracy rises")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_abstention_vs_difficulty(ab: dict, out: Path) -> None:
    """Mean learned abstain prob vs the KNOWN posterior difficulty, per bin."""

    # difficulty from the known posterior: max prob, and the bin's Bayes error
    maxp = ab["post"].max(1)
    r = ab["r"]
    edges = np.array([0.40, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0001])
    idx = np.digitize(maxp, edges) - 1
    centers, mean_r, bayes_err, counts = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        centers.append((edges[b] + min(edges[b + 1], 1.0)) / 2)
        mean_r.append(r[m].mean())
        bayes_err.append((1 - maxp[m]).mean())   # irreducible error of the bin
        counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, mean_r, "o-", color="#2f6db5", lw=2, label="mean abstain prob")
    for c, v, n in zip(centers, mean_r, counts):
        ax.text(c, v + 0.01, f"n={n}", ha="center", fontsize=7, color="#555")
    ax2 = ax.twinx()
    ax2.plot(centers, bayes_err, "s--", color="#c1432e", lw=1.5,
             label="bin Bayes error (1-max p)")
    ax.set_xlabel("max posterior prob  max_y p(y|x)   (right = easy, left = hard)")
    ax.set_ylabel("mean abstain prob", color="#2f6db5")
    ax2.set_ylabel("bin Bayes error", color="#c1432e")
    ax.set_title("Abstentions concentrate on the irreducibly-hard boards")
    ax.invert_xaxis()                                   # hardest on the left
    ax.grid(alpha=0.25)
    # one combined legend from both y-axes, tucked into the empty lower-right
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_confusion(mh: dict, ab: dict, out: Path) -> None:
    """Row-normalized defect confusion: multihead / abstention / Bayes oracle."""

    names = [SHORT[n] for n in mh["names"]]
    panels = [("multihead", mh["d_pred"], mh["y"]),
              ("abstention (full coverage)", ab["d_pred"], ab["y"]),
              ("Bayes oracle", mh["post"].argmax(1), mh["y"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (title, pred, y) in zip(axes, panels):
        cm = confusion_matrix(y, pred, labels=[0, 1, 2]).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)             # row-normalize (recall)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}\n{int(cm[i, j])}", ha="center",
                        va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.5 else "#222")
        ax.set_xticks(range(3)); ax.set_xticklabels(names)
        ax.set_yticks(range(3)); ax.set_yticklabels(names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title)
    fig.suptitle("Defect-head confusion (row-normalized) -- residual error is minority-class",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_training_curves(enc_mh: dict, enc_ab: dict, out: Path) -> None:
    """Per-epoch validation loss for both losses (different scales -> twin axes)."""

    fig, ax = plt.subplots(figsize=(8, 5))
    h_mh, h_ab = enc_mh["val_loss_history"], enc_ab["val_loss_history"]
    ax.plot(range(len(h_mh)), h_mh, "o-", color="#2f6db5", label="multihead (left)")
    ax.axvline(enc_mh["best_epoch"], color="#2f6db5", ls=":", lw=1)
    ax2 = ax.twinx()
    ax2.plot(range(len(h_ab)), h_ab, "s-", color="#e08214", label="abstention (right)")
    ax2.axvline(enc_ab["best_epoch"], color="#e08214", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("multihead val loss", color="#2f6db5")
    ax2.set_ylabel("abstention val loss", color="#e08214")
    ax.set_title("Validation loss per epoch (dotted = early-stop best epoch)")
    ax.grid(alpha=0.25)
    # combined legend from both axes, placed clear of the title
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_calibration(mh: dict, ab: dict, out: Path) -> None:
    """Reliability diagram: predicted confidence vs realized accuracy."""

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([1 / 3, 1], [1 / 3, 1], color="#999", ls="--", lw=1, label="perfect")
    for pack, color, label in [(mh, "#2f6db5", "multihead"),
                               (ab, "#e08214", "abstention")]:
        # abstention real-class probs do not sum to 1 -> renormalize for confidence
        p = pack["defect_prob"]
        p = p / p.sum(1, keepdims=True)
        conf = p.max(1)
        correct = (pack["d_pred"] == pack["y"]).astype(float)
        bins = np.linspace(1 / 3, 1.0, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", color=color, lw=2, label=label)
    ax.set_xlabel("predicted confidence (max class prob)")
    ax.set_ylabel("realized accuracy")
    ax.set_title("Calibration: confidence vs accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_risk_reliability(mh: dict, spec: dict, out: Path) -> None:
    """Predicted parameter risk vs the graded ground-truth target, per parameter."""

    ids = param_ids(spec)
    pred, true = mh["risk_prob"], mh["y_risk"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, (ax, pid) in enumerate(zip(axes.ravel(), ids)):
        p, t = pred[:, j], true[:, j]
        # bin predictions, plot mean true target per bin against the diagonal
        bins = np.linspace(0, 1, 16)
        idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.any():
                xs.append(p[m].mean()); ys.append(t[m].mean())
        ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
        ax.plot(xs, ys, "o-", color="#6aa84f", lw=1.6)
        ax.set_title(f"{pid}  (MAE {np.abs(p - t).mean():.3f})", fontsize=10)
        ax.set_xlabel("predicted risk"); ax.set_ylabel("true graded risk")
    fig.suptitle("Risk-head reliability: predicted vs graded ground-truth (Eq. 8)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables (markdown)
# --------------------------------------------------------------------------- #


def _per_class(pack: dict) -> dict:
    """Per-class precision/recall/F1 from a prediction pack."""
    p, r, f, _ = precision_recall_fscore_support(
        pack["y"], pack["d_pred"], labels=[0, 1, 2], zero_division=0)
    return {n: (p[i], r[i], f[i]) for i, n in enumerate(pack["names"])}


def write_tables(mh: dict, ab: dict, m_mh: dict, m_ab: dict, out: Path) -> None:
    """Write the per-class, selective, and scorecard tables as markdown."""

    lines = ["# Extra figures and tables\n",
             "Generated by make_figs.py (multihead + abstention, seed 0, test split).\n"]

    # 1. per-class precision/recall/F1 for both losses + the Bayes oracle
    oracle = {"y": mh["y"], "d_pred": mh["post"].argmax(1), "names": mh["names"]}
    pc = {"multihead": _per_class(mh), "abstention": _per_class(ab),
          "bayes oracle": _per_class(oracle)}
    lines.append("## Per-class precision / recall / F1 (defect head)\n")
    lines.append("| class | model | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for n in mh["names"]:
        for model in ("multihead", "abstention", "bayes oracle"):
            p, r, f = pc[model][n]
            lines.append(f"| {n} | {model} | {p:.3f} | {r:.3f} | {f:.3f} |")
    lines.append("")

    # 2. selective operating points for the abstention model
    full = m_ab["defect_head"]["accuracy"]
    lines.append("## Selective operating points (abstention model)\n")
    lines.append(f"Full-coverage accuracy = {full:.4f}. "
                 "error_reduction = selective_accuracy - full_accuracy.\n")
    lines.append("| threshold h | coverage | abstention rate | selective acc | error reduction |")
    lines.append("|---|---|---|---|---|")
    for h, v in m_ab["abstention"]["by_threshold"].items():
        sel = v["selective_accuracy"]
        sel_s = f"{sel:.4f}" if sel is not None else "-"
        er = f"{sel - full:+.4f}" if sel is not None else "-"
        lines.append(f"| {h} | {v['coverage']:.4f} | {v['abstention_rate']:.4f} "
                     f"| {sel_s} | {er} |")
    lines.append("")

    # 3. end-to-end multi-head scorecard
    lines.append("## End-to-end scorecard (test split)\n")
    lines.append("| metric | multihead | abstention | reference |")
    lines.append("|---|---|---|---|")
    for key, ref in [("accuracy", "Bayes 0.9534 / paper 0.9500"),
                     ("weighted_f1", "paper 0.9536"), ("macro_f1", "-")]:
        lines.append(f"| defect {key} | {m_mh['defect_head'][key]:.4f} "
                     f"| {m_ab['defect_head'][key]:.4f} | {ref} |")
    for stage in m_mh["mechanism_accuracy"]:
        lines.append(f"| mech {stage} | {m_mh['mechanism_accuracy'][stage]:.4f} "
                     f"| {m_ab['mechanism_accuracy'][stage]:.4f} | - |")
    lines.append(f"| risk MAE | {m_mh['risk_mae']:.4f} | {m_ab['risk_mae']:.4f} | - |")
    lines.append(f"| Bayes ceiling | {m_mh['bayes_optimal_accuracy']:.4f} "
                 f"| {m_ab['bayes_optimal_accuracy']:.4f} | - |")
    lines.append("")

    out.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    # load the spec, data, and the output figures dir
    spec = load_spec("domain/smt_paper.yaml")
    df = pd.read_csv("data/smt_synthetic.csv")
    figs = Path("figs")
    figs.mkdir(parents=True, exist_ok=True)

    # train each loss ONCE so every figure reflects the same models
    print("training multihead ...")
    seed_everything(0)
    model_mh, enc_mh = train(spec, df, loss_cls=LOSSES["multihead"], seed=0, epochs=40)
    print("training abstention ...")
    seed_everything(0)
    model_ab, enc_ab = train(spec, df, loss_cls=LOSSES["abstention"], o=2.0,
                             seed=0, epochs=40)

    # test-split metrics + prediction packs
    m_mh = evaluate(model_mh, enc_mh, df, spec)
    m_ab = evaluate(model_ab, enc_ab, df, spec)
    mh = _pack(model_mh, enc_mh, df, spec)
    ab = _pack(model_ab, enc_ab, df, spec)
    bayes = m_mh["bayes_optimal_accuracy"]

    # figures
    plot_risk_coverage(mh, ab, bayes, figs / "risk_coverage.png")
    plot_abstention_vs_difficulty(ab, figs / "abstention_vs_difficulty.png")
    plot_confusion(mh, ab, figs / "confusion_matrices.png")
    plot_training_curves(enc_mh, enc_ab, figs / "training_curves.png")
    plot_calibration(mh, ab, figs / "calibration.png")
    plot_risk_reliability(mh, spec, figs / "risk_reliability.png")

    # tables
    Path("docs").mkdir(exist_ok=True)
    write_tables(mh, ab, m_mh, m_ab, Path("docs/figures_tables.md"))

    print("wrote 6 figures to figs/ and docs/figures_tables.md")


if __name__ == "__main__":
    main()
