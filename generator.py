"""SMT synthetic process/defect generator (paper replication baseline).

This is the improved, spec-driven successor to the original scratch generator. Everything domain-specific is read from domain/smt_paper.yaml; this
module contains only mechanism, not data. Output is deterministic given a seed.

Pipeline per record:
    1. Draw 6 correlated process parameters via a Gaussian copula (correlation
       matrix from the spec, nearest-PD corrected) with Gaussian marginals.
    2. Layer non-stationary drift (linear wear + diurnal sinusoid) on top.
    3. Compute normalized deviations from nominal (|dev| = 1.0 at the spec limit).
    4. Build defect logits from the causal map and convert to a posterior
       p(y | x); a single no-defect offset is calibrated so the marginal class
       balance matches the spec priors (~87.7% no-defect, paper Table II shape).
    5. Sample the defect label, assign per-stage mechanism labels, and compute
       a graded two-sided risk target per parameter ("Eq. 8").
    6. Stamp full provenance (generator_version, seed, batch_id, timestamp).

The exact posterior p(y | x) is emitted as columns, which lets us compute the
Bayes error of the generative process in closed form (see analysis.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# Spec loading
# --------------------------------------------------------------------------- #


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate the YAML domain spec."""
    with open(path, "r") as fh:
        spec = yaml.safe_load(fh)
    required = {"parameters", "defects", "mechanisms", "causal_edges",
                "risk_function", "label_model", "generator"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"domain spec missing keys: {sorted(missing)}")
    return spec


def param_ids(spec: dict[str, Any]) -> list[str]:
    return [p["id"] for p in spec["parameters"]]


def defect_names(spec: dict[str, Any]) -> list[str]:
    return [d["name"] for d in spec["defects"]]


# --------------------------------------------------------------------------- #
# Correlation matrix (Gaussian copula)
# --------------------------------------------------------------------------- #


def _nearest_pd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the nearest PD correlation matrix.

    Eigenvalue clipping + rescaling to unit diagonal. Sufficient for the small,
    nearly-PD matrices produced from a hand-specified correlation block.
    """
    sym = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    pd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(pd))
    pd = pd / np.outer(d, d)
    return (pd + pd.T) / 2.0


def build_correlation_matrix(spec: dict[str, Any]) -> np.ndarray:
    """Assemble a full symmetric PD correlation matrix from the spec block."""
    ids = param_ids(spec)
    idx = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    corr = np.eye(n)
    for a, partners in spec.get("correlations", {}).items():
        for b, rho in partners.items():
            i, j = idx[a], idx[b]
            corr[i, j] = corr[j, i] = float(rho)
    return _nearest_pd(corr)


# --------------------------------------------------------------------------- #
# Risk function (the paper's "Eq. 8"): two-sided graded risk in |deviation|
# --------------------------------------------------------------------------- #


def graded_risk(dev: np.ndarray, rf: dict[str, float]) -> np.ndarray:
    """Graded ground-truth risk — the paper's Eq. (8) with (9)-(10).

        P_j(Δ) = p_L + (p_M - p_L) z^2 + (p_H - p_M)(1 - exp(-kappa * t))   (8)
        z      = min(1, max(0, (|Δ| - Δ1) / (Δ2 - Δ1)))                     (9)
        t      = max(0, |Δ| - Δ2)                                          (10)

    Δ2 is the spec limit (|dev| = 1.0). Risk is ~p_L in the safe region, rises
    quadratically to p_M AT the spec limit, and asymptotes to p_H out of spec.
    Two-sided in |Δ|.
    """
    p_L, p_M, p_H = rf["p_L"], rf["p_M"], rf["p_H"]
    d1, d2, kappa = rf["delta_1"], rf["delta_2"], rf["kappa"]
    u = np.abs(np.asarray(dev, dtype=float))
    z = np.clip((u - d1) / max(d2 - d1, 1e-9), 0.0, 1.0)
    t = np.maximum(0.0, u - d2)
    return p_L + (p_M - p_L) * z ** 2 + (p_H - p_M) * (1.0 - np.exp(-kappa * t))


# --------------------------------------------------------------------------- #
# Process sampling
# --------------------------------------------------------------------------- #


def _sample_process(spec: dict[str, Any], seed: int) -> pd.DataFrame:
    """Draw all process parameters for the whole run, batch by batch.

    Returns one row per record with raw parameter values plus batch/shift/time
    provenance. Determinism: a single SeedSequence(seed) is spawned per batch, so
    output is byte-identical across runs and independent of batch count ordering.
    """
    gen = spec["generator"]
    ids = param_ids(spec)
    nominal = np.array([p["nominal"] for p in spec["parameters"]])
    sigma = np.array([p["sigma"] for p in spec["parameters"]])
    chol = np.linalg.cholesky(build_correlation_matrix(spec))

    n_batches = gen["n_batches"]
    per_batch = gen["records_per_batch"]
    n_total = gen["n_records"]
    period = spec.get("drift", {}).get("diurnal_period_hours", 24)
    drift_p = spec.get("drift", {}).get("parameters", {})
    wear = np.array([drift_p.get(pid, {}).get("wear", 0.0) for pid in ids])
    diurnal = np.array([drift_p.get(pid, {}).get("diurnal", 0.0) for pid in ids])

    shifts = ["A", "B", "C", "D"]
    hour_offsets = [6, 10, 14, 18]
    seeds = np.random.SeedSequence(seed).spawn(n_batches)

    frames = []
    for b in range(n_batches):
        rng = np.random.default_rng(seeds[b])
        z = rng.standard_normal((per_batch, len(ids))) @ chol.T  # correlated N(0,Corr)
        t_in_batch = np.arange(per_batch)
        t_global = b * per_batch + t_in_batch
        # wear: linear trend over the full run, in sigma units.
        wear_term = wear[None, :] * (t_global[:, None] / max(n_total - 1, 1))
        # diurnal: sinusoid per shift hour-offset, in sigma units.
        phase = 2 * np.pi * (t_in_batch + hour_offsets[b % 4]) / period
        diurnal_term = diurnal[None, :] * np.sin(phase)[:, None]
        values = nominal + sigma * (z + wear_term + diurnal_term)

        df = pd.DataFrame(values, columns=ids)
        df["batch_idx"] = b
        df["shift_id"] = shifts[b % 4]
        df["t_in_batch"] = t_in_batch
        df["t_global"] = t_global
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Labels: defect posterior p(y|x), mechanisms, risk targets
# --------------------------------------------------------------------------- #


def _deviations(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    """Signed normalized deviations; |dev| = 1.0 at the nearer spec limit."""
    ids = param_ids(spec)
    dev = np.empty((len(df), len(ids)))
    for j, p in enumerate(spec["parameters"]):
        half = max(p["usl"] - p["nominal"], p["nominal"] - p["lsl"], 1e-9)
        dev[:, j] = (df[p["id"]].to_numpy() - p["nominal"]) / half
    return dev


def _defect_scores(dev: np.ndarray, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """score(defect) = sum over its causal edges of weight * relu(directional dev)."""
    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    scores = {d: np.zeros(dev.shape[0]) for d in defect_names(spec) if d != "no_defect"}
    for e in spec["causal_edges"]:
        col = dev[:, idx[e["parameter"]]]
        signed = col if e["direction"] == "high" else -col
        scores[e["defect"]] += e["weight"] * np.maximum(0.0, signed)
    return scores


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit_columns(scores: dict[str, np.ndarray], gain: float,
                   off: dict[str, float], names: list[str], n: int) -> np.ndarray:
    """Assemble class logits: no_defect is the reference (base 0); defect d is
    gain*score(d). Each class carries an additive calibration offset."""
    cols = []
    for d in names:
        base = np.zeros(n) if d == "no_defect" else gain * scores[d]
        cols.append(base + off[d])
    return np.column_stack(cols)


def _calibrate_priors(scores: dict[str, np.ndarray], gain: float,
                      spec: dict[str, Any]) -> dict[str, float]:
    """Solve additive per-class offsets so E[p(y)] matches the spec priors.

    no_defect is the reference (offset fixed at 0); the defect offsets are found
    by iterative proportional fitting on the softmax marginals.
    """
    names = defect_names(spec)
    priors = {d["name"]: d["prior"] for d in spec["defects"]}
    n = next(iter(scores.values())).shape[0]
    off = {d: 0.0 for d in names}
    for _ in range(200):
        p = _softmax(_logit_columns(scores, gain, off, names, n))
        means = p.mean(axis=0)
        shift = 0.0
        for k, d in enumerate(names):
            if d == "no_defect":
                continue
            delta = float(np.log(priors[d] / max(means[k], 1e-12)))
            off[d] += delta
            shift = max(shift, abs(delta))
        if shift < 1e-10:
            break
    return off


def _calibrate_gain(scores: dict[str, np.ndarray],
                    spec: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Bisect the logit gain so the Bayes error E[1 - max_y p(y|x)] hits target.

    Priors are re-calibrated inside each trial. Higher gain -> sharper posterior
    -> lower Bayes error (monotone), so bisection is well-posed.
    """
    target = spec["label_model"]["target_bayes_error"]
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]
    lo, hi = 0.1, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        off = _calibrate_priors(scores, mid, spec)
        p = _softmax(_logit_columns(scores, mid, off, names, n))
        bayes = float((1.0 - p.max(axis=1)).mean())
        if bayes > target:      # too noisy -> sharpen -> raise gain
            lo = mid
        else:
            hi = mid
    gain = (lo + hi) / 2.0
    return gain, _calibrate_priors(scores, gain, spec)


def _posterior(scores: dict[str, np.ndarray], gain: float, off: dict[str, float],
               spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Return p(y|x) of shape (N, n_defects); column order = defect_names."""
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]
    return _softmax(_logit_columns(scores, gain, off, names, n)), names


def _assign_mechanisms(dev: np.ndarray, defects: np.ndarray,
                       spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-stage ground-truth mechanism = strongest causal edge for the true defect."""
    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    stages = list(spec["mechanisms"].keys())
    pstage = {p["id"]: p["stage"] for p in spec["parameters"]}
    out = {s: np.array(["no_mechanism"] * dev.shape[0], dtype=object) for s in stages}
    best = {s: np.zeros(dev.shape[0]) for s in stages}
    for e in spec["causal_edges"]:
        rows = defects == e["defect"]
        if not rows.any():
            continue
        s = pstage[e["parameter"]]
        col = dev[:, idx[e["parameter"]]]
        signed = col if e["direction"] == "high" else -col
        strength = np.where(rows, e["weight"] * np.maximum(0.0, signed), -1.0)
        take = strength > best[s]
        best[s][take] = strength[take]
        out[s][take] = e["via"]
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def generate(spec: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Generate the full labeled synthetic dataset, deterministic given seed."""
    gen = spec["generator"]
    seed = gen["seed"] if seed is None else seed
    ids = param_ids(spec)
    names = defect_names(spec)

    df = _sample_process(spec, seed)
    dev = _deviations(df, spec)
    scores = _defect_scores(dev, spec)

    gain, offsets = _calibrate_gain(scores, spec)
    post, post_names = _posterior(scores, gain, offsets, spec)
    for k, name in enumerate(post_names):
        df[f"p_{name}"] = post[:, k]

    # Sample labels from p(y|x) with a dedicated, collision-free RNG stream.
    label_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    u = label_rng.random(len(df))
    cdf = np.cumsum(post, axis=1)
    choice = (u[:, None] < cdf).argmax(axis=1)
    df["defect_label"] = np.array(names)[choice]

    mechs = _assign_mechanisms(dev, df["defect_label"].to_numpy(), spec)
    for stage, vals in mechs.items():
        df[f"{stage}_mechanism_label"] = vals

    risks = graded_risk(dev, spec["risk_function"])
    for j, pid in enumerate(ids):
        df[f"risk_{pid}"] = risks[:, j]

    df["board_id"] = [f"PCB_{i:06d}" for i in range(len(df))]
    df["generator_version"] = gen["generator_version"]
    df["seed"] = seed
    df["build_timestamp"] = gen["build_timestamp"]
    df["calibrated_gain"] = gain
    return _assign_split(df, spec)


def _assign_split(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Batch-grouped train/val/test split (no temporal leakage across rows)."""
    gen = spec["generator"]
    batches = sorted(df["batch_idx"].unique())
    n = len(batches)
    n_train = round(n * gen["train_frac"])
    n_val = round(n * gen["val_frac"])
    split = {}
    for i, b in enumerate(batches):
        split[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    df = df.copy()
    df["split"] = df["batch_idx"].map(split)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="SMT synthetic data generator")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--out", default="data/smt_synthetic.csv")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    spec = load_spec(args.spec)
    df = generate(spec, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    counts = df["defect_label"].value_counts(normalize=True).sort_index()
    print(f"Wrote {len(df):,} records to {args.out}")
    print("Class balance:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
