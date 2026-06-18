"""SMT synthetic process/defect generator (paper replication baseline).

This is the improved, spec-driven successor to the original scratch generator. Everything domain-specific is read from domain/smt_paper.yaml; this
module contains only mechanism, not data. Output is deterministic given a seed.

Pipeline per record:
    1. Draw 6 correlated process parameters via a Gaussian copula (correlation
       matrix from the spec, nearest-PD corrected) with Gaussian marginals.
    2. Layer non-stationary drift (linear wear + diurnal sinusoid) on top.
    3. Compute normalized deviations from nominal (|dev| = 1.0 at the spec limit).
    4. Build defect logits from the causal map and convert to a posterior
       p(y | x); the logit gain is calibrated to a target Bayes error and the
       per-class offsets to the spec priors (paper Table I class balance).
    5. Sample the defect label from p(y | x), assign per-stage mechanism labels,
       and compute the paper's Eq. (8) graded two-sided risk per parameter.
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
# Global seeding (for downstream model code)
# --------------------------------------------------------------------------- #


def seed_everything(seed: int = 42) -> None:
    """Pin every global RNG source we might touch (belt-and-suspenders).

    NOTE: this generator's OUTPUT does not depend on this. Its determinism comes
    from explicit local streams (np.random.SeedSequence(seed).spawn(...)), which
    no library call can overwrite. This helper pins GLOBAL state for downstream
    model training (sklearn falls back to the global RNG when random_state is
    unset; torch is seeded here for the future MLP).
    """
    import os
    import random

    # python, hashing, and the global numpy RNG
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # torch is optional (not a dependency yet) — only seed it if installed
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Apple-GPU RNG: cuda.manual_seed_all does not touch MPS; seed it explicitly
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # deterministic kernels; benchmark MUST be False (autotune is nondeterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # best-effort deterministic ops; warn_only so it never raises on a missing kernel
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Spec loading
# --------------------------------------------------------------------------- #


def load_spec(path: str | Path) -> dict[str, Any]:
    """Algorithm — Load domain spec.

    Input: path to a YAML domain spec.
    Return: spec.
    """

    # read and parse the YAML file into a dict
    with open(path, "r") as fh:
        spec = yaml.safe_load(fh)

    # the blocks every spec must contain
    required = {"parameters", "defects", "mechanisms", "causal_edges",
                "risk_function", "label_model", "generator"}

    # fail fast if any required block is missing
    missing = required - set(spec)
    if missing:
        raise ValueError(f"domain spec missing keys: {sorted(missing)}")
    return spec


def param_ids(spec: dict[str, Any]) -> list[str]:

    # return the ordered parameter ids
    return [p["id"] for p in spec["parameters"]]


def defect_names(spec: dict[str, Any]) -> list[str]:

    # return the ordered defect class names
    return [d["name"] for d in spec["defects"]]


# --------------------------------------------------------------------------- #
# Correlation matrix (Gaussian copula)
# --------------------------------------------------------------------------- #


def _nearest_pd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Algorithm — Nearest positive-definite correlation matrix.

    Input: a symmetric matrix Corr (possibly indefinite), eigenvalue floor eps.
    Return: (A + Aᵀ) / 2.
    """

    # symmetrize the matrix
    sym = (corr + corr.T) / 2.0

    # decomposition of the symmetric matrix
    vals, vecs = np.linalg.eigh(sym)

    # clip values up to eps so the matrix becomes positive-definite
    vals = np.clip(vals, eps, None)

    # reconstruct from the clipped spectrum
    pd = (vecs * vals) @ vecs.T

    # rescale rows/cols to unit diagonal → a valid correlation matrix
    d = np.sqrt(np.diag(pd))
    pd = pd / np.outer(d, d)
    return (pd + pd.T) / 2.0


def build_correlation_matrix(spec: dict[str, Any]) -> np.ndarray:
    """Algorithm — Assemble the correlation matrix from the spec.

    Input: spec with an optional `correlations` block of pairwise values.
    Return: nearest_PD(corr).
    """

    # map each parameter id to its column index
    ids = param_ids(spec)
    idx = {pid: i for i, pid in enumerate(ids)}

    # start from the identity (no correlation)
    n = len(ids)
    corr = np.eye(n)

    # fill in each specified pair, symmetrically
    for a, partners in spec.get("correlations", {}).items():
        for b, rho in partners.items():
            i, j = idx[a], idx[b]
            corr[i, j] = corr[j, i] = float(rho)

    # nudge to the nearest valid positive-definite correlation matrix
    return _nearest_pd(corr)


# --------------------------------------------------------------------------- #
# Risk function (the paper's "Eq. 8"): two-sided graded risk in |deviation|
# --------------------------------------------------------------------------- #


def graded_risk(dev: np.ndarray, rf: dict[str, float]) -> np.ndarray:
    """Algorithm — Graded parameter risk (paper Eq. 8, with 9–10).

    Input: deviations dev and risk params rf = {p_L, p_M, p_H, Δ₁, Δ₂, κ};
           |dev| = 1 sits exactly at the nearer spec limit.
    Return: per-parameter risk P.
    """

    # unpack the risk constants
    p_L, p_M, p_H = rf["p_L"], rf["p_M"], rf["p_H"]
    d1, d2, kappa = rf["delta_1"], rf["delta_2"], rf["kappa"]

    # absolute deviation (risk is two-sided: either spec limit counts)
    u = np.abs(np.asarray(dev, dtype=float))

    # how far into the transition band [d1, d2] we are, capped to [0, 1]
    z = np.clip((u - d1) / max(d2 - d1, 1e-9), 0.0, 1.0)

    # how far past the spec limit we are, 0 if still in spec
    t = np.maximum(0.0, u - d2)

    # low risk in spec, ramps to p_M at the limit, then approaches p_H
    return p_L + (p_M - p_L) * z ** 2 + (p_H - p_M) * (1.0 - np.exp(-kappa * t))


# --------------------------------------------------------------------------- #
# Process sampling
# --------------------------------------------------------------------------- #


def _sample_process(spec: dict[str, Any], seed: int) -> pd.DataFrame:
    """Algorithm — Correlated process sampling with temporal drift.

    Input: spec (parameter nominal/sigma, correlations, drift, generator config), seed.
    Return: one row per record with raw parameter values and batch/time columns.
    """

    # pull config and the per-parameter nominal and sigma
    gen = spec["generator"]
    ids = param_ids(spec)
    nominal = np.array([p["nominal"] for p in spec["parameters"]])
    sigma = np.array([p["sigma"] for p in spec["parameters"]])

    # Cholesky factor turns independent noise into correlated noise
    chol = np.linalg.cholesky(build_correlation_matrix(spec))

    # batch sizes and the drift settings
    n_batches = gen["n_batches"]
    per_batch = gen["records_per_batch"]
    n_total = gen["n_records"]
    period = spec.get("drift", {}).get("diurnal_period_hours", 24)
    drift_p = spec.get("drift", {}).get("parameters", {})
    wear = np.array([drift_p.get(pid, {}).get("wear", 0.0) for pid in ids])
    diurnal = np.array([drift_p.get(pid, {}).get("diurnal", 0.0) for pid in ids])

    # one independent random stream per batch (keeps runs reproducible)
    shifts = ["A", "B", "C", "D"]
    hour_offsets = [6, 10, 14, 18]
    seeds = np.random.SeedSequence(seed).spawn(n_batches)

    frames = []
    for b in range(n_batches):
        rng = np.random.default_rng(seeds[b])

        # correlated standard normals for this batch
        z = rng.standard_normal((per_batch, len(ids))) @ chol.T

        # time index within the batch and across the whole run
        t_in_batch = np.arange(per_batch)
        t_global = b * per_batch + t_in_batch

        # slow equipment wear: a straight-line trend over the run
        wear_term = wear[None, :] * (t_global[:, None] / max(n_total - 1, 1))

        # daily cycle, shifted by this batch's start hour
        phase = 2 * np.pi * (t_in_batch + hour_offsets[b % 4]) / period
        diurnal_term = diurnal[None, :] * np.sin(phase)[:, None]

        # final values: nominal plus sigma times (noise + drift)
        values = nominal + sigma * (z + wear_term + diurnal_term)

        # pack into a frame with batch/shift/time columns
        df = pd.DataFrame(values, columns=ids)
        df["batch_idx"] = b
        df["shift_id"] = shifts[b % 4]
        df["t_in_batch"] = t_in_batch
        df["t_global"] = t_global
        frames.append(df)

    # stack all batches into one table
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Labels: defect posterior p(y|x), mechanisms, risk targets
# --------------------------------------------------------------------------- #


def _deviations(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    """Algorithm — Normalized signed deviations.

    Input: parameter values df and spec.
    Return: dev, where |dev| = 1 means the value sits at its spec limit.
    """

    ids = param_ids(spec)
    dev = np.empty((len(df), len(ids)))

    for j, p in enumerate(spec["parameters"]):
        # distance from nominal to the nearer spec limit
        half = max(p["usl"] - p["nominal"], p["nominal"] - p["lsl"], 1e-9)

        # signed deviation, scaled so the spec limit sits at 1
        dev[:, j] = (df[p["id"]].to_numpy() - p["nominal"]) / half
    return dev


def _defect_scores(dev: np.ndarray, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Algorithm — Causal defect scores.

    Input: deviations dev and the causal edges (defect <- weight * parameter, direction).
    Return: a score per defect (no_defect has none; it is the reference).
    """

    # map parameter id to column, start every defect score at 0
    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    scores = {d: np.zeros(dev.shape[0]) for d in defect_names(spec) if d != "no_defect"}

    for e in spec["causal_edges"]:
        # point the deviation in the direction that causes this defect
        col = dev[:, idx[e["parameter"]]]
        signed = col if e["direction"] == "high" else -col

        # only count deviation in the bad direction, weighted by the edge
        scores[e["defect"]] += e["weight"] * np.maximum(0.0, signed)
    return scores


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, done in a numerically stable way."""

    # subtract the row max before exp so nothing overflows
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit_columns(scores: dict[str, np.ndarray], gain: float,
                   off: dict[str, float], names: list[str], n: int) -> np.ndarray:
    """Algorithm — Assemble class logits.

    Input: scores, gain, per-class offsets off, class order names, row count n.
    Return: the stacked logit columns (one per class).
    """

    cols = []
    for d in names:
        # no_defect is the reference (0); a defect's score is scaled by the gain
        base = np.zeros(n) if d == "no_defect" else gain * scores[d]

        # add this class's calibration offset
        cols.append(base + off[d])
    return np.column_stack(cols)


def _calibrate_priors(scores: dict[str, np.ndarray], gain: float,
                      spec: dict[str, Any]) -> dict[str, float]:
    """Algorithm — Calibrate per-class offsets to the priors.

    Input: scores, gain, spec (defect priors).
    Return: an offset per class so the average class probabilities match the priors.
    """

    # target priors; no_defect is the reference and its offset stays 0
    names = defect_names(spec)
    priors = {d["name"]: d["prior"] for d in spec["defects"]}
    n = next(iter(scores.values())).shape[0]
    off = {d: 0.0 for d in names}

    for _ in range(200):
        # current average probability of each class
        p = _softmax(_logit_columns(scores, gain, off, names, n))
        means = p.mean(axis=0)

        # nudge each defect offset toward its target prior
        shift = 0.0
        for k, d in enumerate(names):
            if d == "no_defect":
                continue
            delta = float(np.log(priors[d] / max(means[k], 1e-12)))
            off[d] += delta
            shift = max(shift, abs(delta))

        # stop once the offsets barely move
        if shift < 1e-10:
            break
    return off


def _calibrate_gain(scores: dict[str, np.ndarray],
                    spec: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Algorithm — Calibrate the logit gain to a target Bayes error.

    Input: scores, spec (label_model.target_bayes_error).
    Return: the gain and the matching per-class offsets.
    """

    # a bigger gain makes the labels sharper, which lowers the Bayes error
    target = spec["label_model"]["target_bayes_error"]
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]

    # binary search on the gain
    lo, hi = 0.1, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2.0

        # fit the priors at this gain, then measure the Bayes error
        off = _calibrate_priors(scores, mid, spec)
        p = _softmax(_logit_columns(scores, mid, off, names, n))
        bayes = float((1.0 - p.max(axis=1)).mean())

        # too noisy -> raise the gain; otherwise lower it
        if bayes > target:
            lo = mid
        else:
            hi = mid

    # final gain and its calibrated offsets
    gain = (lo + hi) / 2.0
    return gain, _calibrate_priors(scores, gain, spec)


def _posterior(scores: dict[str, np.ndarray], gain: float, off: dict[str, float],
               spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Algorithm — Class posterior p(y|x).

    Input: scores, gain, per-class offsets off, spec.
    Return: (p of shape N x n_defects, the class order).
    """

    # softmax of the assembled logits gives the class probabilities
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]
    return _softmax(_logit_columns(scores, gain, off, names, n)), names


def _assign_mechanisms(dev: np.ndarray, defects: np.ndarray,
                       spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Algorithm — Ground-truth stage mechanisms.

    Input: deviations dev, the sampled defect labels, spec.
    Return: one mechanism label per stage.
    """

    # default every row to no_mechanism; track the strongest edge per stage
    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    stages = list(spec["mechanisms"].keys())
    pstage = {p["id"]: p["stage"] for p in spec["parameters"]}
    out = {s: np.array(["no_mechanism"] * dev.shape[0], dtype=object) for s in stages}
    best = {s: np.zeros(dev.shape[0]) for s in stages}

    for e in spec["causal_edges"]:
        # only rows whose true defect matches this edge can use it
        rows = defects == e["defect"]
        if not rows.any():
            continue

        # how strongly this edge fires (deviation in the bad direction, weighted)
        s = pstage[e["parameter"]]
        col = dev[:, idx[e["parameter"]]]
        signed = col if e["direction"] == "high" else -col
        strength = np.where(rows, e["weight"] * np.maximum(0.0, signed), -1.0)

        # keep the mechanism of the strongest edge in that stage
        take = strength > best[s]
        best[s][take] = strength[take]
        out[s][take] = e["via"]
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def generate(spec: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Algorithm — Generate the labeled synthetic dataset (deterministic given seed).

    Input: spec, seed (defaults to spec.generator.seed).
    Return: the full labeled, split DataFrame.
    """

    gen = spec["generator"]
    seed = gen["seed"] if seed is None else seed
    ids = param_ids(spec)
    names = defect_names(spec)

    # 1. draw correlated parameters with drift
    df = _sample_process(spec, seed)

    # 2. normalized deviations from nominal
    dev = _deviations(df, spec)

    # 3. causal score for each defect
    scores = _defect_scores(dev, spec)

    # 4. calibrate difficulty (gain) and class balance (offsets), then the posterior
    gain, offsets = _calibrate_gain(scores, spec)
    post, post_names = _posterior(scores, gain, offsets, spec)
    for k, name in enumerate(post_names):
        df[f"p_{name}"] = post[:, k]

    # 5. sample a label from the posterior (its own random stream)
    label_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    u = label_rng.random(len(df))
    cdf = np.cumsum(post, axis=1)
    choice = (u[:, None] < cdf).argmax(axis=1)
    df["defect_label"] = np.array(names)[choice]

    # 6. mechanism label per stage
    mechs = _assign_mechanisms(dev, df["defect_label"].to_numpy(), spec)
    for stage, vals in mechs.items():
        df[f"{stage}_mechanism_label"] = vals

    # 7. graded risk target per parameter
    risks = graded_risk(dev, spec["risk_function"])
    for j, pid in enumerate(ids):
        df[f"risk_{pid}"] = risks[:, j]

    # 8. provenance columns
    df["board_id"] = [f"PCB_{i:06d}" for i in range(len(df))]
    df["generator_version"] = gen["generator_version"]
    df["seed"] = seed
    df["build_timestamp"] = gen["build_timestamp"]
    df["calibrated_gain"] = gain

    # 9. batch-grouped train/val/test split
    return _assign_split(df, spec)


def _assign_split(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Algorithm — Batch-grouped train/val/test split (no row-level leakage).

    Input: df with batch_idx, spec.generator split fractions.
    Return: df with a `split` column.
    """

    # how many batches go to train and val (the rest is test)
    gen = spec["generator"]
    batches = sorted(df["batch_idx"].unique())
    n = len(batches)
    n_train = round(n * gen["train_frac"])
    n_val = round(n * gen["val_frac"])

    # assign whole batches (not rows) so no batch is split across sets
    split = {}
    for i, b in enumerate(batches):
        split[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    # attach the split label to every row
    df = df.copy()
    df["split"] = df["batch_idx"].map(split)
    return df


def main() -> None:
    """CLI entry: load the spec, generate the data, write the CSV, print the balance."""

    # command-line options
    ap = argparse.ArgumentParser(description="SMT synthetic data generator")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--out", default="data/smt_synthetic.csv")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # load the spec and build the dataset
    spec = load_spec(args.spec)
    df = generate(spec, seed=args.seed)

    # write it out (make the folder if needed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # quick class-balance summary
    counts = df["defect_label"].value_counts(normalize=True).sort_index()
    print(f"Wrote {len(df):,} records to {args.out}")
    print("Class balance:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
