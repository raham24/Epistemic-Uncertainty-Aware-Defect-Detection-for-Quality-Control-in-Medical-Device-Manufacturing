"""SMT synthetic process/defect generator (self-contained).

Everything domain-specific is read from domain/smt_paper.yaml; this file holds
only mechanism, no data, and depends on nothing in the repo. Output is
deterministic given a seed.

Pipeline (per record):
  1. Draw 6 correlated process parameters via a Gaussian copula (correlation
     matrix from the spec, nearest-PD corrected), with Gaussian marginals.
  2. Layer non-stationary drift (linear wear + diurnal sinusoid) on top.
  3. Compute normalized deviations from nominal (|dev| = 1.0 at the spec limit).
  4. Build defect logits from the causal map -> posterior p(y|x). The logit gain
     is calibrated to a target Bayes error; per-class offsets to the spec priors.
  5. Sample the defect label from p(y|x); assign per-stage mechanism labels.
  6. Compute risk targets (all from the SAME Eq. 8 graded-risk function):
       - risk_<param>           : global two-sided per-parameter risk.
       - risk_mech_<m>_<param>  : per-mechanism risk GATED to that mechanism's
                                  own parameters (0 elsewhere; no_mechanism all 0).
  7. Stamp provenance (generator version, seed, batch, timestamp) and split.

Output columns:
  features         : the 6 process parameters (raw values)
  p_<defect>       : the exact posterior p(y|x) (lets us compute the Bayes error)
  defect_label     : one of {no_defect, open_circuit, solder_bridging}
  <stage>_mechanism_label : per-stage ground-truth mechanism
  mechanism_joint  : "<printing>__<reflow>" -- the 3x3 = 9-class joint label
  risk_<param> x6  : global graded risk per parameter
  risk_mech_<m>_<param> x30 : mechanism-gated per-parameter risk (5 mech x 6 param)
  provenance + split

Run: python generator.py
     python generator.py --priors "no_defect=0.5,open_circuit=0.25,solder_bridging=0.25"
     python generator.py --bayes-error 0.09 --sigma-scale 1.5
     python generator.py --help
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# stamped into every row for provenance
VERSION = "smt-gen-2.0"

# separator for the joint mechanism label (mechanism names use single "_")
JOINT_SEP = "__"


# --------------------------------------------------------------------------- #
# Seeding (also used by the model code)
# --------------------------------------------------------------------------- #

def seed_everything(seed: int = 42) -> None:
    """Pin every global RNG source (python, numpy, and torch if installed).

    The generator's OUTPUT does not depend on this -- its determinism comes from
    explicit local streams (np.random.SeedSequence(seed).spawn(...)). This pins
    GLOBAL state for downstream model training.
    """

    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #

def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate the YAML domain spec."""

    with open(path, "r") as fh:
        spec = yaml.safe_load(fh)

    required = {"parameters", "defects", "mechanisms", "causal_edges",
                "risk_function", "label_model", "generator"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"domain spec missing keys: {sorted(missing)}")
    return spec


def param_ids(spec: dict[str, Any]) -> list[str]:
    """The ordered parameter ids."""
    return [p["id"] for p in spec["parameters"]]


def defect_names(spec: dict[str, Any]) -> list[str]:
    """The ordered defect class names."""
    return [d["name"] for d in spec["defects"]]


def mechanism_vocab(spec: dict[str, Any]) -> list[str]:
    """The flat mechanism list: each real mechanism once (spec order), then
    no_mechanism last. Used for the per-mechanism gated risk."""

    seen: list[str] = []
    for mechs in spec["mechanisms"].values():
        for m in mechs:
            if m != "no_mechanism" and m not in seen:
                seen.append(m)
    return seen + ["no_mechanism"]


# --------------------------------------------------------------------------- #
# Correlation matrix (Gaussian copula)
# --------------------------------------------------------------------------- #

def _nearest_pd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Nearest positive-definite correlation matrix (eigenvalue clip + rescale)."""

    sym = (corr + corr.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    pd_ = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(pd_))
    pd_ = pd_ / np.outer(d, d)
    return (pd_ + pd_.T) / 2.0


def build_correlation_matrix(spec: dict[str, Any]) -> np.ndarray:
    """Assemble the 6x6 correlation matrix from the spec's pairwise values."""

    ids = param_ids(spec)
    idx = {pid: i for i, pid in enumerate(ids)}
    corr = np.eye(len(ids))
    for a, partners in spec.get("correlations", {}).items():
        for b, rho in partners.items():
            i, j = idx[a], idx[b]
            corr[i, j] = corr[j, i] = float(rho)
    return _nearest_pd(corr)


# --------------------------------------------------------------------------- #
# Graded risk (paper Eq. 8/9/10): two-sided in |deviation|
# --------------------------------------------------------------------------- #

def graded_risk(dev: np.ndarray, rf: dict[str, float]) -> np.ndarray:
    """Graded parameter risk. |dev| = 1 sits at the nearer spec limit.

    Works elementwise on any shape (per-parameter 2D, or a 1D column).
    """

    p_L, p_M, p_H = rf["p_L"], rf["p_M"], rf["p_H"]
    d1, d2, kappa = rf["delta_1"], rf["delta_2"], rf["kappa"]
    u = np.abs(np.asarray(dev, dtype=float))                 # two-sided
    z = np.clip((u - d1) / max(d2 - d1, 1e-9), 0.0, 1.0)     # transition band
    t = np.maximum(0.0, u - d2)                              # past the limit
    return p_L + (p_M - p_L) * z ** 2 + (p_H - p_M) * (1.0 - np.exp(-kappa * t))


# --------------------------------------------------------------------------- #
# Process sampling
# --------------------------------------------------------------------------- #

def _sample_process(spec: dict[str, Any], seed: int) -> pd.DataFrame:
    """Correlated process sampling with temporal drift -> one row per record."""

    gen = spec["generator"]
    ids = param_ids(spec)
    nominal = np.array([p["nominal"] for p in spec["parameters"]])
    sigma = np.array([p["sigma"] for p in spec["parameters"]])
    chol = np.linalg.cholesky(build_correlation_matrix(spec))

    n_batches = gen["n_batches"]
    per_batch = gen["records_per_batch"]
    n_total = n_batches * per_batch          # run length = the wear-drift horizon
    period = spec.get("drift", {}).get("diurnal_period_hours", 24)
    drift_p = spec.get("drift", {}).get("parameters", {})
    wear = np.array([drift_p.get(pid, {}).get("wear", 0.0) for pid in ids])
    diurnal = np.array([drift_p.get(pid, {}).get("diurnal", 0.0) for pid in ids])

    shifts = ["A", "B", "C", "D"]
    hour_offsets = [6, 10, 14, 18]
    seeds = np.random.SeedSequence(seed).spawn(n_batches)    # one stream per batch

    frames = []
    for b in range(n_batches):
        rng = np.random.default_rng(seeds[b])
        z = rng.standard_normal((per_batch, len(ids))) @ chol.T   # correlated noise

        t_in_batch = np.arange(per_batch)
        t_global = b * per_batch + t_in_batch
        wear_term = wear[None, :] * (t_global[:, None] / max(n_total - 1, 1))
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


def _deviations(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    """Normalized signed deviations; |dev| = 1 means the value is at its limit."""

    ids = param_ids(spec)
    dev = np.empty((len(df), len(ids)))
    for j, p in enumerate(spec["parameters"]):
        half = max(p["usl"] - p["nominal"], p["nominal"] - p["lsl"], 1e-9)
        dev[:, j] = (df[p["id"]].to_numpy() - p["nominal"]) / half
    return dev


# --------------------------------------------------------------------------- #
# Defect posterior p(y|x)
# --------------------------------------------------------------------------- #

def _defect_scores(dev: np.ndarray, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Causal defect scores: sum over edges of weight * relu(directional dev)."""

    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    scores = {d: np.zeros(dev.shape[0]) for d in defect_names(spec) if d != "no_defect"}
    for e in spec["causal_edges"]:
        col = dev[:, idx[e["parameter"]]]
        signed = col if e["direction"] == "high" else -col
        scores[e["defect"]] += e["weight"] * np.maximum(0.0, signed)
    return scores


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit_columns(scores: dict[str, np.ndarray], gain: float,
                   off: dict[str, float], names: list[str], n: int) -> np.ndarray:
    """Stack per-class logits: no_defect = offset (reference); defect = gain*score + offset."""
    cols = []
    for d in names:
        base = np.zeros(n) if d == "no_defect" else gain * scores[d]
        cols.append(base + off[d])
    return np.column_stack(cols)


def _calibrate_priors(scores: dict[str, np.ndarray], gain: float,
                      spec: dict[str, Any]) -> dict[str, float]:
    """Solve per-class offsets so the mean class probabilities match the priors."""

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
    """Binary-search the logit gain so the Bayes error hits target_bayes_error."""

    target = spec["label_model"]["target_bayes_error"]
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]
    lo, hi = 0.1, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        off = _calibrate_priors(scores, mid, spec)
        p = _softmax(_logit_columns(scores, mid, off, names, n))
        bayes = float((1.0 - p.max(axis=1)).mean())
        if bayes > target:                       # too noisy -> raise the gain
            lo = mid
        else:
            hi = mid
    gain = (lo + hi) / 2.0
    return gain, _calibrate_priors(scores, gain, spec)


def _posterior(scores: dict[str, np.ndarray], gain: float, off: dict[str, float],
               spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Class posterior p(y|x) = softmax of the assembled logits."""
    names = defect_names(spec)
    n = next(iter(scores.values())).shape[0]
    return _softmax(_logit_columns(scores, gain, off, names, n)), names


# --------------------------------------------------------------------------- #
# Mechanisms + per-mechanism gated risk
# --------------------------------------------------------------------------- #

def _assign_mechanisms(dev: np.ndarray, defects: np.ndarray,
                       spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Ground-truth mechanism per stage: the strongest firing edge whose defect
    matches the row's label, else no_mechanism."""

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


def mechanism_param_risk(dev: np.ndarray, spec: dict[str, Any]
                         ) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Per-mechanism, per-parameter risk GATED by mechanism.

    For each real mechanism, only the parameters on that mechanism's causal edges
    carry risk: each edge scores its parameter's bad-direction deviation with the
    SAME graded_risk() and writes it into that (mechanism, parameter) cell (MAX if
    a parameter recurs on the mechanism's edges). Every other parameter stays 0,
    and no_mechanism is 0 everywhere.

    Return: ({mechanism: (n, n_params) risk}, mechanism vocab, parameter ids).
    """

    rf = spec["risk_function"]
    ids = param_ids(spec)
    idx = {pid: i for i, pid in enumerate(ids)}
    vocab = mechanism_vocab(spec)
    n = dev.shape[0]

    risk: dict[str, np.ndarray] = {m: np.zeros((n, len(ids))) for m in vocab}
    for e in spec["causal_edges"]:
        m = e["via"]
        if m == "no_mechanism":
            continue
        j = idx[e["parameter"]]
        col = dev[:, j]
        signed = col if e["direction"] == "high" else -col
        bad = np.maximum(0.0, signed)
        risk[m][:, j] = np.maximum(risk[m][:, j], graded_risk(bad, rf))
    return risk, vocab, ids


def _assign_split(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Batch-grouped train/val/test split (whole batches, no row-level leakage)."""

    gen = spec["generator"]
    batches = sorted(df["batch_idx"].unique())
    n = len(batches)
    n_train = round(n * gen["train_frac"])
    n_val = round(n * gen["val_frac"])
    split = {}
    for i, b in enumerate(batches):
        split[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    df = df.copy()
    df["split"] = df["batch_idx"].map(split)                 # rest -> test
    return df


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def generate(spec: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Generate the full labeled, split dataset (deterministic given the seed)."""

    gen = spec["generator"]
    seed = gen["seed"] if seed is None else seed
    ids = param_ids(spec)
    names = defect_names(spec)
    stages = list(spec["mechanisms"].keys())

    # 1-2. correlated, drifting parameters -> normalized deviations
    df = _sample_process(spec, seed)
    dev = _deviations(df, spec)

    # 3-4. causal scores -> calibrated gain/offsets -> posterior p(y|x)
    scores = _defect_scores(dev, spec)
    gain, offsets = _calibrate_gain(scores, spec)
    post, post_names = _posterior(scores, gain, offsets, spec)
    for k, name in enumerate(post_names):
        df[f"p_{name}"] = post[:, k]

    # 5. sample the defect label from the posterior (its own stream)
    label_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    u = label_rng.random(len(df))
    cdf = np.cumsum(post, axis=1)
    choice = (u[:, None] < cdf).argmax(axis=1)
    df["defect_label"] = np.array(names)[choice]

    # 6. per-stage mechanisms + their JOINT Cartesian label (the 9-class target)
    mechs = _assign_mechanisms(dev, df["defect_label"].to_numpy(), spec)
    for stage, vals in mechs.items():
        df[f"{stage}_mechanism_label"] = vals
    df["mechanism_joint"] = (df[f"{stages[0]}_mechanism_label"].astype(str)
                             + JOINT_SEP
                             + df[f"{stages[1]}_mechanism_label"].astype(str))

    # 7. global per-parameter graded risk
    pr = graded_risk(dev, spec["risk_function"])
    for j, pid in enumerate(ids):
        df[f"risk_{pid}"] = pr[:, j]

    # 8. per-mechanism, per-parameter gated risk (all 5 mechanisms x 6 params)
    mrisk, vocab, _ = mechanism_param_risk(dev, spec)
    for m in vocab:
        for j, pid in enumerate(ids):
            df[f"risk_mech_{m}_{pid}"] = mrisk[m][:, j]

    # 9. provenance
    df["board_id"] = [f"PCB_{i:06d}" for i in range(len(df))]
    df["generator_version"] = VERSION
    df["seed"] = seed
    df["build_timestamp"] = gen["build_timestamp"]
    df["calibrated_gain"] = gain

    # 10. batch-grouped split
    return _assign_split(df, spec)


# --------------------------------------------------------------------------- #
# CLI overrides
# --------------------------------------------------------------------------- #

def _parse_kv(s: str) -> dict[str, float]:
    """Parse a 'name=value,name=value' string into {name: float}."""
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected name=value, got '{part}'")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def apply_overrides(spec: dict[str, Any], priors: str | None = None,
                    sigma: str | None = None, sigma_scale: float | None = None,
                    bayes_error: float | None = None) -> dict[str, Any]:
    """Apply CLI overrides to a deep copy of the spec (class balance, per-parameter
    sigma, a global sigma scale, and/or the target Bayes error). Validated; the
    caller's spec is never mutated."""

    spec = copy.deepcopy(spec)

    # class balance: override defect priors (no_defect auto-filled if omitted)
    if priors:
        kv = _parse_kv(priors)
        names = {d["name"] for d in spec["defects"]}
        unknown = set(kv) - names
        if unknown:
            raise ValueError(f"unknown defect(s) {sorted(unknown)}; choices {sorted(names)}")
        for d in spec["defects"]:
            if d["name"] in kv:
                d["prior"] = kv[d["name"]]
        if "no_defect" in names and "no_defect" not in kv:
            others = sum(d["prior"] for d in spec["defects"] if d["name"] != "no_defect")
            for d in spec["defects"]:
                if d["name"] == "no_defect":
                    d["prior"] = 1.0 - others
        total = sum(d["prior"] for d in spec["defects"])
        if abs(total - 1.0) > 1e-3:
            cur = {d["name"]: round(d["prior"], 4) for d in spec["defects"]}
            raise ValueError(f"defect priors must sum to 1.0 (got {total:.4f}): {cur}")
        for d in spec["defects"]:
            if not 0.0 < d["prior"] < 1.0:
                raise ValueError(f"prior for {d['name']} must be in (0,1), got {d['prior']}")

    # per-parameter process spread
    if sigma:
        kv = _parse_kv(sigma)
        ids = set(param_ids(spec))
        unknown = set(kv) - ids
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; choices {sorted(ids)}")
        for p in spec["parameters"]:
            if p["id"] in kv:
                if kv[p["id"]] <= 0:
                    raise ValueError(f"sigma for {p['id']} must be > 0")
                p["sigma"] = kv[p["id"]]

    # global spread multiplier (on top of any per-parameter override)
    if sigma_scale is not None:
        if sigma_scale <= 0:
            raise ValueError("--sigma-scale must be > 0")
        for p in spec["parameters"]:
            p["sigma"] = p["sigma"] * sigma_scale

    # target difficulty (the gain is calibrated to hit this Bayes error)
    if bayes_error is not None:
        if not 0.0 < bayes_error < 0.5:
            raise ValueError("--bayes-error must be in (0, 0.5)")
        spec["label_model"]["target_bayes_error"] = bayes_error

    return spec


def main() -> None:
    """CLI: load the spec, apply overrides, generate, write, print balances."""

    ap = argparse.ArgumentParser(description="SMT synthetic data generator")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--out", default="data/smt_synthetic.csv")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--priors", default=None,
                    help="override defect class balance, e.g. "
                         "'no_defect=0.5,open_circuit=0.25,solder_bridging=0.25' "
                         "(no_defect auto-filled if omitted; must sum to 1)")
    ap.add_argument("--sigma", default=None,
                    help="override per-parameter spread, e.g. 'paste_viscosity=8'")
    ap.add_argument("--sigma-scale", type=float, default=None,
                    help="multiply EVERY parameter's spread by this factor")
    ap.add_argument("--bayes-error", type=float, default=None,
                    help="override target Bayes error / difficulty (in (0, 0.5))")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    spec = apply_overrides(spec, priors=args.priors, sigma=args.sigma,
                           sigma_scale=args.sigma_scale, bayes_error=args.bayes_error)

    print("Effective priors :", {d["name"]: round(d["prior"], 4) for d in spec["defects"]})
    print("Effective sigma  :", {p["id"]: round(p["sigma"], 4) for p in spec["parameters"]})
    print("Target Bayes err :", spec["label_model"]["target_bayes_error"])

    df = generate(spec, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} records to {args.out}")
    print("\nDefect balance:")
    print(df["defect_label"].value_counts(normalize=True).sort_index().to_string())
    print(f"\nJoint mechanism classes ({df['mechanism_joint'].nunique()} of 9 possible occur):")
    print(df["mechanism_joint"].value_counts().to_string())


if __name__ == "__main__":
    main()
