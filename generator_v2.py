"""SMT synthetic generator v2 -- same maths, mechanism-centric output schema.

This reuses every bit of the v1 maths (Gaussian-copula sampling + drift, the
causal defect scores, the calibrated posterior p(y|x), the label sampling, the
per-stage mechanism assignment, and the Eq. 8 graded-risk function) by importing
them from generator.py. Only the output assembly changes, to match the format
the professor asked for:

  Features (unchanged) : the 6 process parameters
  head1  defect        : 3 classes  {no_defect, open_circuit, solder_bridging}
  head2  mechanism     : a JOINT (Cartesian-product) label over the two stages --
                         mechanism_joint = "<printing>__<reflow>". With 3 labels
                         per stage that is 3 x 3 = 9 possible classes (Prof's
                         "table of all possible matches"), so a single softmax
                         head can represent BOTH stages failing at once instead
                         of assuming the two stages are independent. The two
                         per-stage columns are also kept (stage_printing_/
                         stage_reflow_mechanism_label) so an independent-head
                         design stays available for comparison.
  head3  parameter     : per-parameter graded risk risk_<param> (Eq. 8) --
                         identical to v1, the parameter-violation signal.
  head4  risk          : per-MECHANISM risk risk_mech_<mechanism>, all 5 incl.
                         no_mechanism, from the SAME Eq. 8 formula aggregated
                         through the causal map.

Chain: head1 defect -> head2 joint mechanism -> head3 parameter -> head4 risk.

Note on the 9 joint classes: only 7 actually occur. The two cross-defect combos
(aperture_overfill__non_coalescence, poor_paste_transfer__reflow_spreading) are
structurally impossible because a board has a single defect, so they never
appear in training -- a 9-way head simply never predicts them.

Per-mechanism risk: for each real mechanism, take every causal edge that fires
it, score the parameter's deviation in that edge's bad direction with the
unchanged graded_risk(), and keep the MAX over the mechanism's edges. The
no_mechanism risk is the complement 1 - max(real mechanism risks).

The class balance and the process variance are spec-driven, so they can be
overridden from the CLI without editing the YAML:
  --priors      change the defect class balance (the dataset is ~88% no_defect
                by default); e.g. --priors "no_defect=0.5,open_circuit=0.25,
                solder_bridging=0.25". no_defect is auto-filled if omitted.
  --sigma       set a parameter's process spread, e.g. --sigma "paste_viscosity=8".
  --sigma-scale multiply EVERY parameter's spread by a factor, e.g. --sigma-scale 1.5.
  --bayes-error change the target difficulty (see note below).

Note: bumping --sigma alone does NOT make the labels harder -- the gain is
re-calibrated to spec.label_model.target_bayes_error, so the Bayes error is held
fixed regardless of spread. --sigma changes the raw feature spread / out-of-spec
rates and the risk targets; use --bayes-error to actually move difficulty.

Run: python generator_v2.py            # -> data/smt_synthetic_v2.csv
     python generator_v2.py --priors "no_defect=0.5,open_circuit=0.25,solder_bridging=0.25"
     python generator_v2.py --sigma-scale 1.5 --bayes-error 0.08
     python generator_v2.py --help
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# reuse ALL of v1's maths -- nothing here recomputes sampling, scores, or risk
from generator import (_assign_mechanisms, _assign_split, _calibrate_gain,
                       _deviations, _defect_scores, _posterior, _sample_process,
                       defect_names, graded_risk, load_spec, param_ids)

# stamped into every row so the v2 format is distinguishable in provenance
V2_VERSION = "smt-gen-v2-mechrisk"

# separator for the joint mechanism label (mechanism names use single "_")
JOINT_SEP = "__"


def mechanism_vocab(spec: dict[str, Any]) -> list[str]:
    """Algorithm — The flat list of mechanisms used for per-mechanism risk.

    Input: spec (per-stage mechanisms).
    Return: the real mechanisms in spec order, then no_mechanism last.
    """

    # collect each real mechanism once, across all stages, in spec order
    seen: list[str] = []
    for mechs in spec["mechanisms"].values():
        for m in mechs:
            if m != "no_mechanism" and m not in seen:
                seen.append(m)
    return seen + ["no_mechanism"]


def mechanism_risk(dev: np.ndarray,
                   spec: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Algorithm — Per-mechanism risk from the unchanged Eq. 8 formula.

    Input: deviations dev, spec.
    Return: ({mechanism: risk array}, the 5-value vocab order).

    For each real mechanism, every causal edge that fires it contributes
    graded_risk() of its parameter's bad-direction deviation; the mechanism keeps
    the MAX over its edges. no_mechanism gets 1 - max(real mechanism risks).
    """

    rf = spec["risk_function"]
    idx = {pid: i for i, pid in enumerate(param_ids(spec))}
    vocab = mechanism_vocab(spec)
    real = [m for m in vocab if m != "no_mechanism"]
    n = dev.shape[0]

    # each real mechanism = max graded-risk over the parameters on its edges
    risk: dict[str, np.ndarray] = {}
    for m in real:
        rm = np.zeros(n)
        for e in spec["causal_edges"]:
            if e["via"] != m:
                continue
            # deviation pointed in this edge's bad direction, clipped at 0
            col = dev[:, idx[e["parameter"]]]
            signed = col if e["direction"] == "high" else -col
            bad = np.maximum(0.0, signed)
            # SAME Eq. 8 risk function, just fed the bad-direction deviation
            rm = np.maximum(rm, graded_risk(bad, rf))
        risk[m] = rm

    # no_mechanism risk: high when no real mechanism is in violation
    stacked = np.column_stack([risk[m] for m in real])
    risk["no_mechanism"] = 1.0 - stacked.max(axis=1)
    return risk, vocab


def generate_v2(spec: dict[str, Any], seed: int | None = None) -> pd.DataFrame:
    """Algorithm — Generate the v2 (joint-mechanism) labeled dataset.

    Input: spec, seed (defaults to spec.generator.seed).
    Return: the full labeled, split DataFrame in the v2 format.

    Steps 1-5 are byte-for-byte v1 (so features, posteriors, and defect labels
    are identical to smt_synthetic.csv for the same seed). Steps 6-8 are the new
    schema.
    """

    gen = spec["generator"]
    seed = gen["seed"] if seed is None else seed
    ids = param_ids(spec)
    names = defect_names(spec)
    stages = list(spec["mechanisms"].keys())

    # 1-2. correlated, drifting parameters -> normalized deviations (v1 maths)
    df = _sample_process(spec, seed)
    dev = _deviations(df, spec)

    # 3-4. causal scores -> calibrated gain/offsets -> posterior p(y|x) (v1 maths)
    scores = _defect_scores(dev, spec)
    gain, offsets = _calibrate_gain(scores, spec)
    post, post_names = _posterior(scores, gain, offsets, spec)
    for k, name in enumerate(post_names):
        df[f"p_{name}"] = post[:, k]

    # 5. sample the defect label from the posterior (its own stream; v1 maths)
    label_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    u = label_rng.random(len(df))
    cdf = np.cumsum(post, axis=1)
    choice = (u[:, None] < cdf).argmax(axis=1)
    df["defect_label"] = np.array(names)[choice]                        # head1

    # 6. head2: per-stage mechanisms (v1 maths) + their JOINT Cartesian label
    mechs = _assign_mechanisms(dev, df["defect_label"].to_numpy(), spec)
    for stage, vals in mechs.items():
        df[f"{stage}_mechanism_label"] = vals
    # the joint label is the (printing, reflow) pair -> one of the 9 classes
    df["mechanism_joint"] = (df[f"{stages[0]}_mechanism_label"].astype(str)
                             + JOINT_SEP
                             + df[f"{stages[1]}_mechanism_label"].astype(str))

    # 7. head3: per-parameter graded risk (Eq. 8) -- unchanged from v1
    pr = graded_risk(dev, spec["risk_function"])
    for j, pid in enumerate(ids):
        df[f"risk_{pid}"] = pr[:, j]

    # 8. head4: per-mechanism risk (all 5), same Eq. 8 via the causal map
    mrisk, vocab = mechanism_risk(dev, spec)
    for m in vocab:
        df[f"risk_mech_{m}"] = mrisk[m]

    # 9. provenance (v2 version stamp so the format is auditable)
    df["board_id"] = [f"PCB_{i:06d}" for i in range(len(df))]
    df["generator_version"] = V2_VERSION
    df["seed"] = seed
    df["build_timestamp"] = gen["build_timestamp"]
    df["calibrated_gain"] = gain

    # 10. batch-grouped train/val/test split (v1 maths)
    return _assign_split(df, spec)


def _parse_kv(s: str) -> dict[str, float]:
    """Algorithm — Parse a 'name=value,name=value' CLI string.

    Input: a comma-separated key=value string.
    Return: {name: float(value)}.
    """

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
    """Algorithm — Apply CLI overrides to a COPY of the spec.

    Input: spec and the optional override strings/values.
    Return: a deep-copied spec with class priors, per-parameter sigma, a global
            sigma scale, and/or the target Bayes error replaced. Validates names
            and ranges, and (for priors) that the full set sums to 1.
    """

    spec = copy.deepcopy(spec)              # never mutate the caller's spec

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

        # if no_defect was not given, make it the residual so the set sums to 1
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

    # global spread multiplier (applied on top of any per-parameter override)
    if sigma_scale is not None:
        if sigma_scale <= 0:
            raise ValueError("--sigma-scale must be > 0")
        for p in spec["parameters"]:
            p["sigma"] = p["sigma"] * sigma_scale

    # target difficulty (the gain is calibrated to hit this Bayes error)
    if bayes_error is not None:
        if not 0.0 < bayes_error < 0.5:
            raise ValueError("--bayes-error must be in (0, 0.5)")
        spec.setdefault("label_model", {})["target_bayes_error"] = bayes_error

    return spec


def main() -> None:
    """CLI entry: load the spec, apply overrides, generate, write, print balances."""

    # command-line options (own output path so v1 data is untouched)
    ap = argparse.ArgumentParser(description="SMT synthetic data generator v2 (joint mechanism + per-mechanism risk)")
    ap.add_argument("--spec", default="domain/smt_paper.yaml")
    ap.add_argument("--out", default="data/smt_synthetic_v2.csv")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--priors", default=None,
                    help="override defect class balance, e.g. "
                         "'no_defect=0.5,open_circuit=0.25,solder_bridging=0.25' "
                         "(no_defect auto-filled if omitted; must sum to 1)")
    ap.add_argument("--sigma", default=None,
                    help="override per-parameter process spread, e.g. 'paste_viscosity=8'")
    ap.add_argument("--sigma-scale", type=float, default=None,
                    help="multiply EVERY parameter's spread by this factor")
    ap.add_argument("--bayes-error", type=float, default=None,
                    help="override target Bayes error / difficulty (in (0, 0.5))")
    args = ap.parse_args()

    # load the spec, then apply any CLI overrides to a copy of it
    spec = load_spec(args.spec)
    spec = apply_overrides(spec, priors=args.priors, sigma=args.sigma,
                           sigma_scale=args.sigma_scale, bayes_error=args.bayes_error)

    # report the effective knobs so the run is self-documenting
    print("Effective priors :", {d["name"]: round(d["prior"], 4) for d in spec["defects"]})
    print("Effective sigma  :", {p["id"]: round(p["sigma"], 4) for p in spec["parameters"]})
    print("Target Bayes err :", spec["label_model"]["target_bayes_error"])

    # build the dataset
    df = generate_v2(spec, seed=args.seed)

    # write it out (make the folder if needed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # quick class + joint-mechanism balance summary
    print(f"Wrote {len(df):,} records to {args.out}")
    print("\nDefect balance:")
    print(df["defect_label"].value_counts(normalize=True).sort_index().to_string())
    print(f"\nJoint mechanism classes ({df['mechanism_joint'].nunique()} of 9 possible occur):")
    print(df["mechanism_joint"].value_counts().to_string())


if __name__ == "__main__":
    main()
