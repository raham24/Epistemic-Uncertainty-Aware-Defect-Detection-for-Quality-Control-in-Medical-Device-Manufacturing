# Medical Device Neurosymbolic RCA — SMT Replication Baseline

Replication of the synthetic-data, model, and difficulty-characterization stages
of Shenoy & Ameri (2026), *"Uncertainty-Aware Neurosymbolic Root-Cause Analysis
for Surface-Mount Assembly,"* IEEE Trans. Semiconductor Manufacturing
(DOI 10.1109/TSM.2026.3673999). This is the safety-net SMT baseline that the
medical-device port (external insulin pumps) builds on later — see `CLAUDE.md`
for the full project plan.

## What's here

A spec-driven synthetic SMT data generator, distribution histograms, exact
Bayes-error analysis, and the multi-head MLP that learns from it. The dataset and
labels are aligned to the paper: parameters, defect classes, mechanisms, the
Fig. 2 causal map, the Table I class balance, and the Eq. (8) graded risk. Values
the paper does not release are reconstructed and calibrated to its reported
anchors (see `docs/algorithm.md` reproducibility gaps).

Two **self-contained files** at the repo root are the current pipeline — each
depends only on `domain/smt_paper.yaml` and standard libraries, nothing else in
the repo:

- **`generator.py`** — the spec-driven synthetic generator. Deterministic given a
  seed; emits the joint mechanism label and the gated per-mechanism risk.
- **`mlp.py`** — a 3-head **true cascade** (defect → joint 9-class mechanism →
  per-parameter risk), each head conditioned on the upstream heads' predictions.
  Two selectable losses (`--loss cascade|abstention`).

The original **paper replication** is archived under **`v1/`** (its own
`generator.py`, `mlp_paper.py`, `mlp.py`, shared `mlp_common.py`, `plots.ipynb`) —
paper-faithful, with two independent per-stage mechanism heads. The root files no
longer import from `v1/`; the two generations are fully independent.

## Quickstart

```bash
conda env create -f environment.yml      # or: conda activate paper (if it exists)
conda activate paper

python generator.py        # -> data/smt_synthetic.csv  (200k records, deterministic)
python mlp.py              # train the cascade -> defect/mechanism ~0.95, risk MAE ~0.01
python mlp.py --loss abstention --o 2.0   # selective-classification loss on the two
                                          # classification heads (--o = the payoff)
# then open plots.ipynb to analyse the saved checkpoint (no training in the notebook)
```

Everything runs on CPU (and Apple `mps`); no GPU required. All runs are seeded and
reproducible. The archived paper baseline lives in `v1/`; run it from the repo
**root** (`python v1/mlp_paper.py`) so `data/`, `domain/`, `results/` resolve. Note
that `v1/generator.py` writes the same `data/smt_synthetic.csv` path but in the
paper schema (no joint mechanism), so regenerate with the root `generator.py`
before training `mlp.py`.

## Layout

```
domain/smt_paper.yaml   THE domain spec (single source of truth; generator.py reads it)

generator.py            self-contained spec-driven generator (joint mechanism + gated risk)
mlp.py                  self-contained cascaded 3-head MLP (defect -> mechanism -> risk)
plots.ipynb             analysis: loads results/mlp_model.pt, one plot per cell
cluster/                multi-config sweep (SLURM + local fallback) -> cluster_analysis.ipynb

v1/                     archived paper replication (own generator.py / mlp_paper.py / mlp.py
                        / mlp_common.py / plots.ipynb); two independent per-stage heads

docs/algorithm.md       pseudocode, improvements over the original, reproducibility gaps
docs/results.md         team-facing results summary
docs/Generator_Overview.docx   high-level overview + paper comparison (generated)
figs/                   generated figures
results/                metrics JSON + checkpoints (mlp_model.pt, mlp_metrics.json, ...)
data/                   generated datasets (gitignored)
archive/                superseded original scratch generator (truncated stub)
```

Local-only and **gitignored** (not pushed): `misc/` (tests + the plots-notebook
builder), `legacy/` (superseded analysis scripts), and the cluster sweep outputs
(`results/cluster/`, generated matrix files).

## Model (`mlp.py`)

A shared trunk `h = f(x)` (FC, ReLU, dropout) feeds a 3-head **true cascade** —
each downstream head reads the trunk latent concatenated with the upstream heads'
**softmax** distributions (soft, so it stays differentiable and downstream losses
also sharpen the upstream heads). Trained end-to-end on the model's own
predictions (no teacher forcing):

```
x → trunk → h
   head1 defect : Linear(h)                       → 3-way softmax
   head2 mech   : Linear(h ⊕ p_defect)            → 9-way softmax   [joint mechanism_joint]
   head3 risk   : Linear(h ⊕ p_defect ⊕ p_mech)   → 6 sigmoids      [global risk_<param>]
```

- **head2** is ONE 9-way softmax over the joint mechanism label (the 3×3 Cartesian
  product of the two stages), not two independent per-stage heads. Only 7 of the 9
  classes occur — the two cross-defect combos are structurally impossible.
- **head3** is the GLOBAL, two-sided per-parameter risk (`risk_<param>`) — a
  function of how far each parameter drifted, NOT of the mechanism label. So it
  always reports a per-parameter risk, even when head2 predicts `no_mechanism`: a
  clean board sits at the ~`p_L` floor; a sub-threshold-drifting board shows
  elevated risk on the drifting parameter. The cascade feeds head2's call to head3
  as a *hint*, but the target never suppresses it.

**Two losses** (`--loss`):

| `--loss` | defect + mechanism heads | risk head |
|---|---|---|
| `cascade` (default) | cross-entropy | BCE w/ soft targets |
| `abstention` | selective-classification `-log(o·p_y + r)` | BCE w/ soft targets |

`L = λ_d·loss(defect) + λ_m·loss(mech) + λ_r·BCE(risk)`. The risk head regresses
the graded continuous target from the generator's Eq. 8 risk function (BCE with
**soft** targets in (0,1)), never a binary in/out-of-spec label.

The **abstention** loss adds one "abstain" output to each classification head; `r`
is its abstain probability and `o` the payoff (`--o`). `r = 0` reduces the term to
cross-entropy, so larger `o` ⇒ predict more / abstain less. The risk head is never
widened. `evaluate()` then reports coverage and selective accuracy at r-thresholds
0.3/0.5/0.7. (Selective-classification form: Deep Gamblers, Liu et al. 2019.)

**Why the joint (Cartesian) mechanism head.** A defective board is usually
implicated at *both* stages (≈94% of mechanism-bearing boards have a printing
*and* a reflow mechanism — e.g. bridging = `aperture_overfill` + `reflow_spreading`).
Two independent per-stage heads (the paper's structure, kept in `v1/mlp_paper.py`)
assume the stages are conditionally independent given the features; a single
softmax over the 9 joint classes models both stages failing at once directly
(per Prof. Romanelli's suggestion). The two cross-defect combos
(`aperture_overfill__non_coalescence`, `poor_paste_transfer__reflow_spreading`) are
structurally impossible because a board has one defect, so a 9-way head never
predicts them.

## Generator (`generator.py`)

Reads `domain/smt_paper.yaml`, draws 6 correlated drifting parameters, calibrates
a posterior `p(y|x)` to a target Bayes error and the class priors, samples labels,
and writes these columns:

| Column(s) | What it is |
|---|---|
| 6 parameters | the raw process values (the model features) |
| `p_<defect>` ×3 | the exact posterior `p(y|x)` (lets us compute the Bayes error) |
| `defect_label` | 3 classes {no_defect, open_circuit, solder_bridging} |
| `<stage>_mechanism_label` ×2 | per-stage ground-truth mechanism |
| `mechanism_joint` | `"<printing>__<reflow>"` — the 3×3 = 9-class joint label (7 live) |
| `risk_<param>` ×6 | global, two-sided graded risk (Eq. 8) per parameter |
| `risk_mech_<mech>_<param>` ×30 | per-mechanism risk **gated** to that mechanism's own parameters (0 elsewhere; `no_mechanism` 0 everywhere); `mlp.py` doesn't use these, but they're verified in `misc/test_generator_risk.py` |

`mlp.py` trains on `mechanism_joint` (head2) and `risk_<param>` (head3); the
per-stage labels and the gated `risk_mech_*` columns are extra signal in the CSV.

**Gated per-mechanism risk.** For each real mechanism, every causal edge that
fires it scores its parameter's bad-direction deviation with the same
`graded_risk()` and writes it into that `(mechanism, parameter)` cell (**max** if
a parameter recurs on the mechanism's edges); every parameter the mechanism does
not drive stays 0, and `no_mechanism` is 0 everywhere.

### Tuning the dataset (CLI overrides)

The class balance and process variance are spec-driven, so they are overridable
without editing the YAML (each is validated; the original spec is never mutated):

| flag | effect | example |
|---|---|---|
| `--priors` | defect class balance (the "split"); `no_defect` auto-filled if omitted, must sum to 1 | `--priors "open_circuit=0.25,solder_bridging=0.25"` → 0.5/0.25/0.25 |
| `--sigma` | per-parameter process spread | `--sigma "paste_viscosity=8"` |
| `--sigma-scale` | multiply **every** parameter's spread | `--sigma-scale 1.5` |
| `--bayes-error` | target difficulty, in (0, 0.5) | `--bayes-error 0.08` |

```bash
# rebalance away from the ~88% no_defect default
python generator.py --priors "no_defect=0.5,open_circuit=0.25,solder_bridging=0.25" --out data/smt_balanced.csv

# wider spread AND harder labels
python generator.py --sigma-scale 1.5 --bayes-error 0.08 --out data/smt_hard.csv
```

Note: `--sigma` alone does **not** change difficulty — the logit gain is
re-calibrated to `target_bayes_error`, so the Bayes floor is held fixed
regardless of spread. `--sigma` changes the raw feature distributions,
out-of-spec rates, and risk targets; `--bayes-error` is the knob that actually
moves difficulty. Every run prints the effective priors, sigmas, and target
Bayes error up front. The `cluster/` sweep trains both losses over several of
these dataset variants in parallel (see `cluster/README.md`).

## Key results

`python mlp.py` writes `results/mlp_model.pt` + `results/mlp_metrics.json`; then
open `plots.ipynb` (rebuild with `python misc/_build_plots.py`). Model wiring
is covered by `misc/test_mlp_smoke.py`. Test split, seed 0, `--loss cascade`:

| Head | Metric | Value | Reference |
|---|---|---|---|
| defect (head1) | accuracy / wF1 / macroF1 | 0.9516 / 0.9503 / 0.8589 | Bayes ceiling 0.9534; paper 0.95 |
| mechanism (head2, joint 9-class) | accuracy | 0.9511 | printing 0.9516 / reflow 0.9551 (decomposed) |
| risk (head3) | MAE | 0.0105 | — |

- Class balance matches paper Table I: 0.877 / 0.060 / 0.063.
- Bayes floor (oracle ceiling) ~95.5% (population) / ~95.3% (test); the cascade
  reaches ~95.1% defect accuracy, matching the paper's reported 95.00%.
- The `abstention` loss concentrates rejections on the hard minority classes
  (open/bridge) — selective accuracy rises as coverage drops.
- All outputs (data, training) are reproducible given the seed.

## Paper-to-code mapping

How the generator and dataset line up with Shenoy & Ameri (2026), §IV-A/B/C,
Table I, and Fig. 2.

### Exact matches

What matches the paper exactly. Schema, vocabulary, class balance, causal
topology, split protocol, and the risk-equation form all match the paper.

| Paper (Section / Table / Fig) | Generator Code |
|---|---|
| Gaussian-copula sampling + nearest-PD correction (Section IV-A) | `_sample_process` + `_nearest_pd` |
| batch/shift groups, different seeds + start-hour offsets (Section IV-A) | per-batch `SeedSequence.spawn()`; `hour_offsets` [6,10,14,18]; shifts A–D |
| 6 params, 2 stages, exact names (Section IV-A, Table I) | printing {RH, ambient temp, paste viscosity, stencil thickness}; reflow {time-above-liquidus, peak reflow temp} |
| 3 defect classes (Section IV-B) | `no_defect`, `open_circuit`, `solder_bridging` |
| stage-wise mechanism vocab incl. explicit `no_mechanism` (Table I) | printing {aperture_overfill, poor_paste_transfer, no_mechanism}; reflow {reflow_spreading, non_coalescence, no_mechanism} |
| graded risk Eq. 8–10; profile 0.05→0.70→0.99 (Section IV-B) | `graded_risk`; p_L/p_M/p_H = 0.05/0.70/0.99 |
| batch/shift grouped split; normalize on train only (Section IV-C) | `_assign_split`; train-only standardization (`mu`/`sd` fit on train) |
| N=200,000; 140k/30k/30k (Table I) | `n_batches`×`records_per_batch` = 100×2000; train/val/test = 0.70/0.15/0.15 |
| class counts 175,420 / 12,569 / 12,011 (Table I) | calibrated to those exact fractions (IPF on logit offsets) |
| Fig. 2 causal topology (bridging via aperture_overfill + reflow_spreading; open = mirror) | all 12 `causal_edges`, mirror directions |
| multi-head MLP: softmax defect + per-stage mechanism, sigmoid risk (Eq. 1–3) | `v1/mlp_common.MultiHeadMLP` (the paper-faithful baseline; the root `mlp.py` instead uses a single joint mechanism head in a cascade) |

### Inferred, diverged, and missing

What the paper did **not** pin down, or where we deviate from what it described.
**Inferred**: the paper leaves the value blank; ours is a documented choice, not
a contradiction. **Diverged**: the paper describes something specific and we
simplified it. **Missing**: the paper describes a dataset element we do not generate.

| Item | Paper status | What we did (ours) | Category |
|---|---|---|---|
| Correlation-matrix values | not released | hand-set pairwise ρ in `correlations` block | Inferred |
| Per-parameter sigmas + spec widths (nominal/LSL/USL) | not released | chosen per parameter in the spec | Inferred |
| Drift coefficients | not released | `wear` / `diurnal` amplitudes in `drift` block | Inferred |
| Causal-edge weights | not released | `weight` on each of the 12 `causal_edges` | Inferred |
| Risk constants Δ₁, Δ₂, κ | only Eq. 8–10 form + p_L/p_M/p_H given | Δ₁ = 0.5, Δ₂ = 1.0, κ = 3.0 | Inferred |
| MLP loss function | never stated | `L = λ_d·CE + Σ λ_m·CE + λ_r·BCE` (paper); selective-classification (`abstention`) loss variant in `mlp.py` | Inferred |
| MLP trunk depth/width, dropout, lr, batch, optimizer | not released | 2×256 ReLU, dropout 0.1, Adam, lr 1e-3, batch 128 (Optuna-tunable via `tune.py`) | Inferred |
| Stencil-thickness wear "with periodic replacement" (Section IV-A) | described | single monotonic linear ramp; no sawtooth reset on replacement | Diverged |
| Non-stationary effects = stencil wear + ambient-temp diurnal (Section IV-A) | named exactly two | also diurnal on humidity, wear on peak-reflow-temp + time-above-liquidus | Diverged |
| Defect-labeling rule (how labels come from parameter values) | never described | calibrated-logit posterior `p(y\|x)`, with labels **sampled** (not argmax) so Bayes error is exact | Diverged (invented) |
| SPI auxiliary inspection feature (paste volume per aperture) (Section IV-A, Fig. 2 dashed edge) | described as inspection evidence, non-terminal | not generated (no `paste_volume` column) | Missing |
| Inference threshold τ = 0.60 (Table I, Eq. 4) | given | not a generator field; belongs to the inference layer | Missing (out of generator scope) |

## Status

Done: synthetic data generation + characterization (algorithm/pseudocode,
distribution histograms, Bayes error) and the multi-head MLP (paper baseline +
abstention variant, Optuna tuning, abstention analysis). Next: the symbolic/RCA
layers and MAUDE evidence linking (see `CLAUDE.md` phase plan).
