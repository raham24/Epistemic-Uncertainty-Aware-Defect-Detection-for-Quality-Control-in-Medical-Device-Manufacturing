# Medical Device Neurosymbolic RCA — SMT Replication Baseline

Replication of the synthetic-data and difficulty-characterization stages of
Shenoy & Ameri (2026), *"Uncertainty-Aware Neurosymbolic Root-Cause Analysis for
Surface-Mount Assembly,"* IEEE Trans. Semiconductor Manufacturing
(DOI 10.1109/TSM.2026.3673999). This is the safety-net SMT baseline that the
medical-device port (external insulin pumps) builds on later — see `CLAUDE.md`
for the full project plan.

## What's here

A spec-driven synthetic SMT data generator, distribution histograms, and exact
Bayes-error analysis. The dataset and labels are aligned to the paper: parameters,
defect classes, mechanisms, the Fig. 2 causal map, the Table I class balance, and
the Eq. (8) graded risk. Values the paper does not release are reconstructed and
calibrated to its reported anchors (see `docs/algorithm.md` reproducibility gaps).

## Quickstart

```bash
conda env create -f environment.yml      # or: conda activate paper (if it exists)
conda activate paper

python generator.py        # -> data/smt_synthetic.csv   (200k records, deterministic)
python analysis.py         # -> figs/*.png, results/bayes_error.json
python docs/make_overview_docx.py   # -> docs/Generator_Overview.docx
```

## Layout

```
domain/smt_paper.yaml   THE domain spec (single source of truth; the generator reads it)
generator.py            spec-driven, deterministic SMT synthetic generator
analysis.py             class/label histograms + Bayes-error estimation
docs/algorithm.md       pseudocode, improvements over the original, reproducibility gaps
docs/results.md         team-facing results summary
docs/Generator_Overview.docx   high-level overview + paper comparison (generated)
docs/make_overview_docx.py     builder for the .docx (pulls live metrics)
figs/                   class_balance, mechanism_balance, parameter_hists,
                        class_conditional, bayes_summary
results/bayes_error.json   all computed metrics
data/                   generated dataset (gitignored)
archive/                superseded original scratch generator (truncated stub)
```

## Key results

- Class balance matches paper Table I: 0.877 / 0.060 / 0.063.
- Bayes floor (oracle ceiling) ~95.5% (population) / ~95.3% (test); a strong
  learner reaches ~94.8%, so the paper's reported 95.00% defect-head accuracy is
  achievable. ~95% is the target for the trained MLP.
- Output is byte-for-byte reproducible given the seed.

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
| batch/shift grouped split; normalize on train only (Section IV-C) | `_assign_split`; `StandardScaler.fit(Xtr)` |
| N=200,000; 140k/30k/30k (Table I) | `n_records` 200000; train/val/test = 0.70/0.15/0.15 |
| class counts 175,420 / 12,569 / 12,011 (Table I) | calibrated to those exact fractions (IPF on logit offsets) |
| Fig. 2 causal topology (bridging via aperture_overfill + reflow_spreading; open = mirror) | all 12 `causal_edges`, mirror directions |

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
| Stencil-thickness wear "with periodic replacement" (Section IV-A) | described | single monotonic linear ramp; no sawtooth reset on replacement | Diverged |
| Non-stationary effects = stencil wear + ambient-temp diurnal (Section IV-A) | named exactly two | also diurnal on humidity, wear on peak-reflow-temp + time-above-liquidus | Diverged |
| Defect-labeling rule (how labels come from parameter values) | never described | calibrated-logit posterior `p(y\|x)`, with labels **sampled** (not argmax) so Bayes error is exact | Diverged (invented) |
| SPI auxiliary inspection feature (paste volume per aperture) (Section IV-A, Fig. 2 dashed edge) | described as inspection evidence, non-terminal | not generated (no `paste_volume` column) | Missing |
| Inference threshold τ = 0.60 (Table I, Eq. 4) | given | not a generator field; belongs to the inference layer (not built yet) | Missing (out of generator scope) |

## Status

Done: synthetic data generation + characterization (the three deliverables —
algorithm/pseudocode, distribution histograms, Bayes error). Next: the model
stage (multi-head MLP + loss), then the symbolic/RCA layers and MAUDE evidence
linking (see `CLAUDE.md` phase plan).
