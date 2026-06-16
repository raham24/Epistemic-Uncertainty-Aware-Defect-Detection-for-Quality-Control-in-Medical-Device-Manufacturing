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

The model ships in **two flavours that share one engine** (`mlp_common.py`):

- **`mlp_paper.py`** — the paper's exact model. Two softmax/cross-entropy
  classification heads (defect, per-stage mechanism) plus a sigmoid/BCE risk
  head. The safety-net baseline that should hit the paper's Table II numbers.
- **`mlp.py`** — our version. The selective-classification ("gambler")
  abstention loss on **both** classification heads, the sigmoid/BCE risk head
  unchanged. Each classification head gains an extra "abstain" output so the
  model can flag an ambiguous board/stage instead of guessing.

## Quickstart

```bash
conda env create -f environment.yml      # or: conda activate paper (if it exists)
conda activate paper

python generator.py        # -> data/smt_synthetic.csv   (200k records, deterministic)
python generator_v2.py     # -> data/smt_synthetic_v2.csv (our format; see "Generator v2" below)
python analysis.py         # -> figs/*.png, results/bayes_error.json

python mlp_paper.py        # train the paper baseline    -> ~0.95 defect accuracy
python mlp.py              # train our abstention model   (gambler loss on both heads)
python mlp.py --o 4.0      # raise the payoff o -> abstain less, predict more
python mlp.py --loss multihead   # plain-CE baseline inside the same file, for comparison
```

Everything runs on CPU (and Apple `mps`); no GPU required. All runs are seeded
and reproducible.

## Layout

```
domain/smt_paper.yaml   THE domain spec (single source of truth; the generator reads it)
generator.py            spec-driven, deterministic SMT synthetic generator (paper format)
generator_v2.py         our format: joint mechanism + per-mechanism risk, CLI-tunable balance/variance
analysis.py             class/label histograms + Bayes-error estimation

mlp_common.py           shared MLP engine: trunk, encoding, train loop, evaluate, CLI
mlp_paper.py            paper baseline   (CE defect + CE mechanism + BCE risk)
mlp.py                  our model        (gambler loss on both classification heads)

experiments.py          two-loss matrix: trains both losses, checks numerical stability
tune.py                 Optuna search (validation-only objective; leak-safe)
make_figs.py            comparison figures (risk-coverage, calibration, confusion, curves)
learning_curve.py       test performance vs training-set size (sample complexity)
feature_separability.py per-feature class histograms + overlap (separability sanity check)
overlap_check.py        marginal-overlap bin-stability + single-feature vs joint Bayes
prof_questions.py       abstention analysis: reject rate, train/test gap, who gets flagged

docs/algorithm.md       pseudocode, improvements over the original, reproducibility gaps
docs/results.md         team-facing results summary
docs/Generator_Overview.docx   high-level overview + paper comparison (generated)
docs/make_overview_docx.py     builder for the .docx (pulls live metrics)
figs/                   class_balance, mechanism_balance, parameter_hists,
                        class_conditional, bayes_summary, + model/abstention plots
results/                bayes_error.json, mlp_metrics.json, mlp_model.pt (checkpoint)
data/                   generated dataset (gitignored)
archive/                superseded original scratch generator (truncated stub)
```

## Model

Shared trunk `h = f(x)` (fully-connected, ReLU, dropout) feeding three head types,
exactly the paper's architecture:

| Head | Output | Paper loss (`mlp_paper.py`) | Our loss (`mlp.py`) |
|---|---|---|---|
| Defect (Eq. 1) | softmax over defect classes | cross-entropy | gambler `-log(o·p_y + r)` |
| Mechanism (Eq. 2) | one softmax per stage | cross-entropy | gambler `-log(o·p_y + r)` |
| Risk (Eq. 3) | one **independent sigmoid** per parameter | BCE w/ soft targets | BCE w/ soft targets (unchanged) |

Composite loss: `L = λ_d·loss(defect) + Σ_s λ_m·loss(mech_s) + λ_r·BCE(risk)`.
The risk head regresses the graded continuous target from the generator's Eq. 8
risk function (BCE with **soft** targets in (0,1)), not a binary in/out-of-spec
label.

In `mlp.py` the gambler term replaces cross-entropy on the two classification
heads. Each of those heads is one column wider (the abstain output), and `r` is
its abstain probability; `o` is the payoff (the `--o` flag). `r = 0` reduces the
term to cross-entropy + const, so larger `o` ⇒ predict more / abstain less. The
risk head is never widened or changed. The selective-classification form follows
Deep Gamblers (Liu et al., 2019); see `main.pdf`.

`mlp_common.MultiHeadMLP(abstain=...)` is the single switch: when `False` every
path is byte-identical to the plain model (so `mlp_paper.py` and
`mlp.py --loss multihead` both reproduce the baseline).

## Generator v2 (new version)

`generator_v2.py` reuses **all** of v1's maths (imports the sampling, drift,
deviations, defect scores, calibration, posterior, per-stage mechanism
assignment, and the Eq. 8 `graded_risk` straight from `generator.py`) and only
changes the output schema to a mechanism-centric, chained-prediction format:
`head1 defect → head2 mechanism → head3 parameter → head4 risk`.

| Head | Column(s) | What it is |
|---|---|---|
| head1 defect | `defect_label` | 3 classes (unchanged) |
| head2 mechanism | `mechanism_joint` | **Cartesian product** of the two stages, `"<printing>__<reflow>"` — 3×3 = 9 possible classes, **7 live** |
| (also kept) | `stage_printing_mechanism_label`, `stage_reflow_mechanism_label` | the per-stage labels, so an independent-head design stays available |
| head3 parameter | `risk_<param>` ×6 | per-parameter graded risk (Eq. 8) — the parameter-violation signal, identical to v1 |
| head4 risk | `risk_mech_<mechanism>` ×5 | per-mechanism risk incl. `no_mechanism`, from the same Eq. 8 aggregated through the causal map |

Steps 1–5 are byte-identical to v1, so features, posteriors, and `defect_label`
match `smt_synthetic.csv` row-for-row at the same seed; only the mechanism
representation and the added per-mechanism risk differ.

**Why the joint (Cartesian) mechanism head.** A defective board is usually
implicated at *both* stages (≈94% of mechanism-bearing boards have a printing
*and* a reflow mechanism — e.g. bridging = `aperture_overfill` + `reflow_spreading`).
Two independent per-stage softmax heads (the paper's structure, kept in `mlp_paper.py`)
assume the stages are conditionally independent given the features; a single
softmax over the 9 joint classes models both stages failing at once directly
(per Prof. Romanelli's suggestion). Only 7 of the 9 occur — the two cross-defect
combos (`aperture_overfill__non_coalescence`, `poor_paste_transfer__reflow_spreading`)
are structurally impossible because a board has one defect, so a 9-way head never
predicts them.

**Per-mechanism risk.** For each real mechanism, every causal edge that fires it
contributes `graded_risk()` of its parameter's bad-direction deviation; the
mechanism keeps the **max** over its edges. `no_mechanism` risk = `1 - max(real
mechanism risks)`.

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
python generator_v2.py --priors "no_defect=0.5,open_circuit=0.25,solder_bridging=0.25" --out data/smt_balanced.csv

# wider spread AND harder labels
python generator_v2.py --sigma-scale 1.5 --bayes-error 0.08 --out data/smt_hard.csv
```

Note: `--sigma` alone does **not** change difficulty — the logit gain is
re-calibrated to `target_bayes_error`, so the Bayes floor is held fixed
regardless of spread. `--sigma` changes the raw feature distributions,
out-of-spec rates, and risk targets; `--bayes-error` is the knob that actually
moves difficulty. Every run prints the effective priors, sigmas, and target
Bayes error up front.

## Key results

- Class balance matches paper Table I: 0.877 / 0.060 / 0.063.
- Bayes floor (oracle ceiling) ~95.5% (population) / ~95.3% (test); the trained
  MLP reaches ~95.1% defect accuracy, matching the paper's reported 95.00%.
- The abstention model concentrates rejections on the hard minority classes
  (open/bridge), confirming the selective-classification behaviour (`prof_questions.py`).
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
| N=200,000; 140k/30k/30k (Table I) | `n_records` 200000; train/val/test = 0.70/0.15/0.15 |
| class counts 175,420 / 12,569 / 12,011 (Table I) | calibrated to those exact fractions (IPF on logit offsets) |
| Fig. 2 causal topology (bridging via aperture_overfill + reflow_spreading; open = mirror) | all 12 `causal_edges`, mirror directions |
| multi-head MLP: softmax defect + per-stage mechanism, sigmoid risk (Eq. 1–3) | `mlp_common.MultiHeadMLP` (`defect_head`, `mech_heads`, `risk_head`) |

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
| MLP loss function | never stated | `L = λ_d·CE + Σ λ_m·CE + λ_r·BCE` (paper); gambler loss variant in `mlp.py` | Inferred |
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
