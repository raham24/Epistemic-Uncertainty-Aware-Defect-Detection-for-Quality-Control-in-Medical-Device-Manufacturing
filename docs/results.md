# SMT Synthetic Data — Results Summary

Team-facing summary of the SMT replication baseline (the paper's surface-mount
assembly setup). Technical detail and pseudocode are in `docs/algorithm.md`.

## Executive summary

We generate a 200,000-record synthetic SMT dataset from a single spec file
(`domain/smt_paper.yaml`): 6 monitored process parameters, 2 process stages
(printing, reflow), 3 defect classes. The spec is aligned to the paper — the
causal map (Fig. 2), class priors (Table I), and graded risk (Eq. 8) all match.
Labels are sampled from a *known* posterior `p(y|x)` built from the Ishikawa
causal map, which lets us compute the **Bayes error in closed form**. The
labeling difficulty is calibrated so the **Bayes floor is ~4.7% (test) / 4.5%
(population)** — i.e. a best-possible (oracle) accuracy of ~95.3–95.5%. This makes
the paper's reported **95.00%** defect-head accuracy *achievable*: it sits just
below our oracle ceiling and just above a generic strong learner (gradient
boosting reaches 94.8% untuned). So a well-trained MLP reaching ~95% is realistic,
and the synthetic data tracks the paper's difficulty regime rather than an
arbitrary one.

Output is byte-for-byte reproducible given the seed, and every record carries
full provenance (generator version, seed, batch, timestamp, board id).

## Dataset at a glance

- 200,000 records, 100 batches × 2,000, batch-grouped 70/15/15 train/val/test (no leakage)
- 6 parameters: humidity, ambient temp, paste viscosity, stencil thickness,
  time-above-liquidus, peak reflow temp
- 3 defects: `no_defect`, `open_circuit`, `solder_bridging`
- 2 stages, each with its own mechanism vocabulary

## 1. Class / label distribution  →  `figs/class_balance.png`, `figs/mechanism_balance.png`

| defect | realized fraction (count) | paper Table I (count) |
|---|---|---|
| no_defect | 0.8771 (175,410) | 0.8771 (175,420) |
| open_circuit | 0.0602 (12,043) | 0.0601 (12,011) |
| solder_bridging | 0.0627 (12,547) | 0.0628 (12,569) |

All three classes are calibrated to the paper's exact Table I priors (not just
the no-defect rate); realized counts match to sampling noise. Per-stage
mechanisms are ~88% `no_mechanism`, mirroring the defect rates.

## 2. Which parameters carry signal  →  `figs/parameter_hists.png`, `figs/class_conditional.png`

Out-of-spec rates (driven by spec width relative to process sigma):

| parameter | out-of-spec | causal role (paper Fig. 2) |
|---|---|---|
| ambient_relative_humidity | 1.59% | high → bridging, low → open |
| ambient_temperature | 5.90% | high → bridging, low → open |
| paste_viscosity | 4.58% | low → bridging, high → open |
| stencil_thickness | 1.83% | high → bridging, low → open |
| time_above_liquidus | 0.00% | high → bridging, low → open |
| peak_reflow_temperature | 4.90% | high → bridging, low → open |

All six parameters are causal, each with opposite non-compliance directions for
the two defects (the paper's didactic point, Fig. 2). The class-conditional plot
shows clear separation for every parameter. `time_above_liquidus` has a very
capable spec (±5σ) so it rarely violates, but still shifts by class.

## 3. How hard is the problem — defect-head metrics  →  `figs/bayes_summary.png`, `figs/defect_metrics.png`

The defect classifier is evaluated against the paper's Section V-A numbers
(accuracy + weighted-F1), with the Bayes-optimal classifier as the achievable
ceiling. The learner is a basic single-head MLP — the same model family as the
paper.

| classifier | accuracy | weighted-F1 | macro-F1 |
|---|---|---|---|
| Ours — basic MLP | 0.9524 | 0.9518 | 0.8654 |
| Bayes-optimal ceiling | 0.9534 | 0.9524 | 0.8657 |
| **Paper (reported)** | **0.9500** | **0.9536** | — |
| Majority baseline (test split) | 0.8657 | — | — |

Per-class F1 (MLP / Bayes-optimal): no_defect 0.973 / 0.973, open_circuit
0.794 / 0.795, solder_bridging 0.829 / 0.829. The irreducible Bayes floor is
**0.0473 (test) / 0.0450 (population)** — an oracle accuracy ceiling of ~95.3–95.5%.

**Read it as:** the basic MLP reaches the Bayes-optimal ceiling almost exactly
(macro-F1 0.8654 vs 0.8657; per-class F1 within 0.0003) — so an MLP can train to
optimality on this data, validating the pipeline. Its accuracy (95.24%) brackets
the paper's **95.00%** and its weighted-F1 (95.18%) sits just under the paper's
**95.36%**. The remaining ~0.2 pp F1 gap is because our minority defects (open /
bridging) are marginally harder; since the MLP is already at the ceiling, closing
it means making those two classes slightly more separable in the generator, not
better modeling. Because the posterior is known, the Bayes floor is exact, and
both learners respect it.

## Reproduce

```bash
conda activate paper
python generator.py            # -> data/smt_synthetic.csv (deterministic)
python analysis.py             # -> figs/*.png, results/bayes_error.json
```

## Files

- `domain/smt_paper.yaml` — the domain spec (single source of truth)
- `generator.py` — spec-driven, deterministic generator
- `analysis.py` — histograms + Bayes error
- `docs/algorithm.md` — pseudocode, improvements, reproducibility gaps
- `results/bayes_error.json` — all computed metrics
