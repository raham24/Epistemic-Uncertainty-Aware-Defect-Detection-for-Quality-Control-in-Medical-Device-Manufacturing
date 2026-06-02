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

## Status

Done: synthetic data generation + characterization (the three deliverables —
algorithm/pseudocode, distribution histograms, Bayes error). Next: the model
stage (multi-head MLP + loss), then the symbolic/RCA layers and MAUDE evidence
linking (see `CLAUDE.md` phase plan).
