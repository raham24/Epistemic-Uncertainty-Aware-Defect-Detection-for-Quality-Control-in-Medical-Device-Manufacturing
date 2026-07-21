# MAUDE learned-abstention sweep

Real-data counterpart to the synthetic `cluster/` sweep. Instead of generating
synthetic dataset variants, it **scrapes MAUDE once** (the professor's redacted
TF-IDF pipeline, verbatim), caches the dataset + features, then trains a structured
sweep of **learned-abstention** MLPs off that cache -- above all a fine sweep over
the abstention payoff `o` -- and feeds a comparison notebook.

The model is the single-head PyTorch MLP from
`real-data/maude_product_problem_abstention_mlp.py`: same shape as the original
sklearn classifier (hidden `256,128`, ReLU, Adam), but its head is widened by one
"abstain" column and trained with the `-log(o*p_y + r)` term. Every run reports the
**learned reject** against the original **confidence-threshold baseline** at matched
coverage.

## Pipeline (two decoupled jobs)

| Step | Script | Runs |
|---|---|---|
| 1. Scrape + cache | `real-data/prep_dataset.py` (via `prep.slurm`) | ONCE (network + `OPENFDA_API_KEY`) |
| 2. Train sweep | `real-data/train_from_cache.py` (via `train.slurm`) | one task per config x seed |

`prep_dataset.py` writes the cache the whole sweep shares:
`data/maude/{dataset_all.csv, features.npz, text_pipeline.joblib, dataset_meta.json}`.

## What's in the sweep

A `BASE` config (`build_matrix.py`) plus one-factor studies. `BASE`: `o=2.0`,
hidden `256,128`, dropout 0, lr 1e-3, 80 epochs.

| study | varies | answers |
|---|---|---|
| `payoff` | `o` = 1.0-4.0 step 0.1 (31 values) | how the payoff trades coverage for selective accuracy; where the learned reject helps |
| `capacity` | trunk `--hidden` (4 sizes) | does a bigger net change the reject signal |
| `dropout` | `--dropout` in {0,.1,.2,.3} | regularization |
| `lr` | `--lr` in {3e-4,1e-3,3e-3} | optimization |

Each seed-independent config is trained across every seed in `RCA_SEEDS`
(default `0,1,2,3,4`). Distinct configs ~39, so ~195 runs at 5 seeds.

## Files

| File | Role |
|---|---|
| `build_matrix.py` | The matrix (single source of truth). Emits `runs.tsv`, `manifest.json`. |
| `env.sh` | Module loads + `conda activate paper` + scrape/placement knobs. |
| `prep.slurm` | Single job: scrape MAUDE + cache dataset/features. |
| `train.slurm` | Array: one abstention MLP per task (depends on `prep`). |
| `submit.sh` | Builds the matrix, submits prep + training array with the dependency. |
| `run_local.sh` | No-SLURM fallback (resumable). |
| `build_analysis_nb.py` | Emits `../real_data_analysis.ipynb`. |

## Run it

Put your key in `~/.bashrc` on the cluster (`export OPENFDA_API_KEY="..."`), then:

```bash
bash real-data/cluster/submit.sh        # scrape once -> train the whole sweep
squeue -u $USER                         # watch
```

Everything runs on **CPU** (the MLP is tiny; the SVD features are 300-dim). Once the
cache exists you can re-run just the training sweep:

```bash
SKIP_PREP=1 bash real-data/cluster/submit.sh
```

Locally (one machine, resumable):

```bash
bash real-data/cluster/run_local.sh 4       # 4 parallel workers
SKIP_PREP=1 bash real-data/cluster/run_local.sh 8
```

Then analyse:

```bash
python3 real-data/cluster/build_analysis_nb.py   # (re)build the notebook
jupyter lab                                       # open real-data/real_data_analysis.ipynb
```

## Outputs (gitignored)

- `real-data/data/maude/` -- the scraped dataset + cached features
- `real-data/results/cluster/<run_id>.json` -- per-run metrics (both selective curves)
- `real-data/cluster/{runs.tsv, manifest.json, logs/}` -- generated matrix + logs
- `figs/fig_maude_*.{png,pdf}` -- notebook figures

## Change the sweep

Edit `BASE` / `O_GRID` / `HIDDEN_GRID` / `DROPOUT_GRID` / `LR_GRID` at the top of
`build_matrix.py`, then re-run `submit.sh` (array sizes adapt to the new line count).

```bash
python3 real-data/cluster/build_matrix.py   # preview the matrix + per-study counts
```
