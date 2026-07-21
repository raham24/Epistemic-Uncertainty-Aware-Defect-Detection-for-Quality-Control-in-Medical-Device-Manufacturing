# Cluster sweep

Generates several dataset variants (different class balances + difficulties),
trains a structured sweep of models on them in parallel for **both** loss
functions (`cascade`, `abstention`), saves every checkpoint, and feeds a
comparison notebook that finds the optimal configs.

## What's in the sweep

A **core grid** plus focused one-factor-at-a-time **studies** around a single base
config (`BASE` in `build_matrix.py`). Shared center points are de-duplicated, so a
config that several studies want is defined once and tagged with each. Current
matrix: **10 datasets, 132 distinct configurations**, each trained across every seed
in `RCA_SEEDS` (default `0,1,2,3,4`) → **132 × 5 = 660 runs**. Set `RCA_SEEDS` to
change the seed count (e.g. `RCA_SEEDS=0,1,2` → 396 runs); keep the total under
`RCA_MAX_ARRAY`. The bulk is the fine-grained payoff o-sweep below.

| study | varies | on | answers |
|---|---|---|---|
| `core` | loss x dataset | all 10 datasets | cascade vs abstention vs the Bayes ceiling, everywhere |
| `payoff` | abstention `o` = 1.0–4.0 step 0.1 (31 values) | baseline / hard / imbalanced | how the payoff trades coverage for selective accuracy |
| `capacity` | trunk `--hidden` (5 sizes) | baseline / hard | does a bigger trunk close the gap to Bayes |
| `dropout` | `--dropout` ∈ {0,.1,.2,.3} | baseline / hard | best regularization |
| `lr` | `--lr` ∈ {3e-4,1e-3,3e-3} | baseline / hard | best Adam step |
| `class_weight` | `--class-weight` ∈ {none,sqrt,inverse} | imbalanced / baseline | minority recall vs accuracy |

Every run in `runs.tsv` lists its **full** mlp.py CLI (all hyperparameters
explicit), so a run is reproducible regardless of mlp.py's current defaults.

## Files

| File | Role |
|---|---|
| `build_matrix.py` | **The matrix** (single source of truth). Emits `datasets.tsv`, `runs.tsv`, `manifest.json`. Edit `DATASETS` / `BASE` / the study grids here. |
| `env.sh` | **Edit this for your cluster** — module loads + `conda activate paper` + placement/throttle knobs. Sourced by every script. |
| `gen_data.slurm` | SLURM array: one dataset variant per task. |
| `train.slurm` | SLURM array: one model per task (runs in parallel; depends on `gen_data`). |
| `submit.sh` | Builds the matrix and submits both arrays with the right sizes, concurrency cap, and dependency. |
| `run_local.sh` | No-SLURM fallback: runs the whole sweep on one machine, `N` workers in parallel. **Resumable** (skips finished runs). |
| `build_analysis_nb.py` | Emits `../cluster_analysis.ipynb`. |
| `tunnel.slurm` | Optional: run a notebook on a compute node from VS Code (separate from the sweep). |

## Run it

On a SLURM cluster (after editing `env.sh`):

```bash
bash cluster/submit.sh        # build matrix -> generate datasets -> train all models
squeue -u $USER               # watch
```

Training runs **on GPU by default** (`RCA_DEVICE=cuda`, one GPU per task via
`RCA_GPUS=1`); dataset generation stays on CPU (it's pure numpy). Pin a GPU type if
your cluster requires it (`RCA_GPUS=A30:1`), or set `RCA_GPUS=""` for an all-CPU sweep.

`submit.sh` caps concurrent array tasks at `RCA_MAX_PARALLEL` (default 16) and
refuses to submit if the matrix is larger than `RCA_MAX_ARRAY` (default 1000, your
cluster's `MaxArraySize`). On GPU, set `RCA_MAX_PARALLEL` to the number of GPUs you
can use at once (Slurm queues the excess regardless).

Locally (one machine; resumable — re-run to continue after an interrupt):

```bash
bash cluster/run_local.sh 4       # 4 parallel workers; finished runs are skipped
RCA_FORCE=1 bash cluster/run_local.sh 8   # ignore existing outputs, rebuild all
```

Then analyse:

```bash
python3 cluster/build_analysis_nb.py   # (re)build the notebook
jupyter lab                            # from the repo ROOT; open cluster_analysis.ipynb
```

## Outputs (gitignored)

- `data/cluster/<name>.csv` — the generated dataset variants
- `results/cluster/<run_id>.pt` + `.json` — checkpoints + metrics
- `cluster/{datasets,runs}.tsv`, `cluster/manifest.json`, `cluster/logs/` — generated matrix + logs

## Change the sweep

Edit the config blocks at the top of `build_matrix.py` (`DATASETS`, `BASE`, the
`PROBE_*` lists, and the `*_GRID` axes), then re-run `bash cluster/submit.sh` (or
`run_local.sh`). The array sizes adapt automatically to the new line counts.
```bash
python3 cluster/build_matrix.py   # preview the matrix + per-study run counts
```
