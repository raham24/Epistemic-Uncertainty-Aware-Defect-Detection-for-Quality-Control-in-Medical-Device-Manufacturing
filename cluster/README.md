# Cluster sweep

Generates several dataset variants (different class balances + difficulties),
trains every model on them in parallel for **both** loss functions
(`cascade`, `abstention`), saves the checkpoints, and feeds a comparison notebook
that finds the optimal configs.

## Files

| File | Role |
|---|---|
| `build_matrix.py` | **The matrix** (single source of truth). Emits `datasets.tsv`, `runs.tsv`, `manifest.json`. Edit `DATASETS` / `LOSSES` / `O_VALUES` / `SEEDS` here. |
| `env.sh` | **Edit this for your cluster** — module loads + `conda activate paper`. Sourced by every script. |
| `gen_data.slurm` | SLURM array: one dataset variant per task. |
| `train.slurm` | SLURM array: one model per task (runs in parallel; depends on `gen_data`). |
| `submit.sh` | Builds the matrix and submits both arrays with the right sizes + dependency. |
| `run_local.sh` | No-SLURM fallback: runs the whole sweep on one machine, `N` workers in parallel. |
| `build_analysis_nb.py` | Emits `../cluster_analysis.ipynb`. |

## Run it

On a SLURM cluster (after editing `env.sh`):

```bash
bash cluster/submit.sh        # build matrix -> generate datasets -> train all models
squeue -u $USER               # watch
```

Locally (one machine, 4 parallel workers):

```bash
bash cluster/run_local.sh 4
```

Then analyse:

```bash
python cluster/build_analysis_nb.py   # (re)build the notebook
jupyter lab                           # from the repo ROOT; open cluster_analysis.ipynb
```

## Outputs (gitignored)

- `data/cluster/<name>.csv` — the generated dataset variants
- `results/cluster/<run_id>.pt` + `.json` — checkpoints + metrics
- `cluster/{datasets,runs}.tsv`, `cluster/manifest.json`, `cluster/logs/` — generated matrix + logs

## Change the sweep

Edit `build_matrix.py` and re-run `bash cluster/submit.sh` (or `run_local.sh`).
The array sizes adapt automatically to the new line counts.
