#!/bin/bash
# Submit the full sweep to SLURM: build the matrix, generate every dataset, then
# train every model in parallel (training waits for all datasets via a dependency).
#
# Usage:  bash cluster/submit.sh
#
# Partition/account/GPU come from cluster/env.sh (RCA_PARTITION / RCA_ACCOUNT /
# RCA_GPUS). Left empty there -> the cluster's DEFAULT partition is used, matching
# Star HPC's own example sbatch (which omits --partition).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# load modules + conda (so python is the `paper` env) + the placement vars
# shellcheck disable=SC1091
source cluster/env.sh

python cluster/build_matrix.py
mkdir -p data/cluster results/cluster cluster/logs

n_ds=$(grep -c . cluster/datasets.tsv)
n_runs=$(grep -c . cluster/runs.tsv)
echo "matrix: $n_ds datasets, $n_runs training runs"

# optional placement flags (single tokens, no spaces -> word-split on purpose).
# empty RCA_* -> nothing added -> cluster default partition, no account, no GPU.
extra=""
[ -n "${RCA_PARTITION:-}" ] && extra="$extra --partition=$RCA_PARTITION"
[ -n "${RCA_ACCOUNT:-}" ]   && extra="$extra --account=$RCA_ACCOUNT"
gpu=""
[ -n "${RCA_GPUS:-}" ] && gpu="--gres=gpu:$RCA_GPUS"
[ -n "$extra" ] && echo "placement:$extra ${gpu:+(train $gpu)}"

# 1) generate datasets (array over datasets.tsv)
# shellcheck disable=SC2086
gen_jid=$(sbatch --parsable $extra --array=1-"$n_ds" cluster/gen_data.slurm)
echo "submitted dataset generation: array job $gen_jid (1-$n_ds)"

# 2) train all models, but only after every dataset finishes OK
# shellcheck disable=SC2086
train_jid=$(sbatch --parsable $extra $gpu --dependency=afterok:"$gen_jid" \
  --array=1-"$n_runs" cluster/train.slurm)
echo "submitted training: array job $train_jid (1-$n_runs), after $gen_jid"

echo
echo "watch:   squeue -u \$USER"
echo "results: results/cluster/*.json (metrics) + *.pt (checkpoints)"
echo "analyse: open cluster_analysis.ipynb once training finishes"
