#!/bin/bash
# SLURM array: train one abstention MLP per task (line N of runs.tsv). Runs in
# parallel; depends on prep.slurm (submit.sh wires the dependency so training only
# starts once the cached dataset exists).
#SBATCH --job-name=maude_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=real-data/cluster/logs/train_%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source real-data/cluster/env.sh
mkdir -p real-data/results/cluster real-data/cluster/logs

# pick this task's line: run_id <TAB> train_from_cache args. The matrix can exceed the
# Slurm array cap, so submit.sh chunks it: each chunk is an array 1..N with a distinct
# RCA_LINE_OFFSET, and the true runs.tsv line is offset + array index.
lineno=$(( ${RCA_LINE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
line=$(sed -n "${lineno}p" real-data/cluster/runs.tsv)
run_id=$(printf '%s' "$line" | cut -f1)
args=$(printf '%s' "$line" | cut -f2-)

if [[ ! -f "$RCA_CACHE/features.npz" ]]; then
  echo "[train ${SLURM_ARRAY_TASK_ID}] MISSING cache $RCA_CACHE/features.npz -- run prep first" >&2
  exit 1
fi

echo "[train ${SLURM_ARRAY_TASK_ID}] $run_id   args: $args"
# shellcheck disable=SC2086
python real-data/train_from_cache.py $args --device "${RCA_DEVICE:-cpu}"
echo "[train ${SLURM_ARRAY_TASK_ID}] done $run_id"
