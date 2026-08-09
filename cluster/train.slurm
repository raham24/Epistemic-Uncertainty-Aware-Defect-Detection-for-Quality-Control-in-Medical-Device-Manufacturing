#!/bin/bash
# SLURM array: train one model per task (line N of runs.tsv). Runs in parallel
# across the array. Depends on the gen_data array (submit.sh wires the dependency
# so training only starts once every dataset exists).
#SBATCH --job-name=rca_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/train_%A_%a.out
# GPU is opt-in via cluster/env.sh: set RCA_GPUS=1 (submit.sh adds --gres=gpu:1)
# and RCA_DEVICE=cuda. These models are tiny, so CPU is the sensible default.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source cluster/env.sh
mkdir -p results/cluster cluster/logs

# pick this task's line: run_id <TAB> dataset <TAB> mlp args. With chunked submission the
# array is 1..chunk and submit.sh passes RCA_LINE_OFFSET, so the true line = offset + index.
lineno=$(( ${RCA_LINE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
line=$(sed -n "${lineno}p" cluster/runs.tsv)
run_id=$(printf '%s' "$line" | cut -f1)
dataset=$(printf '%s' "$line" | cut -f2)
args=$(printf '%s' "$line" | cut -f3-)
csv="data/cluster/${dataset}.csv"

if [[ ! -f "$csv" ]]; then
  echo "[train ${SLURM_ARRAY_TASK_ID}] MISSING dataset $csv -- run gen_data first" >&2
  exit 1
fi

# already trained? skip, so re-submitting only fills in NEW runs (e.g. extra seeds).
if [[ "${RCA_FORCE:-0}" != "1" && -f "results/cluster/${run_id}.json" && -f "results/cluster/${run_id}.pt" ]]; then
  echo "[train ${SLURM_ARRAY_TASK_ID}] $run_id already done -- skip (RCA_FORCE=1 to retrain)"
  exit 0
fi

echo "[train ${SLURM_ARRAY_TASK_ID}] $run_id   dataset=$dataset   args: $args"
# shellcheck disable=SC2086
python mlp.py --data "$csv" $args --device "${RCA_DEVICE:-cpu}" \
  --model-out "results/cluster/${run_id}.pt" \
  --out "results/cluster/${run_id}.json"
echo "[train ${SLURM_ARRAY_TASK_ID}] saved results/cluster/${run_id}.{pt,json}"
