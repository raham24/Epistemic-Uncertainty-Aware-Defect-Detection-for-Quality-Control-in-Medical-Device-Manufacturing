#!/bin/bash
# Local fallback: run the whole sweep on ONE machine with N parallel training
# workers (no SLURM). Good for a beefy workstation or for testing the matrix
# before submitting to the cluster.
#
# Usage:  bash cluster/run_local.sh [N_PARALLEL]      (default 4)
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source cluster/env.sh 2>/dev/null || true
python cluster/build_matrix.py
mkdir -p data/cluster results/cluster cluster/logs
N="${1:-4}"
DEVICE="${RCA_DEVICE:-cpu}"

echo "=== 1/2 generating datasets (sequential) ==="
while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  echo "[gen] $name"
  # shellcheck disable=SC2086
  python generator.py $args --out "data/cluster/${name}.csv"
done < cluster/datasets.tsv

echo "=== 2/2 training $(grep -c . cluster/runs.tsv) models, $N at a time ==="
train_one() {
  local line="$1"
  local run_id dataset args
  run_id=$(printf '%s' "$line" | cut -f1)
  dataset=$(printf '%s' "$line" | cut -f2)
  args=$(printf '%s' "$line" | cut -f3-)
  echo "[train] $run_id"
  # shellcheck disable=SC2086
  python mlp.py --data "data/cluster/${dataset}.csv" $args --device "$DEVICE" \
    --model-out "results/cluster/${run_id}.pt" \
    --out "results/cluster/${run_id}.json" \
    > "cluster/logs/${run_id}.log" 2>&1 \
    && echo "[done]  $run_id" || echo "[FAIL]  $run_id (see cluster/logs/${run_id}.log)"
}

while IFS= read -r line; do
  [ -z "$line" ] && continue
  train_one "$line" &
  # throttle to N concurrent background jobs (portable: bash 3.2+)
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 0.5; done
done < cluster/runs.tsv
wait

echo "=== done. metrics in results/cluster/*.json -- open cluster_analysis.ipynb ==="
