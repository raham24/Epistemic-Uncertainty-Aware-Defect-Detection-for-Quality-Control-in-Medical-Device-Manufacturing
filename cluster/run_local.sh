#!/bin/bash
# Local fallback: run the whole sweep on ONE machine with N parallel training
# workers (no SLURM). Good for a beefy workstation or for testing the matrix
# before submitting to the cluster.
#
# Idempotent: an existing output is skipped so an interrupted run resumes where it
# stopped. Set RCA_FORCE=1 to regenerate/retrain everything from scratch.
#
# Usage:  bash cluster/run_local.sh [N_PARALLEL]      (default 4)
#         RCA_FORCE=1 bash cluster/run_local.sh 8     (rebuild everything)
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source cluster/env.sh 2>/dev/null || true
python3 cluster/build_matrix.py
mkdir -p data/cluster results/cluster cluster/logs
N="${1:-4}"
# resolve device: honor RCA_DEVICE, but if cuda was requested and isn't available on
# THIS machine (e.g. a Mac), fall back to the Apple GPU (mps) or cpu so the local
# runner always works regardless of env.sh's cluster-oriented default.
DEVICE="${RCA_DEVICE:-cpu}"
if [ "$DEVICE" = "cuda" ]; then
  DEVICE=$(python3 - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None)
print("cuda" if torch.cuda.is_available()
      else ("mps" if mps and mps.is_available() else "cpu"))
PY
)
fi
echo "local device: $DEVICE"

echo "=== 1/2 generating datasets (sequential) ==="
while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  csv="data/cluster/${name}.csv"
  if [ -f "$csv" ] && [ -z "${RCA_FORCE:-}" ]; then
    echo "[gen] $name (exists, skip)"; continue
  fi
  echo "[gen] $name"
  # shellcheck disable=SC2086
  python3 generator.py $args --out "$csv"
done < cluster/datasets.tsv

echo "=== 2/2 training $(grep -c . cluster/runs.tsv) models, $N at a time ==="
train_one() {
  local line="$1"
  local run_id dataset args
  run_id=$(printf '%s' "$line" | cut -f1)
  dataset=$(printf '%s' "$line" | cut -f2)
  args=$(printf '%s' "$line" | cut -f3-)
  if [ -f "results/cluster/${run_id}.json" ] && [ -z "${RCA_FORCE:-}" ]; then
    echo "[skip] $run_id (exists)"; return
  fi
  echo "[train] $run_id"
  # shellcheck disable=SC2086
  python3 mlp.py --data "data/cluster/${dataset}.csv" $args --device "$DEVICE" \
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
