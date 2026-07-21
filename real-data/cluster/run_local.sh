#!/bin/bash
# Local fallback: run the whole MAUDE abstention sweep on ONE machine with N parallel
# training workers (no SLURM). The scrape runs once (sequential), then the o-sweep
# trains N at a time off the cache. Good for testing the matrix before submitting.
#
# Idempotent: an existing output is skipped so an interrupted run resumes. Set
# RCA_FORCE=1 to rebuild everything. Set SKIP_PREP=1 to reuse an existing cache.
#
# Usage:  bash real-data/cluster/run_local.sh [N_PARALLEL]      (default 4)
#         SKIP_PREP=1 bash real-data/cluster/run_local.sh 8
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source real-data/cluster/env.sh 2>/dev/null || true
python3 real-data/cluster/build_matrix.py
CACHE="${RCA_CACHE:-real-data/data/maude}"
mkdir -p "$CACHE" real-data/results/cluster real-data/cluster/logs
N="${1:-4}"
DEVICE="${RCA_DEVICE:-cpu}"
echo "local device: $DEVICE   cache: $CACHE"

echo "=== 1/2 scraping MAUDE (once) ==="
if [ -f "$CACHE/features.npz" ] && [ -z "${RCA_FORCE:-}" ]; then
  echo "[prep] cache exists, skip (RCA_FORCE=1 to rescrape)"
else
  python3 real-data/prep_dataset.py --cache "$CACHE" \
    --api-key "${OPENFDA_API_KEY:-}" \
    --max-per-class "${RCA_MAX_PER_CLASS:-3000}" \
    --page-size "${RCA_PAGE_SIZE:-100}" \
    --seed "${RCA_DATA_SEED:-7}"
fi

echo "=== 2/2 training $(grep -c . real-data/cluster/runs.tsv) models, $N at a time ==="
train_one() {
  local line="$1"
  local run_id args
  run_id=$(printf '%s' "$line" | cut -f1)
  args=$(printf '%s' "$line" | cut -f2-)
  local out
  out=$(printf '%s' "$args" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  if [ -n "$out" ] && [ -f "$out" ] && [ -z "${RCA_FORCE:-}" ]; then
    echo "[skip] $run_id (exists)"; return
  fi
  echo "[train] $run_id"
  # shellcheck disable=SC2086
  python3 real-data/train_from_cache.py $args --device "$DEVICE" \
    > "real-data/cluster/logs/${run_id}.log" 2>&1 \
    && echo "[done]  $run_id" || echo "[FAIL]  $run_id (see real-data/cluster/logs/${run_id}.log)"
}

while IFS= read -r line; do
  [ -z "$line" ] && continue
  train_one "$line" &
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 0.5; done
done < real-data/cluster/runs.tsv
wait

echo "=== done. metrics in real-data/results/cluster/*.json ==="
echo "analyse: python3 real-data/cluster/build_analysis_nb.py && open real-data/real_data_analysis.ipynb"
