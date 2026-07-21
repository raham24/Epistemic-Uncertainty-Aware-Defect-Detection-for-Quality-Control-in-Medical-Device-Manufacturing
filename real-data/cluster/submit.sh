#!/bin/bash
# Submit the full MAUDE abstention sweep to SLURM: build the matrix, scrape the
# dataset ONCE, then train every model in parallel (training waits for the scrape
# via a dependency).
#
# Usage:  bash real-data/cluster/submit.sh
#         SKIP_PREP=1 bash real-data/cluster/submit.sh   # cache already exists
#
# Placement comes from real-data/cluster/env.sh (RCA_PARTITION / RCA_QOS / ...).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# shellcheck disable=SC1091
source real-data/cluster/env.sh

python real-data/cluster/build_matrix.py
mkdir -p "$RCA_CACHE" real-data/results/cluster real-data/cluster/logs

n_runs=$(grep -c . real-data/cluster/runs.tsv)
echo "matrix: 1 dataset (scrape), $n_runs training runs"

if [ "$n_runs" -gt "${RCA_MAX_ARRAY:-1000}" ]; then
  echo "ERROR: $n_runs runs exceeds RCA_MAX_ARRAY=${RCA_MAX_ARRAY:-1000}. Shrink the" \
       "sweep in build_matrix.py or raise RCA_MAX_ARRAY." >&2
  exit 1
fi
throttle="%${RCA_MAX_PARALLEL:-16}"
echo "concurrency: up to ${RCA_MAX_PARALLEL:-16} tasks at once"

# optional placement flags (single tokens, no spaces -> word-split on purpose)
extra=""
[ -n "${RCA_PARTITION:-}" ] && extra="$extra --partition=$RCA_PARTITION"
[ -n "${RCA_ACCOUNT:-}" ]   && extra="$extra --account=$RCA_ACCOUNT"
[ -n "${RCA_QOS:-}" ]       && extra="$extra --qos=$RCA_QOS"

# 1) scrape + cache the dataset (single job), unless SKIP_PREP=1
dep=""
if [ "${SKIP_PREP:-0}" = "1" ]; then
  if [[ ! -f "$RCA_CACHE/features.npz" ]]; then
    echo "ERROR: SKIP_PREP=1 but $RCA_CACHE/features.npz does not exist." >&2
    exit 1
  fi
  echo "skipping prep (SKIP_PREP=1); using existing cache $RCA_CACHE"
else
  # shellcheck disable=SC2086
  prep_jid=$(sbatch --parsable $extra --time="${RCA_PREP_TIME:-12:00:00}" \
    real-data/cluster/prep.slurm)
  echo "submitted scrape: job $prep_jid, time ${RCA_PREP_TIME:-12:00:00}"
  dep="--dependency=afterok:$prep_jid"
fi

# 2) train all models, after the scrape finishes OK (or immediately if SKIP_PREP)
# shellcheck disable=SC2086
train_jid=$(sbatch --parsable $extra --time="${RCA_TRAIN_TIME:-00:20:00}" \
  $dep --array=1-"$n_runs$throttle" real-data/cluster/train.slurm)
echo "submitted training: array job $train_jid (1-$n_runs$throttle), time ${RCA_TRAIN_TIME:-00:20:00} ${dep:+after prep}"

echo
echo "watch:   squeue -u \$USER"
echo "results: real-data/results/cluster/*.json"
echo "analyse: python3 real-data/cluster/build_analysis_nb.py && open real-data/real_data_analysis.ipynb"
