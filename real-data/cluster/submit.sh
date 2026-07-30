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

chunk="${RCA_MAX_ARRAY:-1000}"        # max tasks per array (Slurm array-size cap)
throttle="%${RCA_MAX_PARALLEL:-16}"
echo "concurrency: up to ${RCA_MAX_PARALLEL:-16} tasks per chunk (chunk size $chunk)"

# optional placement flags (single tokens, no spaces -> word-split on purpose).
# `extra` is shared (partition/account); QOS is applied PER-JOB below because the
# scrape and the training array need different QOS (see env.sh).
extra=""
[ -n "${RCA_PARTITION:-}" ] && extra="$extra --partition=$RCA_PARTITION"
[ -n "${RCA_ACCOUNT:-}" ]   && extra="$extra --account=$RCA_ACCOUNT"
prep_qos="";  [ -n "${RCA_PREP_QOS:-}" ]  && prep_qos="--qos=$RCA_PREP_QOS"
train_qos=""; [ -n "${RCA_TRAIN_QOS:-}" ] && train_qos="--qos=$RCA_TRAIN_QOS"

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
  prep_jid=$(sbatch --parsable $extra $prep_qos --time="${RCA_PREP_TIME:-12:00:00}" \
    real-data/cluster/prep.slurm)
  echo "submitted scrape: job $prep_jid, time ${RCA_PREP_TIME:-12:00:00} ${prep_qos:+($prep_qos)}"
  dep="--dependency=afterok:$prep_jid"
fi

# 2) train all models, after the scrape finishes OK (or immediately if SKIP_PREP).
# The matrix can exceed the Slurm array cap, so submit it in chunks of $chunk tasks:
# each chunk is an array 1..size carrying RCA_LINE_OFFSET so train.slurm reads the
# right slice of runs.tsv (line = offset + array index).
offset=0
chunk_idx=0
while [ "$offset" -lt "$n_runs" ]; do
  size=$(( n_runs - offset ))
  [ "$size" -gt "$chunk" ] && size=$chunk
  chunk_idx=$(( chunk_idx + 1 ))
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable $extra $train_qos --time="${RCA_TRAIN_TIME:-00:20:00}" \
    $dep --export=ALL,RCA_LINE_OFFSET=$offset \
    --array=1-"$size$throttle" real-data/cluster/train.slurm)
  echo "submitted training chunk $chunk_idx: job $jid  (runs.tsv lines $((offset+1))-$((offset+size)), time ${RCA_TRAIN_TIME:-00:20:00}) ${dep:+after prep}"
  offset=$(( offset + size ))
done

echo
echo "watch:   squeue -u \$USER"
echo "results: real-data/results/cluster/*.json"
echo "analyse: python3 real-data/cluster/build_analysis_nb.py && open real-data/real_data_analysis.ipynb"
