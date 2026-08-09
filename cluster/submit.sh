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

# Training is submitted in CHUNKS of RCA_MAX_ARRAY tasks so the total can exceed the
# account's MaxSubmitJobsPerAccount cap. Chunk size MUST be <= that cap (default 500,
# safe under Star HPC's 512 on the normal QOS); each chunk carries RCA_LINE_OFFSET.
chunk="${RCA_MAX_ARRAY:-500}"
# %K caps how many array tasks run concurrently (set via RCA_MAX_PARALLEL in env.sh)
throttle="%${RCA_MAX_PARALLEL:-16}"
echo "concurrency: up to ${RCA_MAX_PARALLEL:-16} tasks at once"

# optional placement flags (single tokens, no spaces -> word-split on purpose).
# empty RCA_* -> nothing added -> cluster default partition, no account, no GPU.
extra=""
[ -n "${RCA_PARTITION:-}" ] && extra="$extra --partition=$RCA_PARTITION"
[ -n "${RCA_ACCOUNT:-}" ]   && extra="$extra --account=$RCA_ACCOUNT"
[ -n "${RCA_QOS:-}" ]       && extra="$extra --qos=$RCA_QOS"
gpu=""
[ -n "${RCA_GPUS:-}" ] && gpu="--gres=gpu:$RCA_GPUS"
[ -n "$extra" ] && echo "placement:$extra ${gpu:+(train $gpu)}"

# 1) generate datasets (array over datasets.tsv)
# shellcheck disable=SC2086
gen_jid=$(sbatch --parsable $extra --time="${RCA_GEN_TIME:-00:10:00}" \
  --array=1-"$n_ds$throttle" cluster/gen_data.slurm)
echo "submitted dataset generation: array job $gen_jid (1-$n_ds$throttle), time ${RCA_GEN_TIME:-00:10:00}"

# 2) train all models in chunks (each waits for gen_data). Every array task counts against
# MaxSubmitJobsPerAccount, so if a chunk is rejected because the queue is full we WAIT and
# retry -- self-throttling to the cap without knowing its value. train.slurm reads line
# (RCA_LINE_OFFSET + array index) of runs.tsv and skips runs already done, so adding seeds
# only trains the new ones. Run under tmux/nohup: draining between chunks can take a while.
retry_sleep="${RCA_SUBMIT_RETRY_SLEEP:-120}"
max_retries="${RCA_SUBMIT_MAX_RETRIES:-60}"     # give up after this many waits per chunk
errf="$(mktemp)"; trap 'rm -f "$errf"' EXIT
offset=0; chunk_idx=0
n_chunks=$(( (n_runs + chunk - 1) / chunk ))
while [ "$offset" -lt "$n_runs" ]; do
  size=$(( n_runs - offset )); [ "$size" -gt "$chunk" ] && size=$chunk
  chunk_idx=$(( chunk_idx + 1 )); tries=0
  while true; do
    # shellcheck disable=SC2086
    if jid=$(sbatch --parsable $extra $gpu --time="${RCA_TRAIN_TIME:-00:20:00}" \
        --dependency=afterok:"$gen_jid" --export=ALL,RCA_LINE_OFFSET=$offset \
        --array=1-"$size$throttle" cluster/train.slurm 2>"$errf"); then
      echo "submitted training chunk $chunk_idx/$n_chunks: job $jid  (runs.tsv lines $((offset+1))-$((offset+size))) after $gen_jid"
      break
    fi
    if ! grep -qi 'MaxSubmitJobs\|QOSMaxSubmitJobs\|job submit limit' "$errf"; then
      echo "sbatch failed for chunk $chunk_idx:" >&2; cat "$errf" >&2; exit 1
    fi
    if [ "$chunk_idx" -eq 1 ] && [ "$tries" -eq 0 ]; then
      echo "note: first chunk rejected on MaxSubmitJobs. If the queue is otherwise empty," \
           "your chunk size (RCA_MAX_ARRAY=$chunk) is larger than the account cap -- lower it." >&2
    fi
    tries=$(( tries + 1 ))
    if [ "$tries" -gt "$max_retries" ]; then
      echo "chunk $chunk_idx: still rejected after $max_retries waits. Lower RCA_MAX_ARRAY" \
           "below MaxSubmitJobsPerAccount, or raise RCA_SUBMIT_MAX_RETRIES." >&2
      exit 1
    fi
    echo "chunk $chunk_idx/$n_chunks: queue full (MaxSubmitJobs), try $tries/$max_retries -- waiting ${retry_sleep}s..."
    sleep "$retry_sleep"
  done
  offset=$(( offset + size ))
done

echo
echo "watch:   squeue -u \$USER"
echo "results: results/cluster/*.json (metrics) + *.pt (checkpoints)"
echo "analyse: open cluster_analysis.ipynb once training finishes"
