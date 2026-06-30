#!/bin/bash
# SLURM array: generate one dataset variant per task (line N of datasets.tsv).
# The array range is set by submit.sh (`--array=1-<n_datasets>`); submit through
# cluster/submit.sh so it matches the matrix automatically.
#SBATCH --job-name=rca_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/gen_%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source cluster/env.sh
mkdir -p data/cluster cluster/logs

# pick this task's line: name <TAB> generator args
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/datasets.tsv)
name=$(printf '%s' "$line" | cut -f1)
args=$(printf '%s' "$line" | cut -f2-)

echo "[gen ${SLURM_ARRAY_TASK_ID}] $name   args: ${args:-<defaults>}"
# shellcheck disable=SC2086
python generator.py $args --out "data/cluster/${name}.csv"
echo "[gen ${SLURM_ARRAY_TASK_ID}] wrote data/cluster/${name}.csv"
