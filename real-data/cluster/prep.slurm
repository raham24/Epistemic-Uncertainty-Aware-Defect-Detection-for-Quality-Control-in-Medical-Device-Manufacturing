#!/bin/bash
# SLURM single job: scrape MAUDE + build the cached dataset/features (runs ONCE).
# The training array depends on this via submit.sh, so it only starts once the
# cache exists. Needs network + (optionally) OPENFDA_API_KEY.
#SBATCH --job-name=maude_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=real-data/cluster/logs/prep_%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source real-data/cluster/env.sh
mkdir -p "$RCA_CACHE" real-data/cluster/logs

if [ -z "${OPENFDA_API_KEY:-}" ]; then
  echo "WARNING: OPENFDA_API_KEY not set; using anonymous openFDA access."
fi

echo "[prep] scraping MAUDE -> $RCA_CACHE  (seed=$RCA_DATA_SEED, max_records=$RCA_MAX_RECORDS)"
python real-data/prep_dataset.py \
  --cache "$RCA_CACHE" \
  --api-key "${OPENFDA_API_KEY:-}" \
  --max-records "$RCA_MAX_RECORDS" \
  --page-size "$RCA_PAGE_SIZE" \
  --seed "$RCA_DATA_SEED"
echo "[prep] cached dataset + features in $RCA_CACHE"
