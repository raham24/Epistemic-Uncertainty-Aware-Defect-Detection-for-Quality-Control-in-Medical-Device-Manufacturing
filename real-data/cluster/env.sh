# Sourced by every real-data cluster script (prep.slurm, train.slurm, run_local.sh).
# Batch jobs start in a clean shell and do NOT source ~/.bashrc, so we load modules
# and activate conda HERE. Uses the project's `paper` env (same as the synthetic
# cluster/env.sh) -- the MAUDE scraper's one extra dep (`requests`) is in
# environment.yml, so `conda env update -f environment.yml` brings `paper` current.
#
# One-time setup on a login node (upgrade the existing env with the new deps):
#   module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda env update -f environment.yml     # adds requests to the `paper` env
#   conda activate paper

module load gcc                                   2>/dev/null || true
module load slurm                                 2>/dev/null || true
module load git                                   2>/dev/null || true
module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${RCA_CONDA_ENV:-paper}"

# run from the repo root so real-data/, data/, results/ resolve correctly
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# These models are tiny (single MLP on 300-dim SVD features), so CPU is the default.
export RCA_DEVICE="${RCA_DEVICE:-cpu}"

# OPENFDA_API_KEY is inherited from your login environment (set in ~/.bashrc) and
# passed through --export=ALL. Only the prep job needs it; anonymous access works
# too (lower rate limit). Do NOT hardcode a key here (this file is tracked).

# --- MAUDE scrape knobs (prep.slurm) ---
export RCA_MAX_PER_CLASS="${RCA_MAX_PER_CLASS:-3000}"
export RCA_PAGE_SIZE="${RCA_PAGE_SIZE:-100}"
export RCA_DATA_SEED="${RCA_DATA_SEED:-7}"          # split seed for the cached dataset
export RCA_CACHE="${RCA_CACHE:-real-data/data/maude}"

# --- SLURM placement (leave EMPTY to omit the flag; MAUDE download is CPU-only) ---
export RCA_PARTITION="${RCA_PARTITION:-}"
export RCA_ACCOUNT="${RCA_ACCOUNT:-}"
# QOS is set SEPARATELY for the two jobs. The scrape is a single long job (12h) ->
# needs `long`. The training array is ~195 SHORT tasks, and `long` has a tiny
# per-account submit cap (MaxSubmitJobsPerAccount) meant for a few 7-day jobs, so
# submitting the whole array under it fails. Training therefore uses the DEFAULT
# QOS (empty), which has far more submit headroom and a 1h wall (tasks are minutes).
export RCA_PREP_QOS="${RCA_PREP_QOS:-long}"        # single 12h scrape job
export RCA_TRAIN_QOS="${RCA_TRAIN_QOS:-}"          # empty -> default QOS (normal, 1h)

# --- per-job walltime (submit.sh passes as --time). The scrape hits a rate-limited
# API so give it hours; each train run is minutes. ---
export RCA_PREP_TIME="${RCA_PREP_TIME:-12:00:00}"
export RCA_TRAIN_TIME="${RCA_TRAIN_TIME:-00:20:00}"

# --- array throttle + guard ---
export RCA_MAX_PARALLEL="${RCA_MAX_PARALLEL:-16}"
export RCA_MAX_ARRAY="${RCA_MAX_ARRAY:-1000}"

# --- sweep shape: seeds every configuration is trained at (build_matrix.py reads this) ---
export RCA_SEEDS="${RCA_SEEDS:-0,1,2,3,4}"
