# Sourced by every cluster script (gen_data.slurm, train.slurm, run_local.sh).
# Tuned for Raham's SLURM cluster using the module set from the prof's ~/.bashrc.
# Batch jobs start in a clean shell and do NOT source ~/.bashrc, so we load the
# modules and activate the conda env HERE, inside the job.
#
# One-time setup (do this once on a login node before the first submit):
#   module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda env create -f environment.yml      # creates the `paper` env
#
# If a module version changes on the cluster, update the names below.

# --- modules (from the prof's .bashrc; htop/nvtop are interactive-only, omitted) ---
module load gcc                                   2>/dev/null || true
module load slurm                                 2>/dev/null || true
module load git                                   2>/dev/null || true

# miniconda + make `conda` usable in this (possibly non-interactive) shell.
module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
# Source conda's shell hook so `conda activate` works in batch jobs. Resolve the
# base prefix with `conda info --base` -- robust whether conda is a binary on PATH
# or already a shell function. (Do NOT use `which conda`: when conda is a function
# it returns a bare name and the path breaks.)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

# --- activate the project env (create it once with environment.yml; see header) ---
conda activate paper

# --- run from the repo root so data/, domain/, results/ resolve correctly ---
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# device for mlp_v2.py: cpu (default) or cuda on a GPU node (set RCA_DEVICE=cuda)
export RCA_DEVICE="${RCA_DEVICE:-cpu}"

# --- optional SLURM placement (leave EMPTY to use the cluster default) ---
# Star HPC's docs list partition names as un-finalized placeholders and its own
# example sbatch omits --partition, so jobs go to the DEFAULT partition unless you
# set one here. Check the real names + which is default with:  sinfo -s
# These 30 models are tiny (a 2x256 MLP on 200k rows) -- CPU is plenty and avoids
# GPU queueing, so leave RCA_GPUS empty unless you specifically want a GPU node.
export RCA_PARTITION="${RCA_PARTITION:-}"     # e.g. a CPU partition name from `sinfo -s`
export RCA_ACCOUNT="${RCA_ACCOUNT:-}"         # only if your cluster requires --account
export RCA_GPUS="${RCA_GPUS:-}"               # e.g. "1" -> adds --gres=gpu:1 to training
                                              # (also set RCA_DEVICE=cuda to actually use it)
