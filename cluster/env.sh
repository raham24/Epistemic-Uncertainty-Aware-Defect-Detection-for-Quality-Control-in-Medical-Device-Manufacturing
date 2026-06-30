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

# device for mlp.py: the sweep TRAINS ON GPU by default (RCA_DEVICE=cuda paired
# with RCA_GPUS below). Force CPU with RCA_DEVICE=cpu. run_local.sh auto-falls back
# to the Apple GPU (mps) or cpu when no CUDA is present, so it still works on a Mac.
export RCA_DEVICE="${RCA_DEVICE:-cuda}"

# --- SLURM placement (leave a value EMPTY to omit that flag) ---
# Star HPC's example sbatch omits --partition, so jobs land on the DEFAULT partition
# unless you set one. If GPUs live on a named partition, set it here. Discover names
# + which is default with `sinfo -s`, and GPU types with `sinfo -o '%P %G'`.
export RCA_PARTITION="${RCA_PARTITION:-}"     # e.g. the GPU partition name from `sinfo -s`
export RCA_ACCOUNT="${RCA_ACCOUNT:-}"         # only if your cluster requires --account
# GPUs PER training task. submit.sh turns this into --gres=gpu:$RCA_GPUS on the TRAIN
# array only (dataset generation stays CPU). Use "1" for any GPU, or pin a type if the
# cluster requires it: "A30:1" / "A100:1" / "H100:1". Set RCA_GPUS="" to train on CPU.
export RCA_GPUS="${RCA_GPUS:-1}"

# --- array throttle: cap how many array tasks run AT ONCE (submit.sh -> --array=1-N%K).
# On GPUs, set this to how many GPUs you can use simultaneously (Slurm also gates on
# gres, so a larger value simply queues the excess). 16 is a friendly shared default. ---
export RCA_MAX_PARALLEL="${RCA_MAX_PARALLEL:-16}"
# Hard ceiling guard: refuse to submit if the matrix exceeds the cluster's
# MaxArraySize (Slurm default 1001). Check yours with: scontrol show config | grep MaxArraySize
export RCA_MAX_ARRAY="${RCA_MAX_ARRAY:-1000}"
