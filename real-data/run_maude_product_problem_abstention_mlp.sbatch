#!/bin/bash

#SBATCH --job-name=maude_pp_abstention
#SBATCH --output=/fs1/proj/trustworthy_ai_ml/maude/logs/maude_pp_abstention_%j.out
#SBATCH --error=/fs1/proj/trustworthy_ai_ml/maude/logs/maude_pp_abstention_%j.err
#SBATCH --qos=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Time   : $(date)"
echo "=========================================="

module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
# resolve the base prefix robustly (works whether conda is a binary or a shell fn)
source "$(conda info --base)/etc/profile.d/conda.sh"
# the project `paper` env (upgrade it first: conda env update -f environment.yml)
conda activate paper

echo

echo "Python:"
which python
python --version

echo
python - <<EOF
try:
    import torch
    print("Torch OK")
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch import issue:", e)
EOF

# OPENFDA_API_KEY is inherited from your login environment (set in ~/.bashrc).
# sbatch defaults to --export=ALL, so the value propagates into this job.
# If it is unset, the script still runs -- openFDA allows anonymous access,
# just at a lower rate limit.
if [ -z "${OPENFDA_API_KEY:-}" ]; then
    echo "WARNING: OPENFDA_API_KEY is not set; using anonymous openFDA access."
    OPENFDA_API_KEY=""
fi

OUTDIR="/fs1/proj/trustworthy_ai_ml/maude/results/maude_pp_abstention"
mkdir -p "$OUTDIR"
mkdir -p /fs1/proj/trustworthy_ai_ml/maude/logs

MAX_PER_CLASS=3000
PAGE_SIZE=100
TARGET_COVERAGE=0.65
SEED=7
O=2.0

echo "These are the parameters for this run:"
echo "OUTDIR: $OUTDIR"
echo "MAX_PER_CLASS: $MAX_PER_CLASS"
echo "PAGE_SIZE: $PAGE_SIZE"
echo "TARGET_COVERAGE: $TARGET_COVERAGE"
echo "ABSTENTION PAYOFF O: $O"
echo "SEED: $SEED"

python maude_product_problem_abstention_mlp.py \
    --outdir "$OUTDIR" \
    --api-key "$OPENFDA_API_KEY" \
    --max-per-class "$MAX_PER_CLASS" \
    --page-size "$PAGE_SIZE" \
    --target-coverage "$TARGET_COVERAGE" \
    --o "$O" \
    --device cpu \
    --seed "$SEED"

echo

echo "Finished: $(date)"
echo "Outputs in: $OUTDIR"
