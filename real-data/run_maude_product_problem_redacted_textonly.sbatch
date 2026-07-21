#!/bin/bash

#SBATCH --job-name=maude_pp_redacted_textonly
#SBATCH --output=/fs1/proj/trustworthy_ai_ml/maude/logs/maude_pp_redacted_textonly_%j.out
#SBATCH --error=/fs1/proj/trustworthy_ai_ml/maude/logs/maude_pp_redacted_textonly_%j.err

#SBATCH --qos=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Time   : $(date)"
echo "=========================================="

module load miniconda3/24.7.1-gcc-8.5.0-bxh7x2v
source "$(dirname "$(which conda)")/../etc/profile.d/conda.sh"
conda activate Topic_Topology_env

echo
# Use CPU only
export CUDA_VISIBLE_DEVICES=""
python - <<'EOF'
try:
    import torch
    print("Torch OK")
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("Torch import issue:", e)
EOF

export OPENFDA_API_KEY="your_openfda_api_key_here"
OUTDIR="/fs1/proj/trustworthy_ai_ml/maude/results/maude_pp_redacted_textonly"
mkdir -p "$OUTDIR"

MAX_PER_CLASS=3000
PAGE_SIZE=100
TARGET_COVERAGE=0.65
SEED=7

echo "These are the parameters for this run:"
echo "OUTDIR: $OUTDIR"
echo "MAX_PER_CLASS: $MAX_PER_CLASS"
echo "PAGE_SIZE: $PAGE_SIZE"
echo "TARGET_COVERAGE: $TARGET_COVERAGE"
echo "SEED: $SEED"

python /fs1/projects/trustworthy_ai_ml/maude/maude_product_problem_redacted_textonly.py \
    --outdir "$OUTDIR" \
    --api-key "$OPENFDA_API_KEY" \
    --max-per-class $MAX_PER_CLASS \
    --page-size $PAGE_SIZE \
    --target-coverage $TARGET_COVERAGE \
    --seed $SEED


echo
 echo "Finished: $(date)"
echo "Outputs in: $OUTDIR"
