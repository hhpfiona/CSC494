#!/bin/bash
#SBATCH --job-name=pluraltree_ablation
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_%x_%j.out
#SBATCH --error=runs/slurm_%x_%j.err

# PluralTree agentic ablation - Narval. Full n=100, judge-mode kp_recall.
#
# Run ONE ARM per job so each fits the 12h wall (full-100 both-arms is ~20-26
# GPU-hours and does NOT fit 12h). Submit twice:
#     cd CSC494 && mkdir -p runs
#     ARMS=no-context sbatch orchestration/run_ablation.slurm
#     ARMS=template   sbatch orchestration/run_ablation.slurm
# Records append per-query, so a timeout still leaves partial results.

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

# 1. Modules
module purge
module load StdEnv/2023
module load gcc/12.3
module load rust/1.76.0
module load python/3.11
module load arrow/16
module load cuda/12.2

# 2. Environment (node-local venv)
VENV="$SLURM_TMPDIR/pluraltree_env"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
# NOTE: requests added so the template arm uses SBERT, not the silent Jaccard
# fallback. If you deliberately want Jaccard, drop requests and pass --no_sbert.
pip install --no-index torch transformers tokenizers sentence-transformers \
    accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai requests || {
    echo "Alliance wheels incomplete; retrying with PyPI for the gaps..."
    pip install torch transformers tokenizers sentence-transformers \
        accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai requests
}
echo "--- installed packages ---"; pip list 2>/dev/null | grep -Ei \
    "torch|transformers|sentence|accelerate|requests|pydantic" || true

# 3. Offline caches
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_SRC="${MODEL_SRC:-$SCRATCH/models/llama31-8b}"
QUERIES="${QUERIES:-CulFiT/GlobalCultureQA/eval_set_n100.jsonl}"
MAX_LOOPS="${MAX_LOOPS:-3}"
MAX_PATHS="${MAX_PATHS:-12}"
ARMS="${ARMS:-no-context}"          # run ONE arm per job: no-context | template

# 3b. Stage model to node-local SSD
if [ -d "$MODEL_SRC" ]; then
    STAGED="$SLURM_TMPDIR/$(basename "$MODEL_SRC")"
    echo "Staging model: $MODEL_SRC -> $STAGED"
    time cp -r "$MODEL_SRC" "$STAGED"
    MODEL="$STAGED"
else
    echo "MODEL_SRC is not a local dir; using as-is: $MODEL_SRC"
    MODEL="$MODEL_SRC"
fi

# Fail early if the eval set isn't where we expect (don't waste a 12h slot).
if [ ! -f "$QUERIES" ]; then
    echo "FATAL: queries file not found: $QUERIES" >&2
    exit 1
fi

# 4. Run (single arm)
mkdir -p runs
echo "Ablation: MODEL=$MODEL QUERIES=$QUERIES ARMS=$ARMS MAX_LOOPS=$MAX_LOOPS MAX_PATHS=$MAX_PATHS"
python -m orchestration.run_local \
    --model "$MODEL" \
    --queries "$QUERIES" \
    --arms "$ARMS" \
    --max_loops "$MAX_LOOPS" \
    --max_paths "$MAX_PATHS"

echo "Job $SLURM_JOB_ID finished at $(date)"
