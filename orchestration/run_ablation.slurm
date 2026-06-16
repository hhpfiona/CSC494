#!/bin/bash
#SBATCH --job-name=pluraltree_ablation
#SBATCH --account=def-enaskt             # Enas Altarawneh's allocation (ztd-685)
#SBATCH --gres=gpu:1                     # generic GPU; opportunistic use, don't pin a type
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_%x_%j.out
#SBATCH --error=runs/slurm_%x_%j.err

# ---------------------------------------------------------------------------
# PluralTree agentic ablation — Narval (Digital Research Alliance of Canada)
#
# Submit from the CSC494/ directory:
#     cd CSC494
#     sbatch orchestration/run_ablation.slurm
#
# Notes baked in from the Agent A handoff (Narval env pain points):
#   - load rust BEFORE any vllm/tokenizers build
#   - prefer Alliance module wheels; avoid a corrupted venv (clean wipe if needed)
#   - models/datasets must be pre-downloaded (compute nodes have NO internet)
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

# --- 1. Modules (order matters: rust before vllm/tokenizers) ----------------
module purge
module load StdEnv/2023
module load gcc/12.3            # provides a compiler for native wheels
module load rust/1.76.0         # REQUIRED before installing vllm/tokenizers
module load python/3.11
module load arrow/16            # pyarrow backing for pandas, Alliance-provided
module load cuda/12.2

# --- 2. Environment ---------------------------------------------------------
# Use a project-local venv. If you hit the "invalid wheel" corruption again,
# wipe and recreate (per the Agent A handoff): rm -rf "$VENV" then re-run.
VENV="$SLURM_TMPDIR/pluraltree_env"     # node-local: fast + avoids stale metadata
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# Prefer Alliance prebuilt wheels (--no-index). Fall back to PyPI only if needed.
pip install --no-index torch transformers tokenizers sentence-transformers \
    pandas openpyxl tenacity tqdm pydantic python-dotenv openai || {
    echo "Alliance wheels incomplete; retrying with PyPI for the gaps..."
    pip install torch transformers tokenizers sentence-transformers \
        pandas openpyxl tenacity tqdm pydantic python-dotenv openai
}

# --- 3. Offline caches (compute nodes are air-gapped) -----------------------
# Pre-download the model on a LOGIN node first, e.g.:
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $SCRATCH/models/llama31-8b
# then point HF at the cache and run fully offline:
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_LOOPS="${MAX_LOOPS:-3}"

# --- 4. Run -----------------------------------------------------------------
mkdir -p runs
echo "Running local-mode ablation with MODEL=$MODEL MAX_LOOPS=$MAX_LOOPS"
python -m orchestration.run_local --model "$MODEL" --max_loops "$MAX_LOOPS"

echo "Job $SLURM_JOB_ID finished at $(date)"
