#!/bin/bash
#SBATCH --job-name=pluraltree_ablation
#SBATCH --account=def-enaskt             # Enas Altarawneh's allocation (ztd-685)
#SBATCH --gres=gpu:1                     # generic GPU; opportunistic use, don't pin a type
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm_%x_%j.out
#SBATCH --error=runs/slurm_%x_%j.err

# ---------------------------------------------------------------------------
# PluralTree agentic ablation — Narval (Digital Research Alliance of Canada)
#
# Submit from the CSC494/ directory:
#     cd CSC494
#     sbatch orchestration/run_ablation.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

# 1. Modules (order matters: rust before vllm/tokenizers) 
module purge
module load StdEnv/2023
module load gcc/12.3            # provides a compiler for native wheels
module load rust/1.76.0         # REQUIRED before installing vllm/tokenizers
module load python/3.11
module load arrow/16            # pyarrow backing for pandas, Alliance-provided
module load cuda/12.2

# 2. Environment 
# Use a project-local venv. If you hit the "invalid wheel" corruption again,
# wipe and recreate (per the Agent A handoff): rm -rf "$VENV" then re-run.
VENV="$SLURM_TMPDIR/pluraltree_env"     # node-local: fast + avoids stale metadata
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# Prefer Alliance prebuilt wheels (--no-index). Fall back to PyPI only if needed.
pip install --no-index torch transformers tokenizers sentence-transformers \
    accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai || {
    echo "Alliance wheels incomplete; retrying with PyPI for the gaps..."
    pip install torch transformers tokenizers sentence-transformers \
        accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai
}

# 3. Offline caches (compute nodes are air-gapped) 
# Pre-download the model on a LOGIN node first, e.g.:
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $SCRATCH/models/llama31-8b
# then point HF at the cache and run fully offline:
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MODEL_SRC = where the weights live on /scratch (Lustre). Set this to the
# pre-downloaded local dir. Loading 16GB directly off Lustre is SLOW, so we
# stage it to node-local SSD ($SLURM_TMPDIR) and load from there.
MODEL_SRC="${MODEL_SRC:-$SCRATCH/models/llama31-8b}"
MAX_LOOPS="${MAX_LOOPS:-3}"
MAX_PATHS="${MAX_PATHS:-12}"   # cap paths/query so the full ablation finishes in time

# 3b. Stage model to node-local SSD (fast) 
# If MODEL_SRC is a directory on scratch, copy it to $SLURM_TMPDIR once
# (sequential read = fast) and load from local disk. If MODEL_SRC is already a
# bare HF id (no such dir), skip staging and let transformers use $HF_HOME.
if [ -d "$MODEL_SRC" ]; then
    STAGED="$SLURM_TMPDIR/$(basename "$MODEL_SRC")"
    echo "Staging model: $MODEL_SRC -> $STAGED"
    time cp -r "$MODEL_SRC" "$STAGED"
    MODEL="$STAGED"
else
    echo "MODEL_SRC is not a local dir; using as-is (HF id / cache): $MODEL_SRC"
    MODEL="$MODEL_SRC"
fi

# 4. Run
mkdir -p runs
echo "Running local-mode ablation with MODEL=$MODEL MAX_LOOPS=$MAX_LOOPS MAX_PATHS=$MAX_PATHS"
python -m orchestration.run_local --model "$MODEL" --max_loops "$MAX_LOOPS" --max_paths "$MAX_PATHS"

echo "Job $SLURM_JOB_ID finished at $(date)"
