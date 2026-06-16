# Running on Narval (def-enaskt)

Your project lives at `~/projects/def-enaskt/hhpfiona/CSC494`, copied from local.
Your allocation is **Opportunistic Use** (low priority, no guaranteed resources),
so keep jobs modest and expect variable queue times.

> Compute nodes have **no internet**. Anything downloaded from the web (model
> weights, pip packages from PyPI, HF datasets) must be fetched on a **login
> node** first, then read from disk inside the job.

---

## Step 0 (once) — Pre-download the model on a LOGIN node

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494
module load python/3.11
pip install --user huggingface_hub
export HF_HOME=$SCRATCH/hf_cache       # big files belong in scratch, not project
huggingface-cli login                  # needed for gated models like Llama
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $SCRATCH/models/llama31-8b
```

Then you can point `--model` either at the HF id (resolved from `$HF_HOME`) or
directly at `$SCRATCH/models/llama31-8b`. The local dir is the most reliable
offline.

---

## Step 1 — Verify the drop (no GPU needed, run on login node)

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494
module load python/3.11
python -m orchestration.check_setup
```

Expect `PASS — setup looks good.` If it can't find a repo folder, the
`orchestration/` folder isn't a direct child of `CSC494/`, or a repo folder was
renamed.

---

## Step 2 — Cheap smoke test via `salloc` (interactive, 1 query / 1 topology / 1 loop)

Grab a short interactive GPU session and run the smallest possible real job.
This shakes out env/model/parsing problems for pennies before you submit a full
batch job.

```bash
# Request a small interactive allocation (adjust time/mem down for opportunistic)
salloc --account=def-enaskt --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00

# --- once the shell drops you onto the GPU node ---
cd ~/projects/def-enaskt/hhpfiona/CSC494

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

# build a node-local venv (fast, avoids stale metadata)
python -m venv $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch transformers tokenizers sentence-transformers \
    pandas openpyxl tenacity tqdm pydantic python-dotenv

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# the smoke test: 1 query, sequential only, max_loops=1
python -m orchestration.run_local \
    --model $SCRATCH/models/llama31-8b \
    --smoke

# when done
exit   # releases the salloc allocation
```

What `--smoke` does: 1 query, the `sequential` topology only, `max_loops=1`.
It exercises the full real path — model load, Agent A generate, Agent B critique,
one repair — so if JSON parsing of real Llama output is going to break, it breaks
here cheaply. Check `runs/ablation_smoke_*.jsonl` for the trace.

You can also target one topology explicitly, e.g. `--topology static`.

---

## Step 3 — Full batch run

Once the smoke test passes, submit the real job:

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494
# optional overrides: export MODEL=$SCRATCH/models/llama31-8b MAX_LOOPS=3
sbatch orchestration/run_ablation.slurm
squeue -u $USER          # watch the queue
```

Logs stream to `runs/slurm_pluraltree_ablation_<jobid>.out`. Results land in
`runs/ablation_local_*.jsonl` and `*_summary.json`.

> If you pre-downloaded to `$SCRATCH/models/llama31-8b`, set
> `export MODEL=$SCRATCH/models/llama31-8b` before `sbatch`, or edit the `MODEL`
> default in the `.slurm` file. The HF-id default only works if the id is cached
> in `$HF_HOME`.

---

## Module versions — adjust to what Narval currently exposes

The `module load` lines are best-guess. Confirm with `module spider <name>` and
swap versions as needed. The one rule that's not optional: **load `rust` before
installing `vllm`/`tokenizers`** (from the Agent A handoff — it's what caused the
"invalid wheel" failures). Prefer `--no-index` (Alliance wheels) over PyPI.
