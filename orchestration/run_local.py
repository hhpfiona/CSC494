"""
Cluster entrypoint for the PluralTree ablation in LOCAL mode.

Loads ONE HuggingFace model + tokenizer onto the GPU, wraps it in LocalBackend,
and runs the three-topology ablation. This is what the SLURM script calls.

Why a separate entrypoint (not run_ablation --mode local):
  run_ablation builds a fresh backend per (query, topology) and intentionally
  refuses to instantiate a heavy local model repeatedly. Here we load the model
  ONCE and reuse the same backend across all runs, which is the only sane thing
  to do on a GPU node.

Usage (inside the SLURM job, after `module load` + venv/conda activate):
    python -m orchestration.run_local \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max_loops 3 \
        [--dtype bfloat16] [--max_new_tokens 2048]

Outputs to CSC494/runs/ just like the mock harness.
"""

from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime

from orchestration import bootstrap
bootstrap.install()

from orchestration.llm_backend import LocalBackend
from orchestration.agent_a_adapter import AgentA
from orchestration.agent_b_engine import AgentBCritiqueEngine
from orchestration.orchestrator import CulturalAgentOrchestrator
from orchestration.run_ablation import QUERIES  # reuse the shared eval set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_local")


def load_hf_model(model_name: str, dtype: str = "bfloat16"):
    """Load tokenizer + causal LM onto available GPU(s). Mirrors Tonga main.py."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loading model: %s (dtype=%s, device_map=auto)", model_name, dtype)
    # Newer transformers renamed `torch_dtype` -> `dtype`. 
    # Try the new name first, 
    # fall back to the old one so this works on both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch_dtype
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch_dtype
        )
    model.eval()
    logger.info("Model loaded. Device map ready.")
    return model, tokenizer


def run(model_name: str, max_loops: int, dtype: str, smoke: bool = False,
        only_topology: str | None = None):
    model, tokenizer = load_hf_model(model_name, dtype)

    # ONE backend instance, shared by both agents and the arbiter.
    backend = LocalBackend(model_obj=model, tokenizer_obj=tokenizer, model_name=model_name)

    # Smoke mode: 1 query, 1 topology, 1 loop — cheapest possible real run.
    queries = QUERIES[:1] if smoke else QUERIES
    topologies = (only_topology,) if only_topology else ("static", "parallel", "sequential")
    if smoke and not only_topology:
        topologies = ("sequential",)  # the most complex path, to shake out the loop
    if smoke:
        max_loops = 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_")
    tag = "smoke" if smoke else "local"
    jsonl_path = f"runs/ablation_{tag}_{safe_model}_{ts}.jsonl"
    summary_path = f"runs/ablation_{tag}_{safe_model}_{ts}_summary.json"
    records = []

    for q in queries:
        agent_a = AgentA(backend, location=q["location"], sub_topic=q["sub_topic"])
        agent_b = AgentBCritiqueEngine(backend)
        orch = CulturalAgentOrchestrator(agent_a, agent_b, arbiter_backend=backend)

        for topology in topologies:
            if topology == "static":
                result = orch.static_integration(q["query"], q["ground_truth"])
            elif topology == "parallel":
                result = orch.parallel_debate(q["query"], q["ground_truth"])
            else:
                result = orch.sequential_debate(q["query"], q["ground_truth"], max_loops=max_loops)

            rec = {
                "model": model_name, "query": q["query"], "location": q["location"],
                "topology": topology,
                "n_final_paths": len(result.get("final_paths", [])),
                "trace": result["trace"],
            }
            records.append(rec)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info("[%s | %s] approved=%s mean_precision=%.2f loops=%d repairs=%d",
                        q["query"], topology, rec["trace"]["final_approved"],
                        rec["trace"]["final_mean_precision"],
                        rec["trace"]["loops"], rec["trace"]["repairs"])

    summary = {}
    for topo in ("static", "parallel", "sequential"):
        rs = [r for r in records if r["topology"] == topo]
        n = len(rs) or 1
        summary[topo] = {
            "n_runs": len(rs),
            "approval_rate": sum(r["trace"]["final_approved"] for r in rs) / n,
            "avg_mean_precision": sum(r["trace"]["final_mean_precision"] for r in rs) / n,
            "avg_loops": sum(r["trace"]["loops"] for r in rs) / n,
            "avg_repairs": sum(r["trace"]["repairs"] for r in rs) / n,
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s and %s", jsonl_path, summary_path)
    print("\n=== ABLATION SUMMARY (local) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--smoke", action="store_true",
                   help="1 query / 1 topology / 1 loop — cheap first-run check")
    p.add_argument("--topology", choices=["static", "parallel", "sequential"],
                   default=None, help="run only this topology")
    args = p.parse_args()
    run(args.model, args.max_loops, args.dtype, smoke=args.smoke,
        only_topology=args.topology)
