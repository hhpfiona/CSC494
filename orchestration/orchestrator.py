"""
CulturalAgentOrchestrator: mediates Agent A (CCKG generator) and Agent B
(CulFiT critique) across three communication topologies for the ablation study.

Fixes over the original orchestrator.py:
  - static_integration / parallel_debate previously called a non-existent
    `agent_b.get_static_schema()` -> now provided by the engine.
  - Agent A now returns parsed lists (not raw strings), so `_safe_json_parse`
    is applied only where a backend might still hand back a raw string.
  - Every mode returns a structured `trace` (verdicts, precision, loop count,
    repair attempts) so metrics can be chosen after runs.

An `arbiter_backend` (LLMBackend) is used by parallel_debate for synthesis.
"""

from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


class CulturalAgentOrchestrator:
    def __init__(self, agent_a, agent_b_engine, arbiter_backend=None):
        self.agent_a = agent_a
        self.agent_b = agent_b_engine
        self.arbiter_backend = arbiter_backend

    @staticmethod
    def _coerce_paths(maybe_paths, fallback):
        """Agent A returns a list already; tolerate a raw string just in case."""
        if isinstance(maybe_paths, list):
            return maybe_paths
        if isinstance(maybe_paths, str):
            import re
            try:
                return json.loads(maybe_paths)
            except json.JSONDecodeError:
                m = re.search(r'\[\s*\{.*?\}\s*\]', maybe_paths, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
            logger.warning("Could not coerce Agent A output; using fallback.")
        return fallback

    def static_integration(self, query, ground_truth_schema):
        logger.info("[static] start")
        paths = self.agent_a.generate(query)
        schema = self.agent_b.get_static_schema(query)
        # Single critique pass for a comparable quality signal (no repair).
        critique = self.agent_b.evaluate_payload_batch(paths, ground_truth_schema) if paths else {
            "approved": False, "mean_precision": 0.0, "n_paths": 0, "per_path": [],
            "feedback": "No paths generated."}
        return {
            "mode": "static",
            "final_paths": paths,
            "cultural_rules": schema,
            "trace": {"loops": 0, "repairs": 0,
                      "final_approved": critique["approved"],
                      "final_mean_precision": critique["mean_precision"],
                      "critique": critique},
        }

    def parallel_debate(self, query, ground_truth_schema):
        logger.info("[parallel] start")
        draft = self.agent_a.generate(query)
        schema = self.agent_b.get_static_schema(query)
        critique = self.agent_b.evaluate_payload_batch(draft, ground_truth_schema) if draft else {
            "approved": False, "mean_precision": 0.0, "n_paths": 0, "per_path": [],
            "feedback": "No paths generated."}

        synthesis = None
        if self.arbiter_backend is not None:
            prompt = (
                "You are the final arbiter. Reconcile Agent A's cultural graph "
                "paths and Agent B's critique into a single culturally sensitive "
                "JSON array.\n"
                f"Graph Paths: {json.dumps(draft)}\n"
                f"Critique: {json.dumps(critique.get('feedback'))}\n"
                f"Schema: {json.dumps(schema)}\nOutput strict JSON only."
            )
            raw = self.arbiter_backend.chat(
                [{"role": "user", "content": prompt}], temperature=0.3)
            synthesis = self._coerce_paths(raw, draft)

        final_paths = synthesis if synthesis else draft
        return {
            "mode": "parallel",
            "final_paths": final_paths,
            "cultural_rules": schema,
            "trace": {"loops": 1, "repairs": 0,
                      "final_approved": critique["approved"],
                      "final_mean_precision": critique["mean_precision"],
                      "critique": critique,
                      "arbiter_used": self.arbiter_backend is not None},
        }

    def sequential_debate(self, query, ground_truth_schema, max_loops=3):
        logger.info("[sequential] start (max_loops=%d)", max_loops)
        current = self._coerce_paths(self.agent_a.generate(query), [])
        if not current:
            logger.error("[sequential] empty initial draft; aborting.")
            return {"mode": "sequential", "final_paths": [],
                    "trace": {"loops": 0, "repairs": 0, "final_approved": False,
                              "final_mean_precision": 0.0, "iterations": []}}

        iterations, repairs = [], 0
        final_critique = None
        for i in range(max_loops):
            critique = self.agent_b.evaluate_payload_batch(current, ground_truth_schema)
            final_critique = critique
            iterations.append({
                "iteration": i + 1,
                "approved": critique["approved"],
                "mean_precision": critique["mean_precision"],
                "n_paths": critique["n_paths"],
            })
            if critique.get("approved"):
                logger.info("[sequential] approved at iteration %d", i + 1)
                break
            if i == max_loops - 1:
                break  # no point repairing on the last allowed loop
            repair_prompt = (
                "Your previous draft was critiqued.\n"
                f"Original Draft: {json.dumps(current)}\n"
                f"Critique: {critique.get('feedback')}\n"
                "Repair the empirical paths to satisfy the cultural critique. "
                "Output strict JSON only."
            )
            repaired = self._coerce_paths(self.agent_a.repair(repair_prompt), current)
            repairs += 1
            current = repaired if repaired else current

        return {
            "mode": "sequential",
            "final_paths": current,
            "trace": {
                "loops": len(iterations),
                "repairs": repairs,
                "final_approved": bool(final_critique and final_critique["approved"]),
                "final_mean_precision": final_critique["mean_precision"] if final_critique else 0.0,
                "iterations": iterations,
                "critique": final_critique,
            },
        }
