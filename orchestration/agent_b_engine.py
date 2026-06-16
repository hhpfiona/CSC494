"""
Agent B critique engine (in-place): CulFiT Group/Topic/KnowledgePoint checks
using the REAL EVAL_* prompts from CulFiT/utils/prompt_utils.py (loaded in place
via bootstrap), routed through the model-agnostic backend. Adds get_static_schema().
"""

from __future__ import annotations
import logging

from orchestration.llm_backend import LLMBackend
from orchestration import bootstrap

_mods = bootstrap.install()
_pu = _mods["culfit_prompt_utils"]
EVAL_CULTURAL_POINTS_PROMPT = _pu.EVAL_CULTURAL_POINTS_PROMPT
EVAL_CULTURAL_GROUP_PROMPT = _pu.EVAL_CULTURAL_GROUP_PROMPT
EVAL_TOPIC_PROMPT = _pu.EVAL_TOPIC_PROMPT

logger = logging.getLogger(__name__)


class AgentBCritiqueEngine:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def _generate(self, messages: list[dict]) -> str:
        return self.backend.chat(messages, temperature=0.0) or ""

    def get_static_schema(self, query: str) -> dict:
        return {
            "query": query,
            "evaluation_dimensions": [
                "Cultural Group Alignment", "Topic Alignment", "Knowledge Path Alignment"],
            "note": "Static CulFiT schema; no deliberation performed.",
        }

    def evaluate_single_path(self, agent_a_node: dict, ground_truth: dict) -> dict:
        answer_cultural_group = agent_a_node.get("location", "Unknown")
        answer_topic = agent_a_node.get("sub_topic", "Unknown")
        answer_knowledge_point = agent_a_node.get("llm_result", "")
        grounded_cultural_group = ground_truth.get("location", "Unknown")
        grounded_topic = ground_truth.get("sub_topic", "Unknown")
        grounded_knowledge_points = ground_truth.get("verified_points", [])

        msg_group = [{"role": "user", "content": EVAL_CULTURAL_GROUP_PROMPT.format(
            answer_cultural_group, grounded_cultural_group)}]
        msg_topic = [{"role": "user", "content": EVAL_TOPIC_PROMPT.format(
            answer_topic, grounded_topic)}]
        msg_kp = [{"role": "user", "content": EVAL_CULTURAL_POINTS_PROMPT.format(
            answer_knowledge_point, grounded_knowledge_points)}]

        res_group = self._generate(msg_group)
        res_topic = self._generate(msg_topic)
        res_kp = self._generate(msg_kp)

        labels = ["Cultural Group Alignment", "Topic Alignment", "Knowledge Path Alignment"]
        verdicts, critique_details = [], []
        for label, res in zip(labels, [res_group, res_topic, res_kp]):
            if res and "Yes" in res:
                verdicts.append("Yes")
            else:
                verdicts.append("No")
                critique_details.append(f"[{label} Failure]: {res}")

        precision = sum(1 for v in verdicts if v == "Yes") / len(verdicts)
        approved = (precision == 1.0)
        return {
            "approved": approved, "precision_score": precision,
            "verdicts": dict(zip(labels, verdicts)),
            "feedback": " ".join(critique_details) if not approved
                        else "Culturally verified and validated.",
        }

    def evaluate_payload_batch(self, agent_a_payload_list: list[dict],
                               ground_truth_schema: dict) -> dict:
        all_approved = True
        accumulated_feedback, scores, per_path = [], [], []
        for item in agent_a_payload_list:
            ev = self.evaluate_single_path(item, ground_truth_schema)
            scores.append(ev["precision_score"])
            per_path.append({"path": item.get("llm_result", ""), **ev})
            if not ev["approved"]:
                all_approved = False
                accumulated_feedback.append(
                    f"Offending Path: '{item.get('llm_result','')}' -> Reason: {ev['feedback']}")
        mean_precision = sum(scores) / len(scores) if scores else 0.0
        return {
            "approved": all_approved, "mean_precision": mean_precision,
            "n_paths": len(agent_a_payload_list), "per_path": per_path,
            "feedback": "\n".join(accumulated_feedback) if accumulated_feedback
                        else "All structural reasoning paths passed CulFiT verification.",
        }
