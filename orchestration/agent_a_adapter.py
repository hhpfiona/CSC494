"""
Agent A adapter (in-place): wraps Tonga's CCKG generation into a callable agent
exposing .generate(query) and .repair(prompt), importing the REAL Tonga modules
in place via bootstrap (no copies).
"""

from __future__ import annotations
import logging
from types import SimpleNamespace

from orchestration.llm_backend import LLMBackend
from orchestration import bootstrap

_mods = bootstrap.install()
generation_prompts = _mods["agentA_prompt_templates"].generation_prompts
parse_llm_response = _mods["agentA_response_parser"].parse_llm_response

logger = logging.getLogger(__name__)


class AgentA:
    def __init__(self, backend: LLMBackend, location: str = "England",
                sub_topic: str = "breakfast", language: str = "English",
                prompt_key: str | None = None, max_paths: int | None = None):
        self.backend = backend
        self.location = location
        self.sub_topic = sub_topic
        self.language = language
        self.prompt_key = prompt_key or "England_gen"
        self.max_paths = max_paths  # None = no cap; useful to keep smoke runs fast

    def _build_generation_messages(self, query: str) -> list[dict]:
        template = generation_prompts[self.prompt_key]
        sub_topic = query or self.sub_topic
        msgs = []
        for m in template:
            content = m["content"].format(
                location=self.location, sub_topic=sub_topic,
                language=self.language, premise="",
            )
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def _parse(self, raw_text: str) -> list[dict]:
        args = SimpleNamespace(action="initial_generation")
        try:
            parsed = parse_llm_response(args, raw_text)
        except Exception as e:
            logger.warning("Agent A parse failed (%s); returning empty.", e)
            return []
        if not isinstance(parsed, list):
            return []
        for entry in parsed:
            entry.setdefault("location", self.location)
            entry.setdefault("sub_topic", self.sub_topic)
        if self.max_paths is not None:
            parsed = parsed[:self.max_paths]
        return parsed

    def generate(self, query: str) -> list[dict]:
        messages = self._build_generation_messages(query)
        raw = self.backend.chat(messages, temperature=1.0)
        paths = self._parse(raw)
        logger.info("Agent A generated %d path(s).", len(paths))
        return paths

    def repair(self, repair_prompt: str) -> list[dict]:
        messages = [
            {"role": "system",
             "content": "You are a cultural commonsense knowledge extraction "
                        "assistant. Output STRICT JSON only: a JSON array of "
                        "objects with keys action, knowledge, relation_type, result."},
            {"role": "user", "content": repair_prompt},
        ]
        raw = self.backend.chat(messages, temperature=0.7)
        paths = self._parse(raw)
        logger.info("Agent A repaired into %d path(s).", len(paths))
        return paths

    def static_context(self, query: str) -> list[dict]:
        return self.generate(query)
