"""
Unified, model-agnostic LLM backend for the PluralTree agentic system.

One flag (`mode`) flips the entire pipeline between:
  - "mock"  : deterministic canned responses, runs anywhere, no GPU / no API key.
              Use this NOW while the Narval cluster is down to develop and debug
              the deliberation logic.
  - "api"   : OpenAI-compatible HTTP endpoint (GPT-4o via chatanywhere, or a
              vLLM server at 0.0.0.0:8001). Reuses agent_b.utils.llm_utils.
  - "local" : in-process HuggingFace transformers model on a GPU. Reuses
              agent_b.utils.llm_utils.lama_generation.

Both Agent A and Agent B receive ONE of these objects and call `.chat(messages)`.
Switching mock -> cluster is a one-line change in the ablation config; no agent
code changes.
"""

from __future__ import annotations
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LLMBackend:
    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Take OpenAI-style [{'role','content'}, ...], return raw string."""
        raise NotImplementedError


class MockBackend(LLMBackend):
    """
    Deterministic backend for offline development.

    A `responder` callable receives the messages list and returns a string.
    The default responder is generic; the ablation harness installs richer
    scenario responders so the three topologies actually exercise their logic
    (e.g. Agent A "repairs" a draft, Agent B flips from No -> Yes).
    """

    def __init__(self, responder: Optional[Callable[[list[dict]], str]] = None,
                 name: str = "mock"):
        self.responder = responder or self._default_responder
        self.name = name
        self.call_count = 0

    def _default_responder(self, messages: list[dict]) -> str:
        return "Yes - Validated alignment."

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        self.call_count += 1
        return self.responder(messages)


class APIBackend(LLMBackend):
    """OpenAI-compatible endpoint. Lazily imports so mock mode never needs openai."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        from orchestration import bootstrap
        bootstrap.install()
        from culfit_llm_utils import openai_response  # bootstrap-registered
        self._openai_response = openai_response

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        out = self._openai_response(
            model_name=self.model_name, messages=messages, temperature=temperature
        )
        return out or ""


class LocalBackend(LLMBackend):
    """In-process HF transformers model (cluster GPU). Lazily imports torch/transformers."""

    def __init__(self, model_obj, tokenizer_obj, model_name: str = "llama3"):
        self.model_obj = model_obj
        self.tokenizer_obj = tokenizer_obj
        self.model_name = model_name
        from orchestration import bootstrap
        bootstrap.install()
        from culfit_llm_utils import lama_generation  # bootstrap-registered
        self._lama_generation = lama_generation

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        out = self._lama_generation(
            model=self.model_obj, tokenizer=self.tokenizer_obj,
            input_messages=messages, temperature=temperature,
        )
        return out or ""


def make_backend(mode: str, **kwargs) -> LLMBackend:
    """Factory. `mode` in {"mock","api","local"}."""
    mode = (mode or "mock").lower()
    if mode == "mock":
        return MockBackend(responder=kwargs.get("responder"),
                           name=kwargs.get("name", "mock"))
    if mode == "api":
        return APIBackend(model_name=kwargs.get("model_name", "gpt-4o"))
    if mode == "local":
        return LocalBackend(
            model_obj=kwargs["model_obj"],
            tokenizer_obj=kwargs["tokenizer_obj"],
            model_name=kwargs.get("model_name", "llama3"),
        )
    raise ValueError(f"Unknown backend mode: {mode!r} (expected mock/api/local)")
