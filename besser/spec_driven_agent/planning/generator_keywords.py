"""Keyword routing to a Phase 1 generator: the fallback, never the decision.

The forced ``choose_generator`` call decides the generator. These functions
serve the cases it leaves open (the call failed, or answered NONE for a
Python web app or backend with no named stack) and the preview, which runs
with no LLM call and no key. Both callers pass an object carrying the
assembled models (``LLMOrchestrator`` or ``AssembledModels``), so the
preview and the run route a request the same way.
"""
from __future__ import annotations

import re

# Frameworks with no BESSER generator: a request naming one is built from scratch.
NO_GENERATOR_FRAMEWORKS = (
    "nestjs", "next.js", "nextjs", "express", "spring boot", "springboot",
    "laravel", "rails", "golang", "axum", "actix", "angular", "vue", "svelte", "nuxt",
)


def keyword_generator(models, instructions: str) -> str | None:
    """A generator the request's words name, ``""`` for none, ``None`` if undecided."""
    lower = (instructions or "").lower()

    def _has(word: str) -> bool:
        return bool(re.search(r"\b" + re.escape(word) + r"\b", lower))

    if any(_has(fw) for fw in NO_GENERATOR_FRAMEWORKS):
        return ""

    # Specialist generators for the newer modeling artifacts.
    if models.bpmn_model is not None and (
        models.primary_kind == "bpmn"
        or _has("bpmn")
        or _has("workflow")
        or "business process" in lower
        or "process model" in lower
    ):
        return "generate_bpmn"
    if models.nn_model is not None:
        if _has("tensorflow") or _has("keras"):
            return "generate_tensorflow"
        if (
            models.primary_kind == "nn"
            or _has("pytorch")
            or _has("torch")
            or "neural network" in lower
        ):
            return "generate_pytorch"
    if models.agent_model is not None and (
        models.primary_kind == "agent"
        or _has("baf")
        or _has("chatbot")
        or "agent framework" in lower
    ):
        return "generate_baf"
    if models.object_model is not None and (
        models.primary_kind == "object"
        or "json object" in lower
        or _has("fixture")
        or _has("fixtures")
        or "seed data" in lower
    ):
        return "generate_json_object"
    if models.domain_model is not None and _has("supabase"):
        return "generate_supabase"

    # Positive matches: what the user explicitly asked for.
    if _has("django"):
        return "generate_django"
    full_stack = "generate_web_app" if models.gui_model else "generate_fastapi_backend"
    if _has("fastapi") or _has("fast api"):
        return full_stack
    if _has("backend") or _has("api") or _has("rest"):
        return full_stack
    if _has("full-stack") or _has("fullstack") or _has("full stack") or _has("web app"):
        return full_stack
    if _has("pydantic") and not _has("api") and not _has("backend"):
        return "generate_pydantic"
    if _has("sqlalchemy"):
        return "generate_sqlalchemy"
    return None


def default_generator(models, instructions: str) -> str | None:
    """The generator the loaded models call for when nothing else decided one."""
    if models.primary_kind == "bpmn" and models.bpmn_model is not None:
        return "generate_bpmn"
    if models.primary_kind == "nn" and models.nn_model is not None:
        lower = (instructions or "").lower()
        if "tensorflow" in lower or "keras" in lower:
            return "generate_tensorflow"
        return "generate_pytorch"
    if models.primary_kind == "agent" and models.agent_model is not None:
        return "generate_baf"
    if models.primary_kind == "object" and models.object_model is not None:
        return "generate_json_object"
    if models.quantum_circuit is not None:
        return "generate_qiskit"
    if models.gui_model and models.domain_model is not None:
        return "generate_web_app"
    if models.domain_model is not None:
        try:
            if models.domain_model.get_classes():
                return "generate_fastapi_backend"
        except Exception:
            pass
    return None
