"""Pre-flight plan preview for smart generation.

Builds a ``PreviewPlan`` showing what smart-generate would do, before
spending any LLM tokens:

    1. Which primary model it detected (class, gui, agent, …).
    2. Which auxiliary models are present (counts only — no payloads).
    3. Which BESSER deterministic generator would run in Phase 1.
    4. A rough cost / turns / duration estimate so the user can decide
       whether to commit.

The classifier is deliberately deterministic (keyword + model presence)
in v1 — no LLM call. That means no API key is needed for the preview and
it responds instantly. An LLM refinement pass can be layered on later
behind a flag; the response shape won't change.

Cost estimation is a heuristic based on the primary model, instruction
length, and which auxiliaries the LLM would need to weave in. Better to
under-promise than over-promise — we round up.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from besser.utilities.web_modeling_editor.backend.services.smart_generation.model_assembly import (
    AssembledModels,
)


# Names mirrored from the orchestrator's keyword branch so that the
# preview and the real Phase 1 always agree on which generator will run
# for a given instruction. Keep in sync with
# ``LLMOrchestrator._select_generator_keyword``.
_EXPLICIT_NO_GENERATOR_FRAMEWORKS: tuple[str, ...] = (
    "nestjs", "next.js", "nextjs", "express", "spring boot", "springboot",
    "laravel", "rails", "golang", "axum", "actix", "angular", "vue",
    "svelte", "nuxt",
)


@dataclass(frozen=True)
class PreviewPlan:
    """Structured response for ``POST /besser_api/smart-preview``.

    All numeric fields are integers or floats suitable for a JSON UI.
    ``target_generator`` is ``None`` when smart-generation would run the
    LLM from scratch (e.g. state-machine-only or user asked for NestJS).
    """

    primary_kind: str
    auxiliary_kinds: list[str]
    target_generator: str | None
    target_generator_confidence: float   # 0.0–1.0
    summary: str                          # one-line human-friendly description
    estimated_turns: int
    estimated_cost_usd: float
    estimated_duration_seconds: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_kind": self.primary_kind,
            "auxiliary_kinds": self.auxiliary_kinds,
            "target_generator": self.target_generator,
            "target_generator_confidence": round(self.target_generator_confidence, 2),
            "summary": self.summary,
            "estimated_turns": self.estimated_turns,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "notes": list(self.notes),
        }


def build_preview(
    assembled: AssembledModels,
    instructions: str,
    max_cost_usd: float,
    max_runtime_seconds: int,
) -> PreviewPlan:
    """Build a ``PreviewPlan`` from assembled models + user instructions.

    Does NOT call any LLM. Returns instantly with a deterministic plan
    so the frontend can render a confirmation screen before the user
    commits to a real smart-generate run.

    Parameters
    ----------
    assembled
        Result of ``assemble_models_from_project``.
    instructions
        The user's natural-language request, same string that will be
        sent to ``POST /smart-generate`` if the preview is approved.
    max_cost_usd, max_runtime_seconds
        Server-clamped caps from the request. The cost estimate is
        clipped to ``max_cost_usd`` so the UI never shows a number above
        what the user allowed.
    """
    auxiliary_kinds = [
        entry["kind"] for entry in assembled.summary()["present"]
        if entry["kind"] != assembled.primary_kind
    ]

    target_generator, confidence = _predict_target_generator(
        assembled, instructions,
    )

    estimated_turns = _estimate_turns(assembled, instructions, target_generator)
    estimated_cost_usd = min(
        _estimate_cost_usd(estimated_turns, instructions),
        max_cost_usd,
    )
    estimated_duration_seconds = min(
        _estimate_duration_seconds(estimated_turns),
        max_runtime_seconds,
    )

    summary = _build_summary(
        assembled, target_generator, auxiliary_kinds,
    )

    notes = _build_notes(assembled, target_generator, instructions)

    return PreviewPlan(
        primary_kind=assembled.primary_kind,
        auxiliary_kinds=auxiliary_kinds,
        target_generator=target_generator,
        target_generator_confidence=confidence,
        summary=summary,
        estimated_turns=estimated_turns,
        estimated_cost_usd=estimated_cost_usd,
        estimated_duration_seconds=estimated_duration_seconds,
        notes=notes,
    )


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------


def _predict_target_generator(
    assembled: AssembledModels, instructions: str,
) -> tuple[str | None, float]:
    """Predict which BESSER generator would run in Phase 1.

    Returns ``(generator_name, confidence)``; confidence is a rough
    0.0–1.0 band:

    * 0.9+  — user explicitly named the framework in instructions
    * 0.7   — strong keyword (backend / API / fullstack) + model fit
    * 0.5   — default based on available models
    * 0.3   — "no match, LLM writes from scratch"

    A ``None`` generator name means smart-generation will skip Phase 1
    and build from scratch in Phase 2. That's not a failure — it's the
    expected path for "generate me a NestJS API" or for state-machine-
    only projects.
    """
    lower = instructions.lower()

    def _has(word: str) -> bool:
        return bool(re.search(r"\b" + re.escape(word) + r"\b", lower))

    # Explicit framework request we don't have a generator for
    for framework in _EXPLICIT_NO_GENERATOR_FRAMEWORKS:
        if framework in lower:
            return None, 0.9

    # Quantum — specialist generator always wins when circuit is present
    if assembled.quantum_circuit is not None:
        return "generate_qiskit", 0.9

    # Class diagrams unlock the big BESSER generators
    if assembled.domain_model is not None:
        if _has("django"):
            return "generate_django", 0.9
        if _has("fastapi") or _has("fast api"):
            if assembled.gui_model is not None:
                return "generate_web_app", 0.8
            return "generate_fastapi_backend", 0.9
        if _has("backend") or _has("api") or _has("rest"):
            if assembled.gui_model is not None:
                return "generate_web_app", 0.7
            return "generate_fastapi_backend", 0.7
        if _has("full-stack") or _has("fullstack") or _has("full stack") or _has("web app"):
            if assembled.gui_model is not None:
                return "generate_web_app", 0.8
            return "generate_fastapi_backend", 0.6
        if _has("pydantic"):
            return "generate_pydantic", 0.8
        if _has("sqlalchemy"):
            return "generate_sqlalchemy", 0.8
        if _has("sql"):
            return "generate_sql", 0.7
        # Defaults based on what's available
        if assembled.gui_model is not None:
            return "generate_web_app", 0.5
        return "generate_fastapi_backend", 0.5

    # No domain model → no BESSER generator will run in Phase 1. The
    # LLM drives everything from the primary non-class model.
    return None, 0.3


# ----------------------------------------------------------------------
# Budget estimation
# ----------------------------------------------------------------------


def _estimate_turns(
    assembled: AssembledModels,
    instructions: str,
    target_generator: str | None,
) -> int:
    """Rough turn count based on scope signals.

    Heuristic, not scientific: starts from a baseline and adds turns for
    each non-trivial concern (auth, docker, tests, auxiliary models,
    long instructions). Capped at the orchestrator's MAX_TURNS so we
    never promise more than the engine delivers.
    """
    lower = instructions.lower()
    turns = 6 if target_generator else 12   # from-scratch runs take longer

    # Scope keywords each add a few turns
    scope_hints = {
        "auth": 3, "jwt": 3, "login": 2, "oauth": 3,
        "docker": 2, "compose": 2, "deploy": 2,
        "test": 3, "pytest": 3,
        "dark mode": 1, "theme": 1,
        "postgres": 1, "postgresql": 1, "mysql": 1, "mongodb": 1,
        "email": 2, "smtp": 2,
        "payment": 3, "stripe": 3,
        "cors": 1,
    }
    for hint, bump in scope_hints.items():
        if hint in lower:
            turns += bump

    # Each auxiliary model likely needs a file or two
    if assembled.gui_model is not None:
        turns += 3
    if assembled.agent_model is not None:
        turns += 3
    if assembled.state_machines:
        turns += 2 * len(assembled.state_machines)

    # Longer instructions → more scope; rough linear-ish bump
    extra_words = max(0, len(instructions.split()) - 20) // 15
    turns += extra_words

    return min(int(turns), 80)   # matches LLMOrchestrator.MAX_TURNS


def _estimate_cost_usd(turns: int, instructions: str) -> float:
    """USD estimate assuming Sonnet-class pricing.

    Assumptions: each turn averages ~4k cached input + ~2k output tokens.
    With prompt caching, cached reads are ~10% the cost of fresh input.
    Sonnet pricing: ~$3/Mtok input, $15/Mtok output, $0.30/Mtok cache
    read. Back-of-envelope per-turn cost ≈ $0.033.
    """
    per_turn_usd = 0.033
    return max(0.01, turns * per_turn_usd)


def _estimate_duration_seconds(turns: int) -> int:
    """Each LLM turn ~10s average (API call + tool execution).

    Bumps to 15s/turn for the first few turns to account for cold-cache
    overhead. Floor at 30s so the UI doesn't show "instant" on trivial
    runs that still have to boot the orchestrator.
    """
    if turns <= 3:
        return max(30, turns * 15)
    return max(30, 45 + (turns - 3) * 10)


# ----------------------------------------------------------------------
# Prose helpers
# ----------------------------------------------------------------------


def _build_summary(
    assembled: AssembledModels,
    target_generator: str | None,
    auxiliary_kinds: list[str],
) -> str:
    primary_labels = {
        "class": "Class Diagram",
        "gui": "GUI Model",
        "agent": "Agent Model",
        "state_machine": "State Machine",
        "object": "Object Diagram",
        "quantum": "Quantum Circuit",
    }
    aux_labels = [primary_labels.get(k, k) for k in auxiliary_kinds]
    primary = primary_labels.get(assembled.primary_kind, assembled.primary_kind)

    if target_generator:
        gen_friendly = {
            "generate_fastapi_backend": "FastAPI backend (CRUD + ORM + schemas)",
            "generate_django": "Django project (models + views + admin)",
            "generate_web_app": "full-stack web app (FastAPI + React + Docker)",
            "generate_react": "React frontend",
            "generate_flutter": "Flutter app",
            "generate_pydantic": "Pydantic models",
            "generate_sqlalchemy": "SQLAlchemy ORM models",
            "generate_sql": "SQL CREATE TABLE statements",
            "generate_python_classes": "plain Python classes",
            "generate_java_classes": "Java classes",
            "generate_json_schema": "JSON Schema",
            "generate_qiskit": "Qiskit circuit code",
            "generate_rest_api": "FastAPI REST endpoints",
            "generate_rdf": "RDF / OWL vocabulary",
        }.get(target_generator, target_generator)
        base = f"Generate {gen_friendly} from {primary}"
    else:
        base = f"Generate code from {primary} (no deterministic generator — LLM writes from scratch)"

    if aux_labels:
        base += f"; also using {', '.join(aux_labels)}"
    return base


def _build_notes(
    assembled: AssembledModels,
    target_generator: str | None,
    instructions: str,
) -> list[str]:
    notes: list[str] = []
    if target_generator is None and assembled.domain_model is None:
        notes.append(
            "No domain model — the LLM will infer entity structure from "
            "the primary model. Consider adding a ClassDiagram if you want "
            "CRUD / ORM scaffolding."
        )
    if assembled.gui_model is not None and target_generator and "web_app" not in target_generator:
        notes.append(
            "A GUI model is present but the selected generator won't use "
            "it. To include a frontend, ask for 'full-stack' or 'web app'."
        )
    if assembled.state_machines:
        notes.append(
            f"{len(assembled.state_machines)} state machine(s) detected — "
            "transition guards / event handlers will be wired in Phase 2."
        )
    # Rough budget realism check
    word_count = len(instructions.split())
    if word_count > 120:
        notes.append(
            "Your instructions are long — the LLM may need more turns than "
            "estimated. Consider splitting into a core request + follow-up."
        )
    # Ceiling warnings
    turns = _estimate_turns(assembled, instructions, target_generator)
    if turns >= 60:
        notes.append(
            "Estimated turn count is high (>60). You may hit the runtime "
            "timeout before finishing — consider reducing scope."
        )
    return notes
