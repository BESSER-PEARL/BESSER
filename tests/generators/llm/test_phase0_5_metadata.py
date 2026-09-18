"""Tests for Phase 0.5 — pre-generated stack metadata.

Phase 0.5 sits between Phase 1 (deterministic generators) and Phase 2
(LLM customise loop). It writes a minimal but valid build-metadata
file for stacks BESSER doesn't have a deterministic generator for —
``tsconfig.json`` for Next.js, ``Cargo.toml`` for Rust,
``build.gradle.kts`` + ``settings.gradle.kts`` for Kotlin / Spring.

The contract these tests pin down:

1. The substring detector recognises the supported stacks and
   ignores the Python family (where BESSER already has a Phase 1
   generator that owns the manifest).
2. ``pre_generate_metadata`` emits valid JSON / TOML / Gradle Kotlin
   that can be parsed by a downstream toolchain.
3. The orchestrator invokes Phase 0.5 BEFORE Phase 2 when the target
   stack is non-Python — and STAYS OUT OF THE WAY when a deterministic
   generator already ran.
4. ``_inventory`` is populated with a note about the pre-generated
   files so the customise loop knows not to rewrite them.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.stack_metadata import (
    detect_stack,
    files_for,
    pre_generate_metadata,
    stack_label,
    supported_stacks,
)


# ---------------------------------------------------------------------------
# Detection — pure function, no I/O
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instructions,expected",
    [
        # Positive matches across naming variants
        ("Build a Next.js todo app", "nextjs"),
        ("Build a NEXTJS application with TypeScript", "nextjs"),
        ("next js with tailwind", "nextjs"),
        ("Build a Rust web server using Axum", "rust"),
        ("Standard cargo layout please", "rust"),
        ("rocket + serde", "rust"),
        ("Use actix for the API", "rust"),
        ("Build a Kotlin Spring Boot REST API", "kotlin_spring"),
        ("Kotlin web service", "kotlin_spring"),
        ("springboot starter project", "kotlin_spring"),
        # Negative matches: BESSER has Phase 1 generators for these
        ("Build a FastAPI backend with JWT", None),
        ("Build a Django project for a library", None),
        ("Generate Pydantic v2 models", None),
        ("SQLAlchemy 2.0 declarative classes", None),
        ("Build a Flask app", None),
        # Negative matches: no template yet (Go, Ruby, Express, Java/Maven)
        ("Build a Go web service with Gin", None),
        ("Ruby on Rails 7 hotel booking", None),
        ("Express + TypeScript API", None),
        # Edge cases
        ("", None),
        ("   ", None),
        ("Make a plain Python class library", None),
    ],
)
def test_detect_stack(instructions: str, expected: str | None) -> None:
    assert detect_stack(instructions) == expected


def test_detect_stack_python_family_wins_over_other_keywords() -> None:
    """A FastAPI request with a stray "Rust" mention should still
    resolve to None — BESSER's FastAPI generator owns the manifest.
    """
    assert detect_stack("Build a FastAPI app inspired by Rust services") is None


def test_supported_stacks_label_each() -> None:
    """Every supported stack must expose a friendly label."""
    for stack_id in supported_stacks():
        label = stack_label(stack_id)
        assert isinstance(label, str) and label, f"empty label for {stack_id}"


# ---------------------------------------------------------------------------
# File emission — temp-dir round trip
# ---------------------------------------------------------------------------


def test_pre_generate_nextjs_writes_valid_files(tmp_path) -> None:
    written = pre_generate_metadata("nextjs", str(tmp_path))
    assert set(written) == {"tsconfig.json", "next-env.d.ts", "package.json"}

    tsconfig = tmp_path / "tsconfig.json"
    pkg = tmp_path / "package.json"
    assert tsconfig.is_file()
    assert pkg.is_file()

    # Both JSON files must parse cleanly.
    tsconfig_doc = json.loads(tsconfig.read_text(encoding="utf-8"))
    pkg_doc = json.loads(pkg.read_text(encoding="utf-8"))

    # Spot-check the load-bearing fields the bench's compile check
    # cares about so a refactor of the template can't silently drop them.
    assert tsconfig_doc["compilerOptions"]["strict"] is True
    assert tsconfig_doc["compilerOptions"]["jsx"] == "preserve"
    assert tsconfig_doc["compilerOptions"]["moduleResolution"] == "bundler"
    assert pkg_doc["scripts"]["dev"] == "next dev"
    assert "next" in pkg_doc["dependencies"]
    assert "react" in pkg_doc["dependencies"]
    assert "typescript" in pkg_doc["devDependencies"]


def test_pre_generate_rust_writes_cargo_toml(tmp_path) -> None:
    written = pre_generate_metadata("rust", str(tmp_path))
    assert written == ["Cargo.toml"]

    cargo = tmp_path / "Cargo.toml"
    content = cargo.read_text(encoding="utf-8")
    # Cheap sanity checks — full TOML parsing isn't necessary; we just
    # want to be confident the toolchain will find what it needs.
    assert "[package]" in content
    assert "edition = \"2021\"" in content
    assert "[dependencies]" in content
    assert "axum" in content
    assert "tokio" in content
    assert "serde" in content


def test_pre_generate_kotlin_spring_writes_both_gradle_files(tmp_path) -> None:
    written = pre_generate_metadata("kotlin_spring", str(tmp_path))
    assert set(written) == {"build.gradle.kts", "settings.gradle.kts"}

    build = (tmp_path / "build.gradle.kts").read_text(encoding="utf-8")
    settings = (tmp_path / "settings.gradle.kts").read_text(encoding="utf-8")
    assert "org.springframework.boot" in build
    assert "kotlin(\"jvm\")" in build
    assert "rootProject.name" in settings


def test_pre_generate_unknown_stack_writes_nothing(tmp_path) -> None:
    assert pre_generate_metadata("zigfile", str(tmp_path)) == []
    assert list(tmp_path.iterdir()) == []


def test_pre_generate_is_idempotent_and_additive(tmp_path) -> None:
    """A second call must not overwrite existing files (strictly additive).

    Important: if the customise loop already wrote a richer manifest
    (between Phase 0.5 and a later resume), we don't want to clobber it.
    """
    first = pre_generate_metadata("nextjs", str(tmp_path))
    assert first  # something was written

    # Mutate the file so we can detect a clobber.
    pkg_path = tmp_path / "package.json"
    pkg_path.write_text("{\"name\": \"changed-by-user\"}", encoding="utf-8")

    second = pre_generate_metadata("nextjs", str(tmp_path))
    assert "package.json" not in second  # not re-written
    assert json.loads(pkg_path.read_text(encoding="utf-8")) == {
        "name": "changed-by-user"
    }


def test_files_for_returns_tuple_for_known_stacks() -> None:
    for stack_id in supported_stacks():
        entries = files_for(stack_id)
        assert isinstance(entries, tuple)
        assert entries, f"no files registered for {stack_id}"
        for rel_path, content in entries:
            assert isinstance(rel_path, str) and rel_path
            assert isinstance(content, str) and content


# ---------------------------------------------------------------------------
# Orchestrator wiring — Phase 0.5 fires at the right time
# ---------------------------------------------------------------------------


class _MockUsage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _MockClient:
    """Bare-minimum LLM client stub. The orchestrator's Phase 0.5 path
    never calls the LLM — we patch out Phase 2 / Phase 3 to confirm.
    """

    def __init__(self) -> None:
        self.model = "test-model"
        self.usage = _MockUsage()

    def chat(self, *args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("Phase 0.5 must not invoke the LLM")


class _MockStateMachine:
    def __init__(self, name: str = "OrderSM") -> None:
        self.name = name


def _build_orchestrator(tmp_path) -> LLMOrchestrator:
    """Construct an orchestrator that won't crash without a domain model.

    We use a state-machine-only project because:
      1. Phase 1 short-circuits ("no domain_model or quantum_circuit"),
         so we exercise the exact path Phase 0.5 was designed for.
      2. State machines are accepted by the orchestrator's primary-kind
         resolver, so no ValueError on construction.
    """
    return LLMOrchestrator(
        llm_client=_MockClient(),
        state_machines=[_MockStateMachine()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )


def test_phase0_5_runs_for_nextjs(tmp_path) -> None:
    """End-to-end: Phase 0.5 must emit tsconfig.json + package.json
    when the user asks for a Next.js app.

    We stub Phase 2 / Phase 3 so the test stays offline.
    """
    orch = _build_orchestrator(tmp_path)

    with patch.object(orch, "_run_phase2", lambda *a, **kw: None), \
         patch.object(orch, "_run_phase3_validation", lambda *a, **kw: None), \
         patch.object(orch, "_create_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_remove_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_save_recipe", lambda *a, **kw: None):
        orch.run("Build a Next.js 14 todo app with TypeScript and Tailwind")

    assert (tmp_path / "tsconfig.json").is_file()
    assert (tmp_path / "package.json").is_file()
    assert (tmp_path / "next-env.d.ts").is_file()

    # The orchestrator should remember which stack was detected so the
    # recipe / inventory can reference it.
    assert orch._phase0_5_stack == "nextjs"
    assert "tsconfig.json" in orch._phase0_5_files
    # Inventory must mention the pre-generated files so the customise
    # loop's system prompt surfaces them to the LLM.
    assert "tsconfig.json" in orch._inventory
    assert "package.json" in orch._inventory


def test_phase0_5_runs_for_rust(tmp_path) -> None:
    orch = _build_orchestrator(tmp_path)

    with patch.object(orch, "_run_phase2", lambda *a, **kw: None), \
         patch.object(orch, "_run_phase3_validation", lambda *a, **kw: None), \
         patch.object(orch, "_create_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_remove_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_save_recipe", lambda *a, **kw: None):
        orch.run("Build a Rust web server using Axum and tokio")

    assert (tmp_path / "Cargo.toml").is_file()
    assert orch._phase0_5_stack == "rust"


def test_phase0_5_runs_for_kotlin_spring(tmp_path) -> None:
    orch = _build_orchestrator(tmp_path)

    with patch.object(orch, "_run_phase2", lambda *a, **kw: None), \
         patch.object(orch, "_run_phase3_validation", lambda *a, **kw: None), \
         patch.object(orch, "_create_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_remove_snapshot", lambda *a, **kw: None), \
         patch.object(orch, "_save_recipe", lambda *a, **kw: None):
        orch.run("Build a Kotlin Spring Boot REST API for a Todo app")

    assert (tmp_path / "build.gradle.kts").is_file()
    assert (tmp_path / "settings.gradle.kts").is_file()
    assert orch._phase0_5_stack == "kotlin_spring"


def test_phase0_5_skipped_when_phase1_ran(tmp_path) -> None:
    """If a deterministic Python generator already ran, Phase 0.5 must
    stay out of the way — even if the instructions happen to mention a
    non-Python keyword. Python paths must be byte-identical to today.
    """
    orch = _build_orchestrator(tmp_path)
    # Simulate the post-Phase-1 state where a generator did run.
    orch._generator_used = "generate_fastapi_backend"

    orch._run_phase0_5_metadata("Build a FastAPI backend, inspired by Rust")

    # Nothing was emitted because a Phase 1 generator owned the
    # manifest already.
    assert orch._phase0_5_stack is None
    assert orch._phase0_5_files == []
    # And no Rust artefact landed in the output dir.
    assert not (tmp_path / "Cargo.toml").exists()


def test_phase0_5_no_op_for_unknown_stack(tmp_path) -> None:
    """Go / Ruby / Express don't have templates yet — Phase 0.5 must
    leave the directory empty so the customise loop's behaviour for
    those stacks is unchanged.
    """
    orch = _build_orchestrator(tmp_path)
    orch._run_phase0_5_metadata("Build a Go web service with Gin")

    assert orch._phase0_5_stack is None
    assert orch._phase0_5_files == []
    assert list(tmp_path.iterdir()) == []


def test_phase0_5_preserves_existing_files(tmp_path) -> None:
    """If a file already exists on disk (e.g. a previous resume),
    Phase 0.5 must not clobber it. The whole point is to be a floor,
    not a ceiling.
    """
    pre = tmp_path / "tsconfig.json"
    pre.write_text('{"compilerOptions": {"target": "es2015"}}', encoding="utf-8")

    orch = _build_orchestrator(tmp_path)
    orch._run_phase0_5_metadata("Build a Next.js app with TypeScript")

    # The user's tsconfig.json is preserved
    assert json.loads(pre.read_text(encoding="utf-8")) == {
        "compilerOptions": {"target": "es2015"}
    }
    # …but the other files (which didn't exist) ARE written
    assert (tmp_path / "package.json").is_file()
    assert (tmp_path / "next-env.d.ts").is_file()
