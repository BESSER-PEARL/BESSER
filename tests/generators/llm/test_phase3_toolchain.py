"""Tests for Phase 3 per-project toolchain validation.

Phase 3 used to surface ``tsc`` errors as ``warning``-severity issues
(so the LLM fix loop ignored them) and never invoked ``cargo`` or
``kotlinc`` at all. The result was that TypeScript / Rust / Kotlin
artifacts could leave Phase 3 with a 0/n per-project compile-pass
score on the bench even though Phase 0.5 wrote valid build configs.

The contract these tests pin down:

1. ``_classify_issue`` treats lines emitted by the new collectors
   (``tsc [...]:``, ``cargo [...]:``, ``kotlinc [...]:``) as
   ``blocker``-severity. The fix loop only acts on blockers, so this
   is what wires the toolchain output into the auto-fix flow.
2. The collectors soft-skip when the binary isn't on PATH (matches
   the bench's behaviour on hosts that don't have the toolchain).
3. ``_collect_tsc_issues`` actually parses ``tsc`` output and emits
   prefixed lines when given a project with a deliberate type error.
4. ``_invoke_phase3_fix_loop`` builds a prompt that contains the
   toolchain error verbatim and the appropriate re-run command, plus
   a high-salience reminder, before calling the LLM.
5. The outer toolchain-fix iteration cap is honoured: when the LLM
   can't make progress, the loop exits after at most
   ``_MAX_TOOLCHAIN_FIX_ITERATIONS`` rounds (no runaway turns).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from unittest.mock import patch

import pytest

from besser.generators.llm.llm_client import UsageTracker
from besser.generators.llm.orchestrator import (
    _MAX_TOOLCHAIN_FIX_ITERATIONS,
    LLMOrchestrator,
    ValidationIssue,
    _classify_issue,
)


# ---------------------------------------------------------------------------
# Classifier: per-project toolchain failures become blockers
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_tsc_bracketed_is_blocker(self) -> None:
        issue = _classify_issue(
            "tsc [.]: app/page.tsx(12,5): error TS2322: Type 'string' is not assignable to type 'number'."
        )
        assert issue.severity == "blocker"

    def test_cargo_bracketed_is_blocker(self) -> None:
        issue = _classify_issue(
            "cargo [.]: src/main.rs:8:5: error[E0308]: mismatched types"
        )
        assert issue.severity == "blocker"

    def test_kotlinc_bracketed_is_blocker(self) -> None:
        issue = _classify_issue(
            "kotlinc [src/main/kotlin]: app/Foo.kt:3:5: error: unresolved reference: Bar"
        )
        assert issue.severity == "blocker"

    def test_legacy_unbracketed_tsc_is_warning(self) -> None:
        """Legacy callers that emit ``tsc `` without a bracket must
        still get the old warning severity — guards against regressing
        any external integration that hand-builds these strings.
        """
        issue = _classify_issue("tsc says something off-format")
        assert issue.severity == "warning"

    def test_ruff_style_is_style(self) -> None:
        """Sanity: unrelated classifications still work — we didn't
        accidentally upgrade a style hint to a blocker.
        """
        issue = _classify_issue("ruff: foo.py:1:1: F401 'os' imported but unused")
        assert issue.severity == "style"


# ---------------------------------------------------------------------------
# Toolchain collector: real tsc invocation on a deliberate type error
# ---------------------------------------------------------------------------


# `tsc` isn't installed on every developer machine. The check has to
# be opt-in via a binary lookup so we don't fail CI on hosts that
# happen not to have Node + tsc set up. The bench host that drives
# the smart-gen runs always has it.
_TSC_BIN = shutil.which("tsc") or shutil.which("tsc.cmd")


def _build_orchestrator(tmp_path) -> LLMOrchestrator:
    """An orchestrator just rich enough to exercise the Phase 3 helpers.

    We pick state-machine-only because that bypasses the domain-model
    requirement and the constructor's primary-kind resolution still
    succeeds.
    """

    class _MockStateMachine:
        name = "DummySM"

    class _MockClient:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, system, messages, tools):  # pragma: no cover - not called
            raise AssertionError("These tests must not call the LLM")

    return LLMOrchestrator(
        llm_client=_MockClient(),
        state_machines=[_MockStateMachine()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )


def _write_broken_typescript_project(tmp_path) -> None:
    """A minimal but real TypeScript project with one obvious type error.

    Uses the strict, no-emit flags so ``tsc --noEmit`` will surface
    the error without trying to write JS output. The error
    (assigning a string to a ``number``) is the canonical TS2322 case
    every JS dev will recognise — keeps the assertion below stable
    across tsc versions.
    """
    tsconfig = {
        "compilerOptions": {
            "target": "es2022",
            "module": "esnext",
            "moduleResolution": "node",
            "strict": True,
            "noEmit": True,
            "skipLibCheck": True,
        },
        "include": ["**/*.ts"],
    }
    (tmp_path / "tsconfig.json").write_text(json.dumps(tsconfig), encoding="utf-8")
    (tmp_path / "broken.ts").write_text(
        textwrap.dedent(
            """\
            const x: number = "this is a string, not a number";
            console.log(x);
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.skipif(_TSC_BIN is None, reason="tsc not installed on this host")
def test_collect_tsc_issues_returns_structured_error(tmp_path) -> None:
    """Given a TS project with a TS2322, the collector returns at
    least one ``tsc [...]:`` line and the classifier promotes it to
    blocker. This is the lowest-fidelity end-to-end check that the
    toolchain plumbing works on a real ``tsc`` invocation.
    """
    _write_broken_typescript_project(tmp_path)
    orch = _build_orchestrator(tmp_path)

    raw = orch._collect_tsc_issues()
    assert raw, "expected at least one tsc error line"
    assert all(s.startswith("tsc [") for s in raw)
    assert any("error" in s.lower() for s in raw)

    # End-to-end through the classifier as the orchestrator wires it.
    issues = orch._collect_validation_issues()
    blockers = [i for i in issues if i.severity == "blocker"]
    assert any(
        i.message.startswith("tsc [") for i in blockers
    ), f"tsc error was not classified as blocker: {[i.message for i in issues]}"


def test_collect_tsc_issues_soft_skips_when_binary_missing(tmp_path) -> None:
    """When ``tsc`` isn't on PATH the collector must return an empty
    list — never raise. This mirrors the bench's soft-skip pattern so
    hosts without the toolchain installed still complete cleanly.
    """
    _write_broken_typescript_project(tmp_path)
    orch = _build_orchestrator(tmp_path)

    # Force a "binary not found" outcome by patching the resolver
    # rather than mutating PATH (cleaner across platforms).
    with patch("besser.generators.llm.orchestrator.shutil.which", return_value=None):
        assert orch._collect_tsc_issues() == []


def test_collect_cargo_issues_soft_skips_when_binary_missing(tmp_path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        "[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    orch = _build_orchestrator(tmp_path)
    with patch("besser.generators.llm.orchestrator.shutil.which", return_value=None):
        assert orch._collect_cargo_issues() == []


def test_collect_kotlinc_issues_soft_skips_when_binary_missing(tmp_path) -> None:
    module = tmp_path / "src" / "main" / "kotlin"
    module.mkdir(parents=True)
    (tmp_path / "build.gradle.kts").write_text("// nothing\n", encoding="utf-8")
    (module / "Main.kt").write_text("fun main() {}\n", encoding="utf-8")
    orch = _build_orchestrator(tmp_path)
    with patch("besser.generators.llm.orchestrator.shutil.which", return_value=None):
        assert orch._collect_kotlinc_issues() == []


# ---------------------------------------------------------------------------
# Fix loop: prompt construction and toolchain re-run wiring
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Captures the system + messages of every chat() call so the
    test can assert on the prompt the orchestrator built. Returns
    ``end_turn`` immediately so the fix loop doesn't loop on us.
    """

    model = "mock-model"

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")
        self.calls: list[tuple[str, list[dict]]] = []

    def chat(self, system, messages, tools):
        self.calls.append((system, [dict(m) for m in messages]))

        class _Block:
            type = "text"
            text = "Done"

        return {"stop_reason": "end_turn", "content": [_Block()]}


def _flatten_messages(messages: list[dict]) -> str:
    """Render every text fragment in the message list as one string."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
    return "\n".join(parts)


def test_fix_loop_prompt_contains_toolchain_error_and_rerun_command(tmp_path) -> None:
    """The fix-loop prompt must include the verbatim tsc error AND
    instruct the LLM to re-run ``npx tsc --noEmit`` after each edit.
    """
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )

    blocker = ValidationIssue(
        "blocker",
        "tsc [.]: app/page.tsx(5,3): error TS2322: Type 'string' is not assignable to type 'number'.",
    )
    orch._invoke_phase3_fix_loop([blocker], is_first_attempt=True)

    assert client.calls, "fix loop should have called the LLM at least once"
    system, messages = client.calls[0]
    text = _flatten_messages(messages)

    # Verbatim error line surfaced in the prompt
    assert "TS2322" in text
    # Explicit instruction to re-run the toolchain
    assert "npx tsc --noEmit" in text
    # High-salience reminder injected as a second user turn
    assert "<system-reminder>" in text
    # System prompt makes the contract explicit too
    assert "run_command" in system


def test_fix_loop_prompt_cargo_command(tmp_path) -> None:
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )
    blocker = ValidationIssue(
        "blocker",
        "cargo [.]: src/main.rs:8:5: error[E0308]: mismatched types",
    )
    orch._invoke_phase3_fix_loop([blocker], is_first_attempt=True)

    text = _flatten_messages(client.calls[0][1])
    assert "cargo check" in text
    assert "E0308" in text


def test_fix_loop_prompt_kotlinc_command(tmp_path) -> None:
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )
    blocker = ValidationIssue(
        "blocker",
        "kotlinc [src/main/kotlin]: Main.kt:3:5: error: unresolved reference: Foo",
    )
    orch._invoke_phase3_fix_loop([blocker], is_first_attempt=True)

    text = _flatten_messages(client.calls[0][1])
    assert "kotlinc" in text
    assert "unresolved reference" in text


def test_fix_loop_no_toolchain_reminder_when_only_python_blockers(tmp_path) -> None:
    """When the only blockers are non-toolchain (e.g. Python syntax
    error), the high-salience toolchain reminder must NOT fire — it
    would be noise for the LLM.
    """
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
    )
    blocker = ValidationIssue(
        "blocker", "Syntax error in app.py line 3: unexpected indent"
    )
    orch._invoke_phase3_fix_loop([blocker], is_first_attempt=True)

    text = _flatten_messages(client.calls[0][1])
    assert "<system-reminder>" not in text
    # The original blocker still surfaces in the prompt
    assert "Syntax error in app.py" in text


# ---------------------------------------------------------------------------
# Outer iteration cap
# ---------------------------------------------------------------------------


def test_toolchain_fix_iteration_cap_bounded(tmp_path) -> None:
    """When the LLM never resolves the blockers, the outer cap must
    keep total fix-loop invocations bounded by
    ``_MAX_TOOLCHAIN_FIX_ITERATIONS``.

    We stub ``_collect_validation_issues`` to return a constant
    blocker so the fix loop has something to chase but never makes
    progress. The early-exit ("no progress") branch should kick in
    after the second invocation, so the cap is at most
    ``_MAX_TOOLCHAIN_FIX_ITERATIONS``.
    """
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        auto_fix_issues=True,
        enable_tracing=False,
        enable_checkpointing=False,
    )

    persistent = [
        ValidationIssue(
            "blocker",
            "tsc [.]: a.ts(1,1): error TS2322: Type 'string' is not assignable to type 'number'.",
        )
    ]
    call_count = {"n": 0}

    def stub_collect():
        call_count["n"] += 1
        return list(persistent)

    with patch.object(orch, "_collect_validation_issues", side_effect=stub_collect), \
         patch.object(orch, "_create_snapshot"), \
         patch.object(orch, "_restore_snapshot"):
        orch._run_phase3_validation()

    # The LLM must not have been called more times than the outer cap
    # allows. Each attempt invokes chat() exactly once (since our
    # recording client always returns end_turn).
    assert len(client.calls) <= _MAX_TOOLCHAIN_FIX_ITERATIONS, (
        f"toolchain fix loop did not respect the iteration cap "
        f"(_MAX_TOOLCHAIN_FIX_ITERATIONS={_MAX_TOOLCHAIN_FIX_ITERATIONS}): "
        f"got {len(client.calls)} LLM calls"
    )
    # And we DID at least call the LLM once — otherwise the test
    # would pass vacuously.
    assert len(client.calls) >= 1


def test_toolchain_fix_loop_exits_when_blockers_clear(tmp_path) -> None:
    """When the first fix-loop pass clears the blockers, the outer
    loop must exit immediately rather than burning the rest of the
    iteration budget on a passing project.
    """
    client = _RecordingClient()
    orch = LLMOrchestrator(
        llm_client=client,
        state_machines=[type("SM", (), {"name": "x"})()],
        output_dir=str(tmp_path),
        auto_fix_issues=True,
        enable_tracing=False,
        enable_checkpointing=False,
    )

    first = [
        ValidationIssue(
            "blocker",
            "tsc [.]: a.ts(1,1): error TS2322: bad type",
        )
    ]
    states = [first, []]

    def stub_collect():
        return states.pop(0) if states else []

    with patch.object(orch, "_collect_validation_issues", side_effect=stub_collect), \
         patch.object(orch, "_create_snapshot"), \
         patch.object(orch, "_restore_snapshot"):
        orch._run_phase3_validation()

    # Exactly one LLM call: the first attempt cleared the blockers,
    # so the outer loop returned without spinning up a second attempt.
    assert len(client.calls) == 1
