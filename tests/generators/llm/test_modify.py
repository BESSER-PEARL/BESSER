"""Tests for incremental vibe-modify on ``LLMOrchestrator``.

``modify()`` seeds ``output_dir`` from a previous run's files and edits
them in place instead of rebuilding from scratch. These tests pin the
hard-separation contract:

  * the from-scratch ``run()`` path is byte-identical (Phase 1 still runs,
    the system prompt is unchanged when ``modify_mode`` is False);
  * ``modify()`` SKIPS Phase 1, so a customised file the LLM never touches
    survives untouched;
  * ``_seed_generator_files_from_recipe`` replays generator tags from the
    seed recipe and ``_save_recipe`` re-emits them as ``generator``;
  * the prompt directive appears only in modify mode.

The full LLM loop is driven with a scripted stub client (same shape the
real ``ClaudeLLMClient.chat`` returns) so Phase 2 runs end to end without
a provider.
"""

from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.generators.llm.prompt_builder import build_system_prompt


_MODIFY_DIRECTIVE = "You are MODIFYING an existing, working app."


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _Block:
    """Duck-typed content block matching what ``chat()`` returns."""

    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self):
        self.estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 1, "cost_usd": 0.0}


class _ScriptedClient:
    """LLM client stub that returns queued ``chat`` responses in order.

    No ``_client`` attribute, so the gap analyser and the structured
    generator selector both skip it. No ``chat_stream``, so Phase 2 takes
    the non-streaming path.
    """

    def __init__(self, responses):
        self.model = "test-model"
        self.usage = _Usage()
        self.max_tokens = 4096
        self._responses = list(responses)
        self.chat_calls = 0

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        self.chat_calls += 1
        if self._responses:
            return self._responses.pop(0)
        return {"stop_reason": "end_turn", "content": [_Block("text", text="done")]}


def _tool_use(name: str, tool_id: str, tool_input: dict) -> dict:
    return {
        "stop_reason": "tool_use",
        "content": [_Block("tool_use", name=name, id=tool_id, input=tool_input)],
    }


def _end_turn(text: str = "done") -> dict:
    return {"stop_reason": "end_turn", "content": [_Block("text", text=text)]}


def _make_domain() -> DomainModel:
    string_type = PrimitiveDataType("str")
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=string_type, is_id=True)}
    return DomainModel(name="Library", types={book})


def _make_orchestrator(tmp_path, client) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client,
        domain_model=_make_domain(),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


# ----------------------------------------------------------------------
# (a) From-scratch parity — prompt is byte-identical when modify_mode=False
# ----------------------------------------------------------------------


def test_build_system_prompt_default_is_byte_identical():
    """Omitting ``modify_mode`` and passing ``modify_mode=False`` must
    produce the exact same string — the from-scratch prompt is frozen."""
    domain = _make_domain()
    common = dict(
        domain_model=domain,
        gui_model=None,
        agent_model=None,
        inventory="Generated 5 files",
        instructions="Build a library API",
        max_turns=20,
    )
    default = build_system_prompt(**common)
    explicit_false = build_system_prompt(**common, modify_mode=False)
    assert default == explicit_false
    assert _MODIFY_DIRECTIVE not in default


def test_modify_mode_prepends_directive_only():
    """modify_mode=True prepends the directive; the rest of the prompt is
    the same body that from-scratch would emit (directive is additive)."""
    domain = _make_domain()
    common = dict(
        domain_model=domain,
        gui_model=None,
        agent_model=None,
        inventory="Generated 5 files",
        instructions="Add a dark theme",
        max_turns=20,
    )
    base = build_system_prompt(**common)
    modified = build_system_prompt(**common, modify_mode=True)
    assert modified != base
    assert modified.startswith(_MODIFY_DIRECTIVE)
    assert "prefer `modify_file` over `write_file`" in modified
    # Additive: the from-scratch body is still present verbatim after the
    # prepended directive.
    assert base in modified


def test_run_calls_phase1_but_modify_skips_it(tmp_path, monkeypatch):
    """The load-bearing separation: run() scaffolds via Phase 1; modify()
    never calls Phase 1 (so the generator can't overwrite seeded files)."""
    client = _ScriptedClient([])
    orch = _make_orchestrator(tmp_path, client)

    calls: list[str] = []
    monkeypatch.setattr(orch, "_run_phase1", lambda instr: calls.append("phase1"))
    # Neutralise the rest of the pipeline so this is a pure control-flow test.
    monkeypatch.setattr(orch, "_run_phase0_5_metadata", lambda instr: None)
    monkeypatch.setattr(orch, "_apply_adaptive_budget", lambda: None)
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)

    orch.run("build it")
    assert calls == ["phase1"]

    calls.clear()
    orch2 = _make_orchestrator(tmp_path, client)
    monkeypatch.setattr(orch2, "_run_phase1", lambda instr: calls.append("phase1"))
    monkeypatch.setattr(orch2, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch2, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch2, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch2, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch2, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch2, "_remove_snapshot", lambda: None)

    orch2.modify("add a feature")
    assert calls == []  # Phase 1 was never entered in modify mode
    assert orch2._modify_mode is True


# ----------------------------------------------------------------------
# (b) modify() does not overwrite a customised, untargeted file
# ----------------------------------------------------------------------


def test_modify_preserves_untouched_customised_file(tmp_path, monkeypatch):
    """Seed a customised file + a target file. Drive Phase 2 with a stub
    LLM that only edits the target. The customised file must survive
    byte-for-byte, and Phase 1 must never run."""
    # Gap analysis short-circuits to None with the stub client anyway, but
    # pin it so the test doesn't depend on that detail.
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.analyze_gaps_via_llm",
        lambda **kwargs: None,
    )

    seed = tmp_path / "seed"
    seed.mkdir()
    custom = seed / "custom_theme.css"
    custom_content = "body { background: #0d1117; color: #e6edf3; }\n"
    custom.write_text(custom_content, encoding="utf-8")
    (seed / "README.md").write_text("# Old Title\nWelcome.\n", encoding="utf-8")

    # Scripted: edit README.md, then finish.
    client = _ScriptedClient([
        _tool_use(
            "modify_file",
            "t1",
            {"path": "README.md", "old_text": "# Old Title", "new_text": "# New Title"},
        ),
        _end_turn("Updated the README."),
    ])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=_make_domain(),
        output_dir=str(seed),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )

    phase1_calls: list[str] = []
    monkeypatch.setattr(orch, "_run_phase1", lambda instr: phase1_calls.append("x"))

    result = orch.modify("Rename the README heading")

    assert result == str(seed)
    assert phase1_calls == []  # Phase 1 skipped
    # The untouched customised file is byte-identical.
    assert custom.read_text(encoding="utf-8") == custom_content
    # The targeted file WAS edited.
    assert "# New Title" in (seed / "README.md").read_text(encoding="utf-8")
    # Phase 2 ran to a clean end_turn.
    assert orch._phase2_exited_cleanly is True
    assert client.chat_calls == 2


# ----------------------------------------------------------------------
# (c) recipe helper loads generator tags; _save_recipe re-tags them
# ----------------------------------------------------------------------


def test_seed_generator_files_from_recipe_and_resave(tmp_path):
    seed = tmp_path / "seed"
    (seed / "backend").mkdir(parents=True)
    (seed / "frontend").mkdir()
    (seed / "backend" / "main.py").write_text("print('api')\n", encoding="utf-8")
    (seed / "frontend" / "app.css").write_text(".x{}\n", encoding="utf-8")

    seed_recipe = {
        "generator_used": "generate_fastapi_backend",
        "output_files": [
            {"path": "backend/main.py", "size": 12, "source": "generator"},
            {"path": "frontend/app.css", "size": 5, "source": "llm"},
        ],
    }
    (seed / ".besser_recipe.json").write_text(
        json.dumps(seed_recipe), encoding="utf-8"
    )

    client = _ScriptedClient([])
    orch = LLMOrchestrator(
        llm_client=client,
        domain_model=_make_domain(),
        output_dir=str(seed),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )

    orch._seed_generator_files_from_recipe()

    assert orch._seed_generator_used == "generate_fastapi_backend"
    assert "backend/main.py" in orch.executor._generator_files
    assert "frontend/app.css" not in orch.executor._generator_files

    # _save_recipe re-emits the generator tag for the seeded file.
    orch._save_recipe("add feature", elapsed=1.0)
    new_recipe = json.loads(
        (seed / ".besser_recipe.json").read_text(encoding="utf-8")
    )
    by_path = {f["path"]: f["source"] for f in new_recipe["output_files"]}
    assert by_path["backend/main.py"] == "generator"
    assert by_path["frontend/app.css"] == "llm"


def test_seed_recipe_missing_is_harmless(tmp_path):
    """No recipe → every file treated as llm; helper must not raise."""
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "main.py").write_text("x = 1\n", encoding="utf-8")

    orch = LLMOrchestrator(
        llm_client=_ScriptedClient([]),
        domain_model=_make_domain(),
        output_dir=str(seed),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    orch._seed_generator_files_from_recipe()  # must not raise
    assert orch._seed_generator_used is None
    assert orch.executor._generator_files == set()
