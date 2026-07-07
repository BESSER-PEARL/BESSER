"""Tests for model-sync during vibe-MODIFY on ``LLMOrchestrator``.

When a ``modify()`` instruction implies genuinely-new domain entities
(e.g. "add authentication" → a ``User`` class), the orchestrator's
``_derive_and_apply_model_deltas`` step mutates ``self.domain_model`` in
place and re-serialises an updated project export onto
``self._updated_project_export`` so the GitHub push writes ``buml/`` from
the UPDATED model. These tests pin the contract:

  * (a) PARITY: the from-scratch ``run()`` path NEVER invokes the delta
    step; ``modify()`` does — the step is scoped to modify() only.
  * (b) a mocked LLM returning a new ``User`` class mutates the model,
    ``class_buml_to_json`` reflects it, and the updated export slots it
    into the active ClassDiagram entry.
  * (c) an empty delta leaves the model unchanged and the export None
    (so the push falls back to the request's projectExport).
  * (d) a duplicate / invalid delta is skipped with no crash.

The delta step is gated on a real-looking client (``hasattr(client,
'_client')``, the same gate the generator selector uses), so unit tests
that drive the full ``modify()`` loop with a scripted client stub are
unaffected — that gate is exercised here too.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.orchestrator import LLMOrchestrator
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
)


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


class _DeltaClient:
    """Mock LLM whose ``chat`` returns a scripted forced-tool response.

    Carries a ``_client`` attribute so the model-sync gate opens, and a
    ``**kwargs`` ``chat`` signature so ``_client_supports_structured_chat``
    returns True (the structured path is taken).
    """

    def __init__(self, new_classes):
        self._client = object()  # opens the hasattr(_client) gate
        self.model = "test-model"
        self.planning_model = None
        self.usage = _Usage()
        self.max_tokens = 4096
        self._new_classes = new_classes
        self.chat_calls = 0

    def chat(self, **kwargs):
        self.chat_calls += 1
        return {
            "stop_reason": "tool_use",
            "content": [
                _Block(
                    "tool_use",
                    name="derive_model_deltas",
                    id="d1",
                    input={"new_classes": self._new_classes},
                )
            ],
        }


def _make_domain() -> DomainModel:
    string_type = PrimitiveDataType("str")
    book = Class(name="Book")
    book.attributes = {Property(name="title", type=string_type, is_id=True)}
    return DomainModel(name="Library", types={book})


def _source_export() -> dict:
    """Minimal ProjectInput-shaped export with one ClassDiagram entry."""
    return {
        "id": "p1",
        "type": "BesserProject",
        "name": "P",
        "createdAt": "2026-01-01T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [
                {
                    "id": "cd1",
                    "title": "Library",
                    "model": {
                        "type": "ClassDiagram",
                        "elements": {},
                        "relationships": {},
                    },
                }
            ]
        },
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _make_orchestrator(tmp_path, client, source_export=None) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=client,
        domain_model=_make_domain(),
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
        source_project_export=source_export,
    )


def _class_names(domain_model) -> list[str]:
    return sorted(c.name for c in domain_model.get_classes())


# ----------------------------------------------------------------------
# (a) PARITY — run() never invokes the delta step; modify() does
# ----------------------------------------------------------------------


def test_run_does_not_invoke_model_deltas_but_modify_does(tmp_path, monkeypatch):
    """The load-bearing safety contract: the from-scratch run() path must
    never touch model-sync, while modify() must call it exactly once."""
    calls: list[str] = []

    # ---- run(): every phase neutralised; assert the delta step is never
    # invoked (proves from-scratch generation is untouched). ----
    orch = _make_orchestrator(tmp_path, _DeltaClient([]))
    monkeypatch.setattr(
        orch, "_derive_and_apply_model_deltas",
        lambda instr: calls.append("delta"),
    )
    monkeypatch.setattr(orch, "_run_phase1", lambda instr: None)
    monkeypatch.setattr(orch, "_run_phase0_5_metadata", lambda instr: None)
    monkeypatch.setattr(orch, "_apply_adaptive_budget", lambda: None)
    monkeypatch.setattr(orch, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch, "_remove_snapshot", lambda: None)

    orch.run("build a library API")
    assert calls == []  # run() never syncs the model

    # ---- modify(): assert the delta step IS invoked (once). ----
    orch2 = _make_orchestrator(tmp_path, _DeltaClient([]))
    monkeypatch.setattr(
        orch2, "_derive_and_apply_model_deltas",
        lambda instr: calls.append("delta"),
    )
    monkeypatch.setattr(
        "besser.generators.llm.orchestrator.build_inventory",
        lambda *a, **k: "",
    )
    monkeypatch.setattr(orch2, "_seed_generator_files_from_recipe", lambda: None)
    monkeypatch.setattr(orch2, "_validate_phase1_output", lambda: [])
    monkeypatch.setattr(orch2, "_run_phase2", lambda instr, extra_issues=None: None)
    monkeypatch.setattr(orch2, "_create_snapshot", lambda: None)
    monkeypatch.setattr(orch2, "_run_phase3_validation", lambda: None)
    monkeypatch.setattr(orch2, "_save_recipe", lambda instr, elapsed: None)
    monkeypatch.setattr(orch2, "_remove_snapshot", lambda: None)

    orch2.modify("add authentication")
    assert calls == ["delta"]  # modify() synced exactly once


# ----------------------------------------------------------------------
# (b) new class → model + export gain it
# ----------------------------------------------------------------------


def test_new_class_applied_and_slotted_into_export(tmp_path):
    client = _DeltaClient([
        {
            "name": "User",
            "attributes": [
                {"name": "username", "type": "str"},
                {"name": "password", "type": "str"},
                {"name": "role", "type": "str"},
            ],
        }
    ])
    orch = _make_orchestrator(tmp_path, client, source_export=_source_export())

    orch._derive_and_apply_model_deltas("add authentication")

    # Model mutated in place.
    assert "User" in _class_names(orch.domain_model)
    assert client.chat_calls == 1

    # The new User class carries its attributes.
    user = next(c for c in orch.domain_model.get_classes() if c.name == "User")
    assert {a.name for a in user.attributes} == {"username", "password", "role"}

    # class_buml_to_json reflects the new class.
    serialized = class_buml_to_json(orch.domain_model)
    class_element_names = {
        e.get("name")
        for e in serialized["elements"].values()
        if e.get("type") == "Class"
    }
    assert "User" in class_element_names
    assert "Book" in class_element_names

    # Updated export slots the re-serialised diagram into the active
    # ClassDiagram entry.
    assert orch._updated_project_export is not None
    entry_model = (
        orch._updated_project_export["diagrams"]["ClassDiagram"][0]["model"]
    )
    exported_names = {
        e.get("name")
        for e in entry_model["elements"].values()
        if e.get("type") == "Class"
    }
    assert "User" in exported_names


def test_domain_model_reference_shared_with_executor(tmp_path):
    """In-place mutation must be visible to the ToolExecutor (same ref)."""
    client = _DeltaClient([{"name": "User", "attributes": []}])
    orch = _make_orchestrator(tmp_path, client, source_export=_source_export())

    assert orch.domain_model is orch.executor.domain_model
    orch._derive_and_apply_model_deltas("add auth")
    # The executor sees the same, mutated object — not a replacement.
    assert orch.domain_model is orch.executor.domain_model
    assert "User" in _class_names(orch.executor.domain_model)


# ----------------------------------------------------------------------
# (c) empty delta → no change, export stays None
# ----------------------------------------------------------------------


def test_empty_delta_leaves_model_and_export_untouched(tmp_path):
    client = _DeltaClient([])
    orch = _make_orchestrator(tmp_path, client, source_export=_source_export())

    orch._derive_and_apply_model_deltas("make the UI responsive")

    assert _class_names(orch.domain_model) == ["Book"]
    assert orch._updated_project_export is None  # push falls back


# ----------------------------------------------------------------------
# (d) duplicate / invalid delta → skipped, no crash
# ----------------------------------------------------------------------


def test_duplicate_and_invalid_deltas_are_skipped(tmp_path):
    client = _DeltaClient([
        {"name": "Book"},           # collides with existing class
        {"name": "has space"},      # invalid name (space)
        {"name": "bad-hyphen"},     # invalid name (hyphen)
        {"name": ""},               # invalid name (empty)
        "not-a-dict",               # malformed entry
    ])
    orch = _make_orchestrator(tmp_path, client, source_export=_source_export())

    orch._derive_and_apply_model_deltas("add a book")  # must not raise

    assert _class_names(orch.domain_model) == ["Book"]  # unchanged
    assert orch._updated_project_export is None  # nothing added


def test_valid_class_added_even_when_mixed_with_a_duplicate(tmp_path):
    """A duplicate alongside a valid new class: skip the dup, add the rest."""
    client = _DeltaClient([
        {"name": "Book"},  # duplicate → skipped
        {"name": "User", "attributes": [{"name": "email", "type": "str"}]},
    ])
    orch = _make_orchestrator(tmp_path, client, source_export=_source_export())

    orch._derive_and_apply_model_deltas("add users")

    assert _class_names(orch.domain_model) == ["Book", "User"]
    assert orch._updated_project_export is not None


# ----------------------------------------------------------------------
# Gate: a test/mock client without ``_client`` is skipped entirely
# ----------------------------------------------------------------------


def test_mock_client_without_underscore_client_is_skipped(tmp_path):
    """A duck-typed client lacking ``_client`` must not be called — this is
    what keeps the existing full-modify() loop tests deterministic."""

    class _NoUnderscoreClient:
        def __init__(self):
            self.model = "m"
            self.planning_model = None

        def chat(self, **kwargs):
            raise AssertionError("model-sync must skip mock clients")

    orch = _make_orchestrator(
        tmp_path, _NoUnderscoreClient(), source_export=_source_export()
    )
    orch._derive_and_apply_model_deltas("add auth")  # must not call chat
    assert orch._updated_project_export is None
    assert _class_names(orch.domain_model) == ["Book"]


def test_no_domain_model_skips_without_error(tmp_path):
    """A GUI/agent-only modify (domain_model is None) is a clean no-op."""
    client = _DeltaClient([{"name": "User"}])

    class _GuiModel:
        modules = []

    orch = LLMOrchestrator(
        llm_client=client,
        gui_model=_GuiModel(),
        primary_kind="gui",
        output_dir=str(tmp_path),
        enable_tracing=False,
        enable_checkpointing=False,
        enable_toolchain_validation=False,
        source_project_export=_source_export(),
    )
    assert orch.domain_model is None
    orch._derive_and_apply_model_deltas("add auth")  # must not raise
    assert orch._updated_project_export is None
    assert client.chat_calls == 0
