"""What the harness can check, the harness should check.

Run ys4gfj4v made **22 task_list calls against 9 edits**, twelve of them
refused: bookkeeping cost 2.4x the actual work. A task carrying a
deterministic verifier does not need the model to argue for it — the harness
can see whether the endpoint is implemented.

Closing those automatically after a landed edit also makes the loop
model-agnostic: a model that never learns the checklist protocol still gets
credit for what it built, instead of spending turns being refused.

Evidence-only tasks are untouched. Nothing here can judge them, so the model
must still cite its work.
"""

import pytest

from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator


class _Block:
    def __init__(self, name, tool_input, block_id="b1"):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input
        self.id = block_id


@pytest.fixture
def orchestrator(simple_library_book_model, tmp_path):
    class Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    (tmp_path / "handler.py").write_text("def cancel():\n    raise NotImplementedError\n",
                                         encoding="utf-8")
    return LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )


def _implemented(tmp_path):
    return "NotImplementedError" not in (tmp_path / "handler.py").read_text(encoding="utf-8")


def _edit(orchestrator, old, new):
    return orchestrator._execute_single_tool(
        _Block("modify_file", {"path": "handler.py", "old_text": old, "new_text": new}),
        turn=1)


def test_a_landed_edit_closes_the_item_it_satisfied(orchestrator, tmp_path):
    orchestrator.executor.set_tasks([
        {"text": "Implement cancel", "verify": lambda: _implemented(tmp_path)},
    ])
    assert len(orchestrator.executor.open_tasks()) == 1

    _edit(orchestrator, "    raise NotImplementedError", "    return True")

    assert orchestrator.executor.open_tasks() == []
    snapshot = orchestrator.executor.task_snapshot()[0]
    assert snapshot["done"] is True and snapshot["verification"] == "verified"


def test_an_unsatisfied_item_stays_open(orchestrator, tmp_path):
    orchestrator.executor.set_tasks([
        {"text": "Implement cancel", "verify": lambda: _implemented(tmp_path)},
    ])

    _edit(orchestrator, "def cancel():", "def cancel():  # touched")

    assert len(orchestrator.executor.open_tasks()) == 1


def test_an_evidence_only_item_is_never_auto_closed(orchestrator, tmp_path):
    """Nothing here can judge it, so the model must still cite its work."""
    orchestrator.executor.set_tasks([{"text": "Make the UI nicer"}])

    _edit(orchestrator, "    raise NotImplementedError", "    return True")

    assert len(orchestrator.executor.open_tasks()) == 1


def test_a_refused_edit_closes_nothing(orchestrator, tmp_path):
    """Only a landed write can have changed the answer."""
    closes = []
    orchestrator.executor.set_tasks([
        {"text": "Implement cancel", "verify": lambda: closes.append(1) or True},
    ])

    _edit(orchestrator, "text that is not in the file", "x")

    assert not closes, "verifiers ran after a refused edit"
    assert len(orchestrator.executor.open_tasks()) == 1


def test_a_raising_verifier_leaves_the_item_open(orchestrator):
    def boom():
        raise RuntimeError("probe blew up")

    orchestrator.executor.set_tasks([{"text": "Implement cancel", "verify": boom}])

    _edit(orchestrator, "    raise NotImplementedError", "    return True")

    assert len(orchestrator.executor.open_tasks()) == 1
