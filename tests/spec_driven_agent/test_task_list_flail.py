"""Refused checklist bookkeeping is a loop; successful bookkeeping is not.

``task_list`` sits in ``_READONLY_TOOLS``, so loop detection ignored it
entirely — deliberately, because closing several items back to back is what
the end_turn gate asks for. But run n_6i2i5r spent turns 12 through 20
marking tasks 9 to 17 done, one per turn, every one refused for supplying no
evidence at all. Nine turns, nine identical lessons, no guard.

``_is_stuck`` only fires when every call in its window failed, so a healthy
batch of successful closes still never trips it.
"""

import json

import pytest

from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator


class _Block:
    def __init__(self, name, tool_input, block_id="t1"):
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

    built = LLMOrchestrator(
        llm_client=Client(), domain_model=simple_library_book_model,
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
    )
    for n in range(1, 8):
        built.executor.execute_typed("task_list", {"action": "add", "text": f"task {n}"})
    return built


def _close_without_evidence(orchestrator, task_id):
    """Exactly what the live run did: 'done' with no evidence at all."""
    result = orchestrator._execute_single_tool(
        _Block("task_list", {"action": "done", "id": task_id}, f"t{task_id}"), turn=task_id)
    return json.loads(result["content"]) if isinstance(result.get("content"), str) else result


def test_repeated_refused_closes_are_called_out(orchestrator):
    warnings = []
    for task_id in range(1, orchestrator._LOOP_THRESHOLD + 1):
        payload = _close_without_evidence(orchestrator, task_id)
        if "warning" in payload:
            warnings.append(payload["warning"])

    assert warnings, "nine identical refusals in the live run raised nothing"
    assert "bookkeeping, not progress" in warnings[-1]
    assert "task_list calls in a row were refused" in warnings[-1]


def test_the_first_few_refusals_are_not_nagged(orchestrator):
    payload = _close_without_evidence(orchestrator, 1)

    assert "warning" not in payload


def test_successful_bookkeeping_never_trips_the_guard(orchestrator, tmp_path):
    """Closing several verified items back to back is the intended shape."""
    (tmp_path / "impl.py").write_text("def handler():\n    return True\n", encoding="utf-8")
    orchestrator.executor.execute_typed("read_file", {"path": "impl.py"})

    payloads = []
    for task_id in range(1, orchestrator._LOOP_THRESHOLD + 2):
        result = orchestrator._execute_single_tool(
            _Block("task_list", {
                "action": "done", "id": task_id, "existing": True,
                "evidence": [{"id": task_id, "path": "impl.py",
                              "quote": "def handler():"}],
            }, f"s{task_id}"), turn=task_id)
        payloads.append(json.loads(result["content"]))

    assert not any("warning" in p for p in payloads), payloads
