"""Duplicate checklist items cost turns, and the plan preamble cost more.

In a recorded run the model observed "tasks 15, 16 and 18 are
duplicated" and then worked through each copy. Nothing deduplicated the
checklist at either entry point - the gap analyser's output, or the model's
own ``task_list(action='add')``. The same run narrated a plan before every
batch of edits because the prompt said "Before making any edits, briefly
state your plan" - read as before ANY edit, not once per run.
"""

import json

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


class _Planner:
    """Looks like a real provider; returns a fixed task list."""
    _client = object()

    def __init__(self, tasks):
        self._reply = json.dumps(tasks)

    def chat(self, system, messages, tools):
        return {"content": [{"type": "text", "text": self._reply}]}


def _analyse(tasks):
    from besser.spec_driven_agent.planning.gap_analyzer import analyze_gaps_via_llm
    return analyze_gaps_via_llm(
        instructions="Hotel bookings.",
        generator_used="generate_fastapi_backend",
        domain_model=None,
        inventory="backend/main_api.py 1200",
        llm_client=_Planner(tasks),
    )


class TestGapAnalyserDedupes:

    def test_duplicates_collapse_in_first_seen_order(self):
        tasks = [
            "Implement Booking.cancel in booking_methods.py",
            "Add a /bookings/{id}/bill endpoint",
            "implement booking.cancel in booking_methods.py",        # case
            "Add  a /bookings/{id}/bill   endpoint.",               # whitespace + period
            "Implement Bill.registerPayment",
            "Implement Booking.cancel in booking_methods.py",        # exact
        ]
        assert _analyse(tasks) == [tasks[0], tasks[1], tasks[4]]

    def test_the_cap_applies_after_dedupe(self):
        """A planner that repeats itself must not crowd unique work out of
        the capped list."""
        from besser.spec_driven_agent.planning.gap_analyzer import _MAX_TASKS

        unique = [f"Implement method_{i} in methods.py" for i in range(_MAX_TASKS // 2)]
        repeated = unique * 3
        assert len(repeated) > _MAX_TASKS
        assert _analyse(repeated) == unique


class TestChecklistAddDedupes:

    def test_adding_an_existing_item_returns_its_id(self, tmp_path):
        executor = ToolExecutor(workspace=str(tmp_path))
        executor.set_tasks(["Implement Booking.cancel in booking_methods.py"])
        result = executor._task_list(
            {"action": "add", "text": "implement booking.cancel in  booking_methods.py."}
        )
        assert result["status"] == "exists"
        assert result["id"] == 1
        assert len(executor._task_list({"action": "list"})["tasks"]) == 1

    def test_a_new_item_is_still_added(self, tmp_path):
        executor = ToolExecutor(workspace=str(tmp_path))
        executor.set_tasks(["Implement Booking.cancel in booking_methods.py"])
        result = executor._task_list({"action": "add", "text": "Implement Bill.registerPayment"})
        assert result["status"] == "added"
        assert result["id"] == 2


class TestPlanOnce:

    def test_prompt_asks_for_the_plan_once_not_before_every_edit(self):
        from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt

        prompt = build_system_prompt(
            None, None, None, inventory="", instructions="Build the hotel app.", max_turns=10,
        )
        section = prompt[prompt.index("## Plan before you implement"):]
        section = section[:section.index("## Rules")]
        assert "Before making any edits" not in section
        assert "ONCE" in section
