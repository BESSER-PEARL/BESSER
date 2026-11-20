"""The lab agent's tool loop, driven by a scripted client against real tools in a temporary folder."""

import json
import threading
import time

import pytest

import besser.spec_driven_agent.agent.compaction as compaction
from besser.spec_driven_agent.errors import InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.lab.contracts import ExtraTool, LabCallbacks, LabLimits
from besser.spec_driven_agent.lab.loop import MAX_TRUNCATION_RETRIES, TRUNCATION_PROMPT, run_lab_loop
from besser.spec_driven_agent.lab.tooling import LabToolExecutor
from tests.spec_driven_agent.lab.conftest import Block, end_turn, tool_use

REQUEST = "Add a note element to the class diagram."


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    return tmp_path


@pytest.fixture
def make_executor(workspace):
    made = []

    def make(cls=LabToolExecutor, **kwargs):
        executor = cls(str(workspace), **kwargs)
        made.append(executor)
        return executor
    yield make
    for executor in made:
        executor.close()


def _run(client, executor, *, limits=LabLimits(), callbacks=None, messages=None, **kwargs):
    messages = [{"role": "user", "content": REQUEST}] if messages is None else messages
    result = run_lab_loop(client=client, executor=executor, system="You change BESSER.", messages=messages,
                          limits=limits, callbacks=callbacks or LabCallbacks(), **kwargs)
    return result, messages


def _results(message):
    return [block for block in message["content"] if block["type"] == "tool_result"]


def test_reads_writes_and_completes_with_the_final_text(workspace, scripted_client, make_executor):
    progress = []
    client = scripted_client([
        tool_use("read_file", {"path": "a.py"}),
        tool_use("write_file", {"path": "b.py", "content": "y = 2\n"}),
        end_turn("Added b.py."),
    ])
    result, messages = _run(client, make_executor(),
                            callbacks=LabCallbacks(on_progress=lambda *event: progress.append(event)))
    assert (result.stop_reason, result.turns, result.final_text) == ("completed", 3, "Added b.py.")
    assert (workspace / "b.py").read_text() == "y = 2\n"
    assert [(call["tool"], call["success"]) for call in result.tool_calls] == [("read_file", True),
                                                                                ("write_file", True)]
    assert [event[1:3] for event in progress] == [("read_file", "executing"), ("read_file", "done"),
                                                  ("write_file", "executing"), ("write_file", "done")]
    # Each call is answered in the next message, paired by id; the conversation is kept for the caller.
    call = messages[1]["content"][0]
    answer = _results(messages[2])[0]
    assert answer["tool_use_id"] == call.id and answer["is_error"] is False
    assert "x = 1" in json.loads(answer["content"])["content"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert client.calls[0][2] == [s["name"] for s in make_executor().schemas]


def test_reading_for_many_turns_is_not_cut_short(scripted_client, make_executor):
    client = scripted_client([tool_use("read_file", {"path": "a.py"}) for _ in range(40)] + [end_turn()])
    result, _ = _run(client, make_executor(read_only=True))
    assert (result.stop_reason, result.turns) == ("completed", 41)


def test_finish_tool_ends_the_loop_once_it_succeeds(scripted_client, make_executor):
    plans = []

    def submit_plan(args):
        if not args.get("title"):
            raise ValueError("title is required")
        plans.append(args["title"])
        return "Plan recorded."

    schema = {"name": "submit_plan", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    executor = make_executor(read_only=True, extra_tools=[ExtraTool(schema, submit_plan, writes=False)])
    client = scripted_client([
        tool_use("submit_plan", {}),
        tool_use("read_file", {"path": "a.py"}, ("submit_plan", {"title": "Note element"})),
    ])
    result, messages = _run(client, executor, finish_tool="submit_plan")
    assert (result.stop_reason, result.turns, plans) == ("completed", 2, ["Note element"])
    assert _results(messages[2])[0]["is_error"] is True
    # Every call of the last turn still has its result, so the conversation stays valid.
    assert [r["is_error"] for r in _results(messages[-1])] == [False, False]


def test_read_only_step_refuses_a_write_and_carries_on(workspace, scripted_client, make_executor):
    client = scripted_client([tool_use("write_file", {"path": "b.py", "content": "y = 2\n"}), end_turn("Planned.")])
    result, messages = _run(client, make_executor(read_only=True))
    assert result.stop_reason == "completed"
    assert not (workspace / "b.py").exists()
    answer = _results(messages[2])[0]
    assert answer["is_error"] is True and "read-only" in answer["content"]
    assert result.tool_calls[0]["success"] is False


def test_turn_cap(scripted_client, make_executor):
    client = scripted_client([tool_use("read_file", {"path": "a.py"}) for _ in range(5)])
    result, _ = _run(client, make_executor(), limits=LabLimits(max_turns=2))
    assert (result.stop_reason, result.turns, len(client.calls)) == ("max_turns", 2, 2)


def test_cost_cap_before_a_call_spends_nothing(scripted_client, make_executor):
    client = scripted_client([end_turn()])
    client.usage.estimated_cost = 5.0
    result, _ = _run(client, make_executor(), limits=LabLimits(max_cost_usd=5.0))
    assert (result.stop_reason, client.calls) == ("cost_cap", [])


def test_cost_cap_after_a_call_runs_none_of_its_tools(workspace, scripted_client, make_executor):
    client = scripted_client([
        tool_use("write_file", {"path": "b.py", "content": "1\n"}),
        tool_use("write_file", {"path": "c.py", "content": "2\n"}),
    ], cost_per_call=1.0)
    result, messages = _run(client, make_executor(), limits=LabLimits(max_cost_usd=1.5))
    assert (result.stop_reason, result.turns) == ("cost_cap", 2)
    assert (workspace / "b.py").exists() and not (workspace / "c.py").exists()
    # The unanswered call is not kept: the conversation ends on the last tool results.
    assert messages[-1]["role"] == "user" and _results(messages[-1])


def _plan_tool(plans):
    def submit_plan(args):
        if not args.get("title"):
            raise ValueError("title is required")
        plans.append(args["title"])
        return "Plan recorded."
    schema = {"name": "submit_plan", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    return ExtraTool(schema, submit_plan, writes=False)


def test_the_finish_tools_turn_is_run_though_it_crossed_the_cost_cap(scripted_client, make_executor):
    # Otherwise the plan the user paid for is thrown away and the next message plans again from nothing.
    plans = []
    client = scripted_client([tool_use("read_file", {"path": "a.py"}),
                              tool_use("read_file", {"path": "a.py"}, ("submit_plan", {"title": "Note element"}))],
                             cost_per_call=1.0)
    result, messages = _run(client, make_executor(read_only=True, extra_tools=[_plan_tool(plans)]),
                            limits=LabLimits(max_cost_usd=1.5), finish_tool="submit_plan")
    assert (result.stop_reason, plans, len(client.calls)) == ("completed", ["Note element"], 2)
    assert [r["is_error"] for r in _results(messages[-1])] == [False, False]


def test_a_refused_finish_call_past_the_cost_cap_stops_on_the_cap(scripted_client, make_executor):
    plans = []
    client = scripted_client([tool_use("read_file", {"path": "a.py"}), tool_use("submit_plan", {})],
                             cost_per_call=1.0)
    result, messages = _run(client, make_executor(read_only=True, extra_tools=[_plan_tool(plans)]),
                            limits=LabLimits(max_cost_usd=1.5), finish_tool="submit_plan")
    assert (result.stop_reason, plans, len(client.calls)) == ("cost_cap", [], 2)
    assert _results(messages[-1])[0]["is_error"] is True


def test_a_final_answer_over_the_cost_cap_still_completes(scripted_client, make_executor):
    client = scripted_client([end_turn("All done.")], cost_per_call=2.0)
    result, _ = _run(client, make_executor(), limits=LabLimits(max_cost_usd=1.0))
    assert (result.stop_reason, result.final_text) == ("completed", "All done.")


def test_cost_is_this_loops_own_spend(scripted_client, make_executor):
    client = scripted_client([tool_use("read_file", {"path": "a.py"}), end_turn()], cost_per_call=0.25)
    client.usage.estimated_cost = 0.5
    result, _ = _run(client, make_executor())
    assert result.cost_usd == pytest.approx(0.5)


def test_runtime_cap_counts_from_started_at(scripted_client, make_executor):
    client = scripted_client([end_turn()])
    result, _ = _run(client, make_executor(), limits=LabLimits(max_runtime_seconds=60),
                     started_at=time.monotonic() - 61)
    assert (result.stop_reason, client.calls) == ("timeout", [])


def test_cancellation_is_checked_every_turn(scripted_client, make_executor):
    turns = []
    client = scripted_client([tool_use("read_file", {"path": "a.py"}) for _ in range(3)])
    result, _ = _run(client, make_executor(),
                     callbacks=LabCallbacks(should_continue=lambda: not turns.append(1) and len(turns) < 2))
    assert (result.stop_reason, result.turns) == ("cancelled", 1)


def test_output_limit_gets_a_smaller_step_retry(workspace, scripted_client, make_executor):
    cut_off = {"stop_reason": "max_tokens",
               "content": [Block("tool_use", id="toolu_cut", name="write_file", input={"path": "big.py"})]}
    client = scripted_client([cut_off, tool_use("write_file", {"path": "big.py", "content": "z = 3\n"}), end_turn()])
    result, messages = _run(client, make_executor())
    assert result.stop_reason == "completed"
    assert messages[1] == {"role": "user", "content": TRUNCATION_PROMPT}
    assert (workspace / "big.py").read_text() == "z = 3\n"
    assert [call["tool"] for call in result.tool_calls] == ["write_file"]


def test_output_limit_retries_are_bounded(scripted_client, make_executor):
    cut_off = {"stop_reason": "max_tokens", "content": [Block("text", text="...")]}
    client = scripted_client([cut_off] * (MAX_TRUNCATION_RETRIES + 1))
    result, _ = _run(client, make_executor())
    assert result.stop_reason == "api_error" and "output token limit" in result.detail
    assert len(client.calls) == MAX_TRUNCATION_RETRIES + 1


def test_an_invalid_key_propagates(scripted_client, make_executor):
    client = scripted_client([InvalidApiKeyError("Claude API rejected the key")])
    with pytest.raises(InvalidApiKeyError):
        _run(client, make_executor())


@pytest.mark.parametrize("response, detail", [
    (UpstreamLLMError("Claude API call failed: 529 overloaded"), "529 overloaded"),
    ({"stop_reason": "refusal", "content": []}, "declined"),
    ({"stop_reason": "pause_turn", "content": [Block("text", text="hm")]}, "pause_turn"),
])
def test_other_provider_failures_end_as_api_error(scripted_client, make_executor, response, detail):
    result, _ = _run(scripted_client([response]), make_executor())
    assert result.stop_reason == "api_error" and detail in result.detail


def test_text_is_streamed_when_wanted(scripted_client, make_executor):
    deltas = []
    client = scripted_client([tool_use("read_file", {"path": "a.py"}, text="Reading."), end_turn("Done.")],
                             stream=True)
    result, _ = _run(client, make_executor(), callbacks=LabCallbacks(on_text=deltas.append))
    assert (result.stop_reason, deltas, client.streamed_calls) == ("completed", ["Reading.", "Done."], 2)

    quiet = scripted_client([end_turn("Done.")], stream=True)
    _run(quiet, make_executor())
    assert quiet.streamed_calls == 0


def test_a_read_after_a_write_in_one_turn_sees_the_write(scripted_client, make_executor):
    client = scripted_client([
        tool_use("write_file", {"path": "n.py", "content": "new = True\n"}, ("read_file", {"path": "n.py"})),
        end_turn(),
    ])
    _, messages = _run(client, make_executor())
    write, read = _results(messages[2])
    assert write["is_error"] is False and "new = True" in json.loads(read["content"])["content"]


class _RendezvousExecutor(LabToolExecutor):
    """Two read_file calls must be in flight together, or the barrier breaks."""

    barrier = threading.Barrier(2, timeout=10)

    def execute(self, name, arguments):
        if name == "read_file":
            self.barrier.wait()
        return super().execute(name, arguments)


def test_reads_of_one_turn_run_in_parallel(workspace, scripted_client, make_executor):
    (workspace / "b.py").write_text("y = 2\n")
    client = scripted_client([tool_use("read_file", {"path": "a.py"}, ("read_file", {"path": "b.py"})), end_turn()])
    _, messages = _run(client, make_executor(_RendezvousExecutor))
    first, second = _results(messages[2])
    assert "x = 1" in first["content"] and "y = 2" in second["content"]


def test_compaction_keeps_the_request_and_shows_re_reads_in_full(workspace, scripted_client, make_executor,
                                                                monkeypatch):
    # A 40k-token window: (40,000 - 16,000 reserve) * 0.8 = 19,200 tokens, ~77k characters.
    monkeypatch.setattr(compaction, "_CATALOG", {"scripted-model": 40_000})
    # 30k characters (~7,500 tokens) each: two reads fit, the third crosses the threshold.
    for name in ("one", "two", "three"):
        (workspace / f"{name}.txt").write_text(((name + " ") * 10_000)[:30_000])
    client = scripted_client([
        tool_use("read_file", {"path": "one.txt"}),
        tool_use("read_file", {"path": "two.txt"}),
        tool_use("read_file", {"path": "three.txt"}),
        tool_use("read_file", {"path": "one.txt"}),
        end_turn(),
    ])
    result, messages = _run(client, make_executor())
    assert result.stop_reason == "completed"
    sent = client.calls[3][1]
    assert len(sent) < 7
    head = sent[0]["content"]
    assert head[0] == {"type": "text", "text": REQUEST}
    assert head[1]["text"].startswith("[Earlier work summarized]")
    # The earlier read of one.txt is gone from the conversation, so the re-read is not a stub.
    reread = json.loads(_results(messages[-2])[0]["content"])
    assert "unchanged" not in reread and reread["content"].count("one") > 1000
