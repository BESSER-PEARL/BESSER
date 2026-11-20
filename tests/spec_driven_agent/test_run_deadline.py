"""One deadline for every LLM call and command of a run, and a cancel that stops them.

Recorded behaviour this guards against: the run's cap and its cancel were read
only between turns. A 154 s judge call started near the cap ran the run 49 s
past it; a cancel took effect only when a 300 s ``install_dependencies``
returned; a run spent 40 minutes inside installs that never finished.
"""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.errors import DeadlineReached
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.deadline import (
    CANCELLED,
    MIN_CALL_SECONDS,
    OUT_OF_TIME,
    PACKAGING_RESERVE_SECONDS,
    Deadline,
    current_deadline,
    under_deadline,
)
from besser.spec_driven_agent.execution.process import CommandStopped, run_bounded
from besser.spec_driven_agent.execution.sandbox import SandboxedCommand
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import ClaudeLLMClient, OpenAIProvider

MARGIN = 8

# A command whose child leaves a grandchild writing a heartbeat file: the tree
# is dead when the file stops growing (no platform-specific process API).
_GRANDCHILD = (
    "import sys, time\n"
    "for _ in range(150):\n"
    "    with open(sys.argv[1], 'a') as fh:\n"
    "        fh.write('.')\n"
    "    time.sleep(0.2)\n"
)
_ROOT = (
    "import subprocess, sys, time\n"
    f"subprocess.Popen([sys.executable, '-c', {_GRANDCHILD!r}, sys.argv[1]])\n"
    "print('started', flush=True)\n"
    "time.sleep(20)\n"
)


def _assert_tree_is_dead(heartbeat):
    deadline = time.monotonic() + 5
    while not os.path.exists(heartbeat) and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.6)
    before = os.path.getsize(heartbeat)
    time.sleep(1.2)
    assert os.path.getsize(heartbeat) == before, "a child of the command survived"


class _Run:
    """Stands in for a run entry point: ``call`` runs under ``deadline``."""

    def __init__(self, deadline):
        self.deadline = deadline

    @under_deadline
    def call(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _clock(seconds_left: float):
    """A ``time_left`` counting down from ``seconds_left`` in real time."""
    ends = time.monotonic() + seconds_left
    return lambda: ends - time.monotonic()


def _cancel_after(deadline: Deadline, seconds: float) -> None:
    timer = threading.Timer(seconds, deadline.cancel)
    timer.daemon = True
    timer.start()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def test_cancel_kills_a_running_command_and_its_children(tmp_path):
    heartbeat = tmp_path / "beat"
    deadline = Deadline(_clock(1000))
    _cancel_after(deadline, 1.0)

    started = time.monotonic()
    with pytest.raises(CommandStopped) as info:
        _Run(deadline).call(run_bounded, [sys.executable, "-c", _ROOT, str(heartbeat)], timeout=60)

    assert time.monotonic() - started < 1.0 + MARGIN
    assert info.value.reason == CANCELLED
    assert "started" in (info.value.output or ""), "output before the kill is kept"
    _assert_tree_is_dead(heartbeat)


def test_a_command_gets_no_more_than_the_time_left(tmp_path):
    heartbeat = tmp_path / "beat"
    deadline = Deadline(_clock(3))

    started = time.monotonic()
    with pytest.raises(CommandStopped) as info:
        _Run(deadline).call(run_bounded, [sys.executable, "-c", _ROOT, str(heartbeat)], timeout=60)

    assert time.monotonic() - started < 3 + MARGIN
    assert info.value.reason == OUT_OF_TIME
    _assert_tree_is_dead(heartbeat)


def test_no_command_starts_once_the_run_is_out_of_time(tmp_path):
    marker = tmp_path / "ran"
    deadline = Deadline(lambda: 0.5)

    with pytest.raises(CommandStopped):
        _Run(deadline).call(run_bounded, [sys.executable, "-c",
                                          f"open({str(marker)!r}, 'w').close()"], timeout=60)

    assert not marker.exists()


def test_a_timeout_inside_a_run_with_time_left_is_still_a_timeout():
    deadline = Deadline(_clock(1000))
    with pytest.raises(subprocess.TimeoutExpired) as info:
        _Run(deadline).call(run_bounded, [sys.executable, "-c", "import time; time.sleep(20)"], timeout=1)
    assert not isinstance(info.value, CommandStopped)


def test_run_command_returns_a_stopped_result_on_cancel(tmp_path, monkeypatch):
    # The sandbox is not what is under test; run the command as the shell would.
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda command, **_: SandboxedCommand(command, True, "test"))
    heartbeat = tmp_path / "beat"
    script = tmp_path / "root.py"
    script.write_text(_ROOT)
    executor = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    deadline = Deadline(_clock(1000))
    _cancel_after(deadline, 1.0)

    started = time.monotonic()
    result = _Run(deadline).call(executor._run_command, {
        "command": f'"{sys.executable}" "{script}" "{heartbeat}"', "timeout": 300})

    assert time.monotonic() - started < 1.0 + MARGIN
    assert result["stopped"] == CANCELLED
    assert result["error"].startswith("Command stopped: the run was cancelled")
    _assert_tree_is_dead(heartbeat)


@pytest.mark.skipif(os.name == "nt", reason="the session supervisor is POSIX-only")
def test_cancel_ends_the_sandboxed_session_running_the_command(tmp_path, monkeypatch):
    # The supervisor runs as if it were the sandbox's init (see test_shell_session.py).
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda argv, **_: SandboxedCommand(argv, False, "bwrap"))
    session = shell_session.ShellSession(str(tmp_path))
    deadline = Deadline(_clock(1000))
    _cancel_after(deadline, 1.0)
    try:
        started = time.monotonic()
        result = _Run(deadline).call(session.run, "sleep 300 & echo $! > child.pid; wait",
                                     working_dir=None, timeout=300)
        assert time.monotonic() - started < 1.0 + MARGIN
        assert result.stopped == CANCELLED
    finally:
        session.close()
        # Without namespaces the child outlives the supervisor; the real sandbox takes it along.
        try:
            os.kill(int((tmp_path / "child.pid").read_text()), 9)
        except (OSError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #
_MESSAGES = [{"role": "user", "content": "hi"}]


def _openai_reply(text="done"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=None, model="gpt-4o")


class _RecordingCompletions:
    def __init__(self, reply=None):
        self.timeouts = []
        self._reply = reply or (lambda **kwargs: _openai_reply())

    def create(self, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return self._reply(**kwargs)


def _openai_provider(completions, **kwargs):
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o", **kwargs)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def test_an_llm_call_gets_no_more_than_the_time_left():
    completions = _RecordingCompletions()
    provider = _openai_provider(completions)
    deadline = Deadline(_clock(100))

    _Run(deadline).call(provider.chat, "system", _MESSAGES, [])

    [timeout] = completions.timeouts
    assert timeout is not None and timeout <= 100 - PACKAGING_RESERVE_SECONDS


def test_a_claude_call_gets_no_more_than_the_time_left():
    timeouts = []

    def create(**kwargs):
        timeouts.append(kwargs.get("timeout"))
        return SimpleNamespace(stop_reason="end_turn", content=[], usage=SimpleNamespace(
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0))

    client = ClaudeLLMClient(api_key="sk-ant-test")
    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    _Run(Deadline(_clock(100))).call(client.chat, "system", _MESSAGES, [])

    assert timeouts and timeouts[0] is not None and timeouts[0] <= 100 - PACKAGING_RESERVE_SECONDS


def test_no_llm_call_starts_with_less_than_the_floor_left():
    completions = _RecordingCompletions()
    provider = _openai_provider(completions)
    deadline = Deadline(lambda: PACKAGING_RESERVE_SECONDS + MIN_CALL_SECONDS - 1)

    with pytest.raises(DeadlineReached) as info:
        _Run(deadline).call(provider.chat, "system", _MESSAGES, [])

    assert completions.timeouts == []
    assert not info.value.cancelled


def test_a_call_cut_by_the_deadline_does_not_move_to_the_fallback_model():
    """A timeout caused by the run's own deadline is not a provider outage."""
    left = {"seconds": 1000.0}

    def time_out(**kwargs):
        left["seconds"] = 5.0  # the call used the run's time up
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))

    completions = _RecordingCompletions(time_out)
    provider = _openai_provider(completions, fallback=[("http://127.0.0.1:9/v1", "tok", "other-model")])

    with pytest.raises(DeadlineReached):
        _Run(Deadline(lambda: left["seconds"])).call(provider.chat, "system", _MESSAGES, [])

    assert len(completions.timeouts) == 1, "not retried"
    assert provider._fallback_step == 0 and provider.model == "gpt-4o"


class _BlockingStream:
    """A streaming response whose next chunk never comes; closing it ends the
    blocked read the way a closed connection does."""

    def __init__(self):
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        self.closed.wait(5)
        raise httpx.ReadError("connection closed")

    def close(self):
        self.closed.set()


def test_cancel_aborts_a_streaming_response():
    stream = _BlockingStream()
    provider = _openai_provider(_RecordingCompletions(lambda **kwargs: stream))
    deadline = Deadline(_clock(1000))
    _cancel_after(deadline, 0.5)

    started = time.monotonic()
    with pytest.raises(DeadlineReached) as info:
        _Run(deadline).call(lambda: list(provider.chat_stream("system", _MESSAGES, [])))

    assert time.monotonic() - started < 3
    assert info.value.cancelled
    assert stream.closed.is_set()


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def _model():
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="Shop", types={order})


def test_tool_threads_work_under_the_run_deadline(tmp_path):
    orch = LLMOrchestrator(llm_client=SimpleNamespace(model="mock-model"), domain_model=_model(),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    seen = []

    def execute(block, turn):
        seen.append(current_deadline())
        return {"type": "tool_result", "tool_use_id": block.id, "content": "{}"}

    orch._execute_single_tool = execute
    blocks = [SimpleNamespace(id=f"t{i}", name="read_file", input={"path": f"f{i}.py"}) for i in range(3)]

    _Run(orch.deadline).call(orch._execute_tool_blocks, blocks, 0)

    assert seen == [orch.deadline] * 3


def test_a_phase2_call_cut_by_the_deadline_stops_the_run_and_it_still_reports(tmp_path):
    """Every call gets at most the time left; the cut is a timeout stop, not a
    provider error, the fallback chain stays where it was, and the run still
    writes its recipe."""
    orch = None
    calls = []

    def reply(**kwargs):
        calls.append((kwargs.get("timeout"), orch.deadline.work_remaining()))
        tools = {tool["function"]["name"] for tool in kwargs.get("tools") or []}
        if "write_file" in tools and "tool_choice" not in kwargs:  # a Phase 2 turn
            orch._start_time -= orch.max_runtime_seconds  # the call used the run's time up
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
        return _openai_reply()

    provider = _openai_provider(_RecordingCompletions(reply),
                                fallback=[("http://127.0.0.1:9/v1", "tok", "other-model")])
    orch = LLMOrchestrator(llm_client=provider, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, max_runtime_seconds=900)

    orch.run("Build an order tracking app")

    assert calls, "the run made LLM calls"
    # ``left`` is read just after the call started, so it may be a hair smaller.
    assert all(timeout is not None and timeout <= left + 1 for timeout, left in calls), calls
    assert orch._phase2_stop_reason == "timeout"
    assert provider._fallback_step == 0 and provider.model == "gpt-4o"
    assert os.path.isfile(os.path.join(str(tmp_path), ".besser_recipe.json"))
