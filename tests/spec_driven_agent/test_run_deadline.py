"""One deadline for every LLM call and command of a run, and a cancel that stops them.

Recorded behaviour this guards against: the run's cap and its cancel were read
only between turns. A 154 s judge call started near the cap ran the run 49 s
past it; a cancel took effect only when a 300 s ``install_dependencies``
returned; a run spent 40 minutes inside installs that never finished.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import openai
import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.agent import tool_executor as te
from besser.spec_driven_agent.errors import DeadlineReached, InvalidApiKeyError, UpstreamLLMError
from besser.spec_driven_agent.execution import deadline as deadline_module
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
    if not os.path.exists(heartbeat):
        return  # killed before the grandchild started: nothing can have survived
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


def _cancel_once_running(deadline: Deadline, heartbeat) -> dict:
    """Cancel once the command's grandchild beats (a loaded machine may take
    seconds to start it); the returned dict gets the moment of the cancel."""
    when: dict = {}

    def watch():
        give_up = time.monotonic() + 60
        while not os.path.exists(heartbeat) and time.monotonic() < give_up:
            time.sleep(0.05)
        time.sleep(0.3)
        when["at"] = time.monotonic()
        deadline.cancel()

    threading.Thread(target=watch, daemon=True).start()
    return when


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def test_cancel_kills_a_running_command_and_its_children(tmp_path):
    heartbeat = tmp_path / "beat"
    deadline = Deadline(_clock(1000))
    cancelled = _cancel_once_running(deadline, heartbeat)

    with pytest.raises(CommandStopped) as info:
        _Run(deadline).call(run_bounded, [sys.executable, "-c", _ROOT, str(heartbeat)], timeout=60)

    assert time.monotonic() - cancelled["at"] < MARGIN
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
    cancelled = _cancel_once_running(deadline, heartbeat)

    result = _Run(deadline).call(executor._run_command, {
        "command": f'"{sys.executable}" "{script}" "{heartbeat}"', "timeout": 300})

    assert time.monotonic() - cancelled["at"] < MARGIN
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



# --------------------------------------------------------------------------- #
# Streaming calls against a local server (127.0.0.1 only)
# --------------------------------------------------------------------------- #
_TOOL = {"name": "write_file", "description": "w", "input_schema": {"type": "object", "properties": {}}}
_BODY = json.dumps({"path": "app.py", "content": "x = 1\n" * 40})


def _claude_events(pieces):
    yield "message_start", {"type": "message_start", "message": {
        "id": "m1", "type": "message", "role": "assistant", "model": "claude-x", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}}
    yield "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {
        "type": "tool_use", "id": "t1", "name": "write_file", "input": {}}}
    step = len(_BODY) // pieces + 1
    for i in range(0, len(_BODY), step):
        yield "content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "input_json_delta", "partial_json": _BODY[i:i + step]}}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                            "usage": {"output_tokens": 5}}
    yield "message_stop", {"type": "message_stop"}


def _openai_events(pieces):
    step = len(_BODY) // pieces + 1
    for i in range(0, len(_BODY), step):
        call = {"index": 0, "function": {"arguments": _BODY[i:i + step]}}
        if i == 0:
            call.update({"id": "t1", "type": "function"})
            call["function"]["name"] = "write_file"
        yield None, {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                     "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}
    yield None, {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}


@pytest.fixture
def sse_server():
    """A local streaming endpoint: ``serve(kind, gap, stall_after)`` sends one
    event every ``gap`` seconds and, after ``stall_after`` events, goes silent
    for a minute. Returns the base URL and an event set once the first event is out."""
    servers = []

    def serve(kind, gap=0.5, stall_after=None):
        first_sent = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                events = _claude_events(12) if kind == "claude" else _openai_events(12)
                try:
                    for sent, (name, data) in enumerate(events):
                        raw = ((f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n").encode()
                        self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                        self.wfile.flush()
                        first_sent.set()
                        if stall_after is not None and sent + 1 >= stall_after:
                            time.sleep(60)
                        time.sleep(gap)
                    if kind == "openai":
                        raw = b"data: [DONE]\n\n"
                        self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", first_sent

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def _streaming_client(kind, base):
    if kind == "claude":
        return ClaudeLLMClient(api_key="k", model="claude-x", base_url=base, max_retries=0)
    return OpenAIProvider(api_key="k", model="m", base_url=base + "/v1", max_retries=0)


def _consume(client, deadline):
    events = []

    def read():
        for event in client.chat_stream(system="s", messages=_MESSAGES, tools=[_TOOL]):
            events.append(event)

    return events, (lambda: _Run(deadline).call(read))


@pytest.mark.parametrize("kind", ["claude", "openai"])
def test_the_cap_cuts_a_stream_that_keeps_sending(kind, sse_server, monkeypatch):
    """A float timeout is per read: a stream that keeps sending was never cut,
    so a turn started near the cap ran on past it. The partial tool call must
    not come back as a reply."""
    monkeypatch.setattr(deadline_module, "MIN_CALL_SECONDS", 0.5)
    base, _ = sse_server(kind, gap=0.5)  # about 9 s of events
    deadline = Deadline(_clock(PACKAGING_RESERVE_SECONDS + 2))
    events, read = _consume(_streaming_client(kind, base), deadline)

    started = time.monotonic()
    with pytest.raises(DeadlineReached) as info:
        read()

    assert time.monotonic() - started < 5
    assert not info.value.cancelled
    assert not [e for e in events if e.get("type") == "message_done"]


@pytest.mark.parametrize("kind", ["claude", "openai"])
def test_cancel_ends_a_stalled_stream(kind, sse_server):
    """A stream waiting for a chunk that never comes: closing the response from
    another thread does not wake that read on every platform, so the socket
    is shut down as well."""
    base, first_sent = sse_server(kind, gap=0.05, stall_after=1)
    deadline = Deadline(_clock(1000))
    events, read = _consume(_streaming_client(kind, base), deadline)
    when: dict = {}

    def cancel_when_stalled():
        first_sent.wait(30)
        time.sleep(0.5)
        when["at"] = time.monotonic()
        deadline.cancel()

    threading.Thread(target=cancel_when_stalled, daemon=True).start()
    with pytest.raises(DeadlineReached) as info:
        read()

    assert time.monotonic() - when["at"] < 3
    assert info.value.cancelled


# --------------------------------------------------------------------------- #
# What cut a command is reported as such
# --------------------------------------------------------------------------- #
def _python_command(code):
    return f'"{sys.executable}" -c "{code}"'


def test_a_command_with_no_work_time_left_is_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda command, **_: SandboxedCommand(command, True, "test"))
    marker = tmp_path / "ran"
    executor = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    deadline = Deadline(lambda: PACKAGING_RESERVE_SECONDS - 5)  # inside the packaging reserve
    executor.time_left = deadline.work_remaining

    result = _Run(deadline).call(executor._run_command, {
        "command": _python_command(f"open(r'{marker}', 'w').close()")})

    assert result.get("stopped") == OUT_OF_TIME, result
    assert not marker.exists()


def test_a_command_cut_by_the_work_time_left_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(shell_session, "sandboxed_command",
                        lambda command, **_: SandboxedCommand(command, True, "test"))
    executor = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    deadline = Deadline(_clock(PACKAGING_RESERVE_SECONDS + 3))
    executor.time_left = deadline.work_remaining

    result = _Run(deadline).call(executor._run_command, {
        "command": _python_command("import time; time.sleep(30)"), "timeout": 300})

    assert result.get("stopped") == OUT_OF_TIME, result


def test_a_type_check_cut_by_the_run_names_the_run_not_its_own_timeout(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import sandbox
    from besser.spec_driven_agent.validation import toolchain

    (tmp_path / "tsconfig.json").write_text("{}")
    monkeypatch.setattr("shutil.which", lambda name: "tsc" if name.startswith("tsc") else None)

    def cut(argv, **kwargs):
        raise CommandStopped(argv, 4, OUT_OF_TIME)

    monkeypatch.setattr(sandbox, "run_confined", cut)
    issues = toolchain._collect_tsc_issues(str(tmp_path), False, True, set())

    assert any(OUT_OF_TIME in issue for issue in issues), issues
    assert not any("timed out after 60s" in issue for issue in issues), issues


def test_formatting_stopped_by_the_run_names_the_run(tmp_path, monkeypatch):
    from besser.spec_driven_agent.validation import toolchain

    (tmp_path / "app.py").write_text("x=1\n")
    monkeypatch.setattr("shutil.which", lambda name: "ruff")

    def stopped(argv, **kwargs):
        raise CommandStopped(argv, 0, CANCELLED)

    monkeypatch.setattr(toolchain, "run_bounded", stopped)
    issues = toolchain.format_python_files(str(tmp_path), ["app.py"])

    assert len(issues) == 1 and CANCELLED in issues[0], issues


# --------------------------------------------------------------------------- #
# Only time and cancellation become a deadline stop
# --------------------------------------------------------------------------- #
def test_a_quota_error_near_the_cap_is_still_a_provider_error():
    """The quota stop must stay visible to the run, which records it, and the
    run's end is no reason to move to a fallback model."""
    left = {"seconds": 1000.0}

    def exhausted(**kwargs):
        left["seconds"] = 5.0
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
        raise openai.RateLimitError("You exceeded your current quota (insufficient_quota)",
                                    response=response, body={"code": "insufficient_quota"})

    provider = _openai_provider(_RecordingCompletions(exhausted),
                                fallback=[("http://127.0.0.1:9/v1", "tok", "other-model")])

    with pytest.raises(UpstreamLLMError) as info:
        _Run(Deadline(lambda: left["seconds"])).call(provider.chat, "system", _MESSAGES, [])

    assert "quota" in str(info.value)
    assert provider._fallback_step == 0


def test_a_judge_whose_retry_wait_does_not_fit_is_not_checked_for_time(tmp_path):
    """A rate limit asking for a wait longer than the time left allows ends
    the judge call: a time stop, not a judge failure."""

    def throttled(**kwargs):
        response = httpx.Response(429, headers={"retry-after": "25"},
                                  request=httpx.Request("POST", "https://example.invalid"))
        raise openai.RateLimitError("rate limited", response=response, body=None)

    provider = _openai_provider(_RecordingCompletions(throttled))
    orch = LLMOrchestrator(llm_client=provider, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._instructions = "Orders have a unique id."
    orch._generator_used = "generate_fastapi_backend"
    orch._requirements = [{"id": 1, "text": "Orders have a unique id", "kind": "uniqueness"}]
    (tmp_path / "routers").mkdir()
    (tmp_path / "routers" / "order.py").write_text("def create_order():\n    return 1\n")
    # 40 s of work time: a call may start; a 25 s wait and then a call does not fit.
    _seconds_to_cap(orch, PACKAGING_RESERVE_SECONDS + 40)

    issues = _Run(orch.deadline).call(orch._collect_requirement_issues)

    assert not orch._requirement_judge_failures, "a time stop is not a judge failure"
    assert len(issues) == 1 and issues[0].startswith("requirement unverified: not checked"), issues


# --------------------------------------------------------------------------- #
# Phase 3: checks need time to the cap, repair turns need time for a call
# --------------------------------------------------------------------------- #
def _seconds_to_cap(orch, seconds):
    orch._start_time = time.monotonic() - orch.max_runtime_seconds + seconds


def test_phase3_checks_still_run_when_only_repair_turns_are_out_of_time(tmp_path):
    client = SimpleNamespace(model="mock-model", usage=SimpleNamespace(estimated_cost=0.0))
    orch = LLMOrchestrator(llm_client=client, domain_model=_model(),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    _seconds_to_cap(orch, PACKAGING_RESERVE_SECONDS + MIN_CALL_SECONDS - 5)

    assert orch._phase3_stop_requested() == "runtime budget exhausted"
    # Ending the repair turns for time must not mark the attempt as interrupted.
    assert orch._phase3_stop_requested(check_turn_budget=False) is None
    _seconds_to_cap(orch, 5)
    assert orch._phase3_stop_requested(check_turn_budget=False) == "runtime budget exhausted"


def test_a_repair_turn_refused_for_time_leaves_the_attempt_checkable(simple_library_book_model, tmp_path):
    from tests.spec_driven_agent.test_phase3_stall_guards import _blockers

    def refused(**kwargs):
        raise DeadlineReached(OUT_OF_TIME)

    client = SimpleNamespace(model="mock-model", max_tokens=4096, usage=SimpleNamespace(estimated_cost=0.0),
                             chat=refused)
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_library_book_model, output_dir=str(tmp_path),
                           enable_checkpointing=False, enable_tracing=False, auto_fix_issues=True)
    _seconds_to_cap(orch, 600)

    orch._invoke_phase3_fix_loop(_blockers(2), is_first_attempt=True)

    assert not orch._phase3_interrupted
    assert orch._phase3_stop_requested(check_turn_budget=False) is None


def test_a_repair_that_returns_with_no_work_time_left_is_validated_not_discarded(
        simple_library_book_model, tmp_path):
    from tests.spec_driven_agent.test_phase3_stall_guards import _blockers, _drive

    client = SimpleNamespace(model="mock-model", max_tokens=4096, usage=SimpleNamespace(estimated_cost=0.0))
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_library_book_model, output_dir=str(tmp_path),
                           enable_checkpointing=False, enable_tracing=False, auto_fix_issues=True)
    _seconds_to_cap(orch, 600)
    discarded = []
    orch._discard_unverified_attempt = lambda *args: discarded.append(args) or True

    attempts = _drive(orch, [(1, True, False, _blockers(2))],
                      after_round=lambda _attempt: _seconds_to_cap(orch, PACKAGING_RESERVE_SECONDS + 5))

    assert attempts == 1
    assert discarded == []
    assert orch._phase3_exit_reason.startswith("runtime budget exhausted")
    assert sum(issue.severity == "blocker" for issue in orch._validation_issues) == 2, "the attempt was re-checked"



# --------------------------------------------------------------------------- #
# A rejected server key while the run is stopping
# --------------------------------------------------------------------------- #
def _rejected_key():
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid"))
    return openai.AuthenticationError("invalid token", response=response, body=None)


def test_a_rejected_server_key_during_a_cancel_ends_the_run_cleanly(tmp_path):
    """On the server-paid tier a rejected key moves the run to the next
    endpoint. When the run is being cancelled there is no next call: it is a
    stop, and the run still writes its result, not an invalid-key failure."""
    orch = None

    def reply(**kwargs):
        tools = {tool["function"]["name"] for tool in kwargs.get("tools") or []}
        if "write_file" in tools and "tool_choice" not in kwargs:  # a Phase 2 turn
            orch.deadline.cancel()  # the user cancels while the request is out
            raise _rejected_key()
        return _openai_reply()

    provider = _openai_provider(_RecordingCompletions(reply),
                                fallback=[("http://127.0.0.1:9/v1", "tok", "other-model")])
    orch = LLMOrchestrator(llm_client=provider, domain_model=_model(), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, max_runtime_seconds=900)

    orch.run("Build an order tracking app")

    assert orch._phase2_stop_reason == "cancelled"
    assert provider._fallback_step == 0
    assert os.path.isfile(os.path.join(str(tmp_path), ".besser_recipe.json"))


def test_a_users_own_rejected_key_is_still_reported_during_a_cancel():
    """Without a fallback the key is the user's: they must be told it was rejected."""
    deadline = Deadline(_clock(1000))

    def rejected(**kwargs):
        deadline.cancel()
        raise _rejected_key()

    provider = _openai_provider(_RecordingCompletions(rejected))

    with pytest.raises(InvalidApiKeyError):
        _Run(deadline).call(provider.chat, "system", _MESSAGES, [])
