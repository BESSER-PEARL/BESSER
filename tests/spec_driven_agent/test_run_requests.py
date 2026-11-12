"""Whole-run properties of the requests a Phase 2 loop sends.

The REAL orchestrator drives the REAL provider clients; only the SDK
transport is scripted (``run_transport``). Each scenario plays a plausible run
on the Shelf FastAPI scaffold: a plan, batched reads, an edit the executor
refuses three times (which makes the loop force the next tool), the corrected
edit, a new module, checklist bookkeeping, ``validate_app``, a premature
end_turn that the checklist gate answers with a nudge, and a clean end_turn.
The compaction threshold is lowered so the history is compacted mid-run.

What is checked is what a single-unit test cannot see: that the system prompt
and tools stay byte-identical (the provider's prompt cache keys on them), that
each request extends the previous one except where the history is rewritten
on purpose, that every replayed thinking block is sent after exactly the
conversation that produced it, that tool calls stay paired, and that cost
accounting and the cost cap see every call.

Known gap, not encoded as a pass: the executor's record of the run's own
successful writes is not checkpointed, so after a resume ``task_list done``
citing an edit made before the crash is refused ("no successful write matches
the current contents"). The resume scenario therefore crashes before the
first landed edit.
"""

from __future__ import annotations

import json

import pytest

import besser.spec_driven_agent.agent.compaction as compaction
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.state.checkpoint import load_checkpoint
from tests.spec_driven_agent.run_transport import (
    CRASH, ScriptedModel, Step, Usage, blocks, claude_run_client, count_cache_markers,
    openai_run_client, plain, without_cache_control,
)
from tests.spec_driven_agent.shelf_app import shelf_model

# Low enough that the scripted history is compacted once, mid-run; high
# enough that the forced turn's reply is replayed before that happens. The
# forced re-read of the unchanged router is an "unchanged" stub, not the file.
COMPACT_AT_TOKENS = 9_500

TASKS = [
    "Add a GET /book/stats/ endpoint returning the total page count of all books",
    "Add a shelf statistics module with per-library page totals",
]
SHORT_REQUEST = "Build a REST API for my book shelf, with a stats endpoint for page totals."
LAST_SENTENCE = "Finally, every statistics response must report the zebra-quartz audit marker."
LONG_REQUEST = "\n\n".join(
    [f"Requirement {i}: the shelf service keeps books and libraries consistent; "
     "a book has a title and a page count, a library has a name and holds books. "
     "Totals are computed by the server from stored values, never accepted from the client."
     for i in range(1, 15)]
    + [LAST_SENTENCE]
)
assert len(LONG_REQUEST) > 2_000

BOOK_ROUTER = "backend/routers/book.py"
SEARCH_ROUTE = '@router.get("/book/search/", response_model=None, tags=["Book"])'
STATS_ROUTE = (
    '@router.get("/book/stats/", response_model=None, tags=["Book"])\n'
    "def book_stats(database: Session = Depends(get_db)) -> dict:\n"
    '    return {"pages": sum(book.pages or 0 for book in database.query(Book).all())}\n\n\n'
)
STATS_MODULE = (
    '"""Per-library page totals."""\n'
    "from fastapi import APIRouter, Depends\nfrom sqlalchemy.orm import Session\n\n"
    "from database import get_db\nfrom sql_alchemy import Library\n\nrouter = APIRouter()\n\n\n"
    '@router.get("/shelf/stats/", tags=["Shelf"])\n'
    "def shelf_stats(database: Session = Depends(get_db)) -> dict:\n"
    "    totals = {}\n    for library in database.query(Library).all():\n"
    "        totals[library.name] = sum(book.pages or 0 for book in library.books)\n"
    "    return totals\n"
    + "".join(f"\n\n# Note {i}: totals are recomputed on every request; " + "n" * 80
              for i in range(80))
)
INCLUDE = "app.include_router(book_router.router)"
REJECTED_EDIT = {"path": BOOK_ROUTER, "old_text": "def search_books(", "new_text": "def find_books("}


def _task_id(model: ScriptedModel, text: str) -> int:
    for task in model.orchestrator.executor.task_snapshot():
        if task["text"] == text:
            return task["id"]
    raise AssertionError(f"no task {text!r}")


def _done(text: str, path: str, quote: str):
    def step(model: ScriptedModel) -> Step:
        task = _task_id(model, text)
        return Step(thinking=f"{text} is implemented.", calls=[("task_list", {
            "action": "done", "id": task, "evidence": [{"id": task, "path": path, "quote": quote}]})])
    return step


def shelf_script() -> list:
    """The Phase 2 turns, in order (see the module docstring)."""
    return [
        Step(thinking="Read the router, the app and the ORM before editing.",
             text="Plan: add the stats route to the book router, then a stats module.",
             calls=[("read_file", {"path": "backend/main_api.py"}),
                    ("read_file", {"path": BOOK_ROUTER}),
                    ("read_file", {"path": "backend/sql_alchemy.py"})]),
        # The same wrong quote three times: refused each time, so the loop
        # forces read_file on the next request.
        *[Step(thinking=f"Rename the search route (attempt {i}).",
               calls=[("modify_file", dict(REJECTED_EDIT))]) for i in range(1, 4)],
        Step(thinking="Re-read the router as asked.", calls=[("read_file", {"path": BOOK_ROUTER})]),
        Step(thinking="Insert the route before the search route, quoting it exactly.",
             calls=[("modify_file", {"path": BOOK_ROUTER, "old_text": SEARCH_ROUTE,
                                     "new_text": STATS_ROUTE + SEARCH_ROUTE})]),
        Step(thinking="Add the module and mount it.",
             calls=[("write_file", {"path": "backend/routers/shelf_stats.py", "content": STATS_MODULE}),
                    ("modify_file", {"path": "backend/main_api.py", "old_text": INCLUDE, "new_text": (
                        INCLUDE + "\nfrom routers import shelf_stats as shelf_stats_router\n"
                        "app.include_router(shelf_stats_router.router)")})]),
        _done(TASKS[0], BOOK_ROUTER, "def book_stats("),
        Step(thinking="Check the app boots.", calls=[("validate_app", {})]),
        Step(thinking="Looks complete.", text="The stats endpoint is in place."),  # premature
        _done(TASKS[1], "backend/routers/shelf_stats.py", "def shelf_stats("),
        Step(thinking="Everything is closed.", text="All checklist items are closed."),
    ]


@pytest.fixture(autouse=True)
def _compact_mid_run(monkeypatch):
    monkeypatch.setattr(compaction, "_OPERATOR_CAP", COMPACT_AT_TOKENS)


@pytest.fixture(autouse=True)
def _no_app_boot(monkeypatch):
    """Booting the app in a subprocess is ~2s a validation and says nothing about requests."""
    monkeypatch.setattr(LLMOrchestrator, "_runtime_probe_issues", lambda self: [])


def _run(tmp_path, make_client, instruction, *, script=None, streaming=False, usage=None,
         max_cost_usd=5.0, resume=False):
    model = ScriptedModel(str(tmp_path), shelf_script() if script is None else script, TASKS, usage=usage)
    client, fake = make_client(model)
    orchestrator = LLMOrchestrator(
        llm_client=client, domain_model=shelf_model(), output_dir=str(tmp_path), max_turns=40,
        max_cost_usd=max_cost_usd, use_streaming=streaming,
        on_text=(lambda text: None) if streaming else None,
        enable_requirements_ledger=False, auto_fix_issues=False,
    )
    model.orchestrator = orchestrator
    fake.cost_probe = lambda: client.usage.estimated_cost
    (orchestrator.resume if resume else orchestrator.run)(instruction)
    return orchestrator, model, fake, client


def _claude(model_id, *, thinking):
    return lambda model: claude_run_client(model_id, model, with_thinking=thinking)


def _openai(model_id):
    return lambda model: openai_run_client(model_id, model)


# ----------------------------------------------------------------------
# Reading requests
# ----------------------------------------------------------------------

_ONE_OFF_INSTRUCTION = "Respond by calling the `"
_SUMMARY_PREFIX = "[Earlier work summarized]"


def _normalized(messages: list) -> list[dict]:
    """Messages as the provider reads them: plain blocks, no cache markers."""
    out = []
    for message in plain(messages):
        message = without_cache_control(message)
        if message.get("role") in ("user", "assistant") and "tool_calls" not in message \
                and not isinstance(message.get("content"), (type(None),)):
            message = {**message, "content": blocks(message)}
        out.append(message)
    return out


def _history(request: dict) -> list[dict]:
    """The conversation a request carries, without a one-off forced-tool instruction."""
    messages = _normalized(request["messages"])
    if messages and messages[0].get("role") == "system":
        messages = messages[1:]
    last = messages[-1] if messages else None
    if request.get("tool_choice") == {"type": "auto"} and last and last["role"] == "user":
        content = last["content"]
        if content and content[-1].get("type") == "text" \
                and content[-1]["text"].startswith(_ONE_OFF_INSTRUCTION):
            messages = [*messages[:-1], {**last, "content": content[:-1]}]
    return messages


def _first_text(request: dict) -> str:
    history = _history(request)
    content = history[0]["content"] if history else []
    return content if isinstance(content, str) else (content[0].get("text", "") if content else "")


def _is_compaction(request: dict) -> bool:
    return _first_text(request).startswith(_SUMMARY_PREFIX)


def _rewrites(requests: list[dict]) -> list[tuple[int, str, int]]:
    """``(request index, reason, first differing message)`` wherever a request
    is not the previous one plus appended messages."""
    found = []
    for index in range(1, len(requests)):
        before, after = _history(requests[index - 1]), _history(requests[index])
        if after[:len(before)] == before:
            continue
        first = next((j for j, (a, b) in enumerate(zip(before, after)) if a != b),
                     min(len(before), len(after)))
        reason = "compaction" if _is_compaction(requests[index]) else "other"
        found.append((index, reason, first))
    return found


def _thinking_blocks(request: dict):
    for j, message in enumerate(_normalized(request["messages"])):
        for block in message["content"] if isinstance(message.get("content"), list) else []:
            if block.get("type") in ("thinking", "redacted_thinking"):
                yield j, block


def _producer(block: dict) -> int:
    """0-based index, in the transport's request log, of the request whose reply held ``block``."""
    return int(block["signature"].split("-")[1]) - 1


def _thinking_violations(fake) -> list[str]:
    """Every replayed thinking block must follow exactly the conversation that
    produced it (system, tools and every earlier message, as that request sent them)."""
    problems = []
    for index, record in enumerate(fake.requests):
        request = record.kwargs
        for j, block in _thinking_blocks(request):
            producer = fake.requests[_producer(block)].kwargs
            issued = fake.issued_thinking[block["signature"]]
            if block["thinking"] != issued:
                problems.append(f"request {index}: thinking {block['signature']} was altered")
            if plain(request["system"]) != plain(producer["system"]) \
                    or plain(request.get("tools")) != plain(producer.get("tools")):
                problems.append(f"request {index}: {block['signature']} replayed under another system/tools")
            if _normalized(request["messages"][:j]) != _normalized(producer["messages"]):
                problems.append(
                    f"request {index}: {block['signature']} replayed after a different conversation "
                    f"than request {_producer(block)} that produced it")
    return problems


def _pairing_problems_anthropic(request: dict) -> list[str]:
    messages = _normalized(request["messages"])
    problems = []
    if not messages or messages[0]["role"] != "user":
        problems.append("first message is not a user message")
    for j, message in enumerate(messages):
        uses = [b["id"] for b in message["content"] if b.get("type") == "tool_use"]
        results = [b["tool_use_id"] for b in message["content"] if b.get("type") == "tool_result"]
        if message["role"] == "assistant" and uses:
            following = messages[j + 1] if j + 1 < len(messages) else None
            answered = [b.get("tool_use_id") for b in (following or {}).get("content", [])
                        if b.get("type") == "tool_result"]
            if following is None or following["role"] != "user" or sorted(answered) != sorted(uses):
                problems.append(f"message {j}: tool_use {uses} not answered by the next message")
        if results:
            previous = messages[j - 1] if j else None
            asked = [b["id"] for b in (previous or {}).get("content", []) if b.get("type") == "tool_use"]
            if previous is None or previous["role"] != "assistant" or not set(results) <= set(asked):
                problems.append(f"message {j}: tool_result {results} answers no preceding tool_use")
    return problems


def _pairing_problems_openai(request: dict) -> list[str]:
    messages = plain(request["messages"])
    problems = []
    if messages[0]["role"] != "system":
        problems.append("system message is not first")
    open_calls: set[str] = set()
    for j, message in enumerate(messages):
        if message["role"] == "tool":
            if message["tool_call_id"] not in open_calls:
                problems.append(f"message {j}: tool message answers no open tool call")
            open_calls.discard(message["tool_call_id"])
            continue
        if open_calls:
            problems.append(f"message {j}: tool calls {sorted(open_calls)} left unanswered")
            open_calls = set()
        if message["role"] == "assistant":
            if not message.get("content") and not message.get("tool_calls"):
                problems.append(f"message {j}: empty assistant message")
            open_calls = {call["id"] for call in message.get("tool_calls") or []}
    if open_calls:
        problems.append(f"tool calls {sorted(open_calls)} left unanswered at the end")
    return problems


def _occurrences(request: dict, needle: str) -> int:
    return json.dumps(plain({k: request[k] for k in ("system", "messages") if k in request})).count(
        json.dumps(needle)[1:-1])


def _global_index(fake, request: dict) -> int:
    return next(i for i, record in enumerate(fake.requests) if record.kwargs is request)


def _is_forced(request: dict) -> bool:
    if request.get("tool_choice") not in (None, {"type": "auto"}):
        return True
    last = _normalized(request["messages"])[-1]
    content = last.get("content") if isinstance(last.get("content"), list) else []
    return request.get("tool_choice") == {"type": "auto"} and bool(content) \
        and content[-1].get("text", "").startswith(_ONE_OFF_INSTRUCTION)


def _assert_scripted_run_completed(orchestrator, phase2):
    assert orchestrator._phase2_stop_reason == "completed", orchestrator._phase2_api_error
    assert orchestrator._compaction_count >= 1, "the run never compacted; lower COMPACT_AT_TOKENS"
    assert any(_is_forced(request) for request in phase2), "no request forced a tool"
    nudged = [request for request in phase2
              if "You ended the turn, but these checklist items" in json.dumps(plain(request["messages"]))]
    assert nudged, "the premature end_turn was not answered with a nudge"
    tools = [call["tool"] for call in orchestrator.tool_calls_log]
    assert tools.count("modify_file") == 5 and "write_file" in tools and "validate_app" in tools


def _assert_cache_prefix_stable(phase2, compactions: int, *, system=lambda r: r["system"],
                                tools=lambda r: r["tools"]):
    """system and tools never change; messages only grow, except at a compaction."""
    first = phase2[0]
    changed = [i for i, request in enumerate(phase2)
               if plain(system(request)) != plain(system(first))
               or plain(tools(request)) != plain(tools(first))]
    assert changed == [], f"system/tools changed on Phase 2 requests {changed}"
    rewrites = _rewrites(phase2)
    assert [reason for _, reason, _ in rewrites] == ["compaction"] * compactions, rewrites


def _assert_usage_matches_script(client, model, fake, *, net_cached=False):
    usage = client.usage
    scripted = model.scripted_usage
    assert usage.api_calls == len(fake.requests) == len(scripted)
    assert usage.output_tokens == sum(u.output for u in scripted)
    assert usage.cache_read_tokens == sum(u.cache_read for u in scripted)
    assert usage.input_tokens == sum(u.input for u in scripted)
    if not net_cached:
        assert usage.cache_creation_tokens == sum(u.cache_write for u in scripted)


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------

def test_a_claude_5x_run_keeps_its_cache_prefix_and_valid_thinking(tmp_path):
    """Sonnet 5.5: thinking always on, no forced tool_choice, a long request."""
    orchestrator, model, fake, client = _run(
        tmp_path, _claude("claude-sonnet-5-5", thinking=True), LONG_REQUEST)
    phase2 = fake.of("phase2")
    _assert_scripted_run_completed(orchestrator, phase2)
    _assert_cache_prefix_stable(phase2, orchestrator._compaction_count)

    for request in (record.kwargs for record in fake.requests):
        assert count_cache_markers(plain(request)) <= 4
    for index, request in enumerate(phase2):
        assert "thinking" not in request and request.get("tool_choice") in (None, {"type": "auto"})
        assert _pairing_problems_anthropic(request) == [], index

    # A thinking block is valid only after exactly the conversation it was
    # produced in: the same system, tools and earlier messages.
    assert _thinking_violations(fake) == []
    compacted_at = next(_global_index(fake, r) for r in phase2 if _is_compaction(r))
    for request in phase2:
        position = _global_index(fake, request)
        produced_by = {_producer(block) for _, block in _thinking_blocks(request)}
        if position >= compacted_at:
            assert all(p >= compacted_at for p in produced_by), "a pre-compaction thinking block survived"
        else:
            earlier = {_global_index(fake, r) for r in phase2
                       if _global_index(fake, r) < position and not _is_forced(r)}
            assert produced_by == earlier, "an earlier thinking block was not replayed"
    assert list(_thinking_blocks(phase2[2])), "nothing was replayed before the compaction"

    # The long request is in the system prompt, once, and not repeated as a turn.
    for request in phase2:
        assert _occurrences(request, LAST_SENTENCE) == 1
        assert LAST_SENTENCE in plain(request["system"])[0]["text"]
    assert _first_text(phase2[0]) == LLMOrchestrator._KICKOFF_POINTER
    _assert_usage_matches_script(client, model, fake)


def test_an_older_claude_run_streams_and_forces_the_tool(tmp_path):
    """Sonnet 4.6: a forced tool_choice with thinking off; a short request is the first turn."""
    orchestrator, model, fake, client = _run(
        tmp_path, _claude("claude-sonnet-4-6", thinking=False), SHORT_REQUEST, streaming=True)
    phase2 = fake.of("phase2")
    _assert_scripted_run_completed(orchestrator, phase2)
    _assert_cache_prefix_stable(phase2, orchestrator._compaction_count)

    forced = [request for request in phase2 if _is_forced(request)]
    assert [request["tool_choice"] for request in forced] == [{"type": "tool", "name": "read_file"}]
    assert forced[0]["thinking"] == {"type": "disabled"}
    compacted = False
    for index, request in enumerate(phase2):
        assert count_cache_markers(plain(request)) <= 4
        assert _pairing_problems_anthropic(request) == [], index
        compacted = compacted or _is_compaction(request)
        if not compacted:
            assert _first_text(request) == SHORT_REQUEST
    _assert_usage_matches_script(client, model, fake)


def test_a_crashed_claude_run_resumes_with_a_valid_first_request(tmp_path):
    """Crash on the request after the forced read, then resume on a fresh orchestrator."""
    script = shelf_script()
    # Plan, three refused edits, the forced read. Crashing after the corrected
    # edit would make leg 2's evidence for it refused: the executor's record of
    # this run's writes is not checkpointed (see the module docstring).
    done_before_crash = 5
    leg1, model1, fake1, _ = _run(tmp_path, _claude("claude-sonnet-5-5", thinking=True), SHORT_REQUEST,
                                  script=script[:done_before_crash] + [CRASH])
    assert leg1._phase2_stop_reason == "api_error"
    saved = load_checkpoint(str(tmp_path))
    assert saved is not None and '"thinking"' in json.dumps(saved.messages)
    executed = len(saved.tool_calls_log)

    leg2, model2, fake2, client2 = _run(tmp_path, _claude("claude-sonnet-5-5", thinking=True),
                                        SHORT_REQUEST, script=script[done_before_crash:], resume=True)

    assert [r.kind for r in fake2.requests][0] == "phase2", "resume planned again"
    first = fake2.of("phase2")[0]
    assert list(_thinking_blocks(first)) == []
    assert _pairing_problems_anthropic(first) == []
    assert count_cache_markers(plain(first)) <= 4
    assert leg2._phase2_stop_reason == "completed"
    replayed = [call["tool"] for call in leg2.tool_calls_log[executed:]]
    assert replayed == ["modify_file", "write_file", "modify_file", "task_list", "validate_app", "task_list"]
    phase2 = fake2.of("phase2")
    assert all(reason == "compaction" for _, reason, _ in _rewrites(phase2)), _rewrites(phase2)
    assert _thinking_violations(fake2) == []


def test_the_cost_cap_stops_the_run_before_another_request(tmp_path):
    cap = 0.5
    # Planning calls cost nothing; each Phase 2 turn about $0.21 on Sonnet 5.5.
    expensive = lambda n: Usage(10, 1) if n < 2 else Usage(60_000, 2_000)  # noqa: E731
    orchestrator, model, fake, client = _run(
        tmp_path, _claude("claude-sonnet-5-5", thinking=True), SHORT_REQUEST,
        usage=expensive, max_cost_usd=cap)

    assert orchestrator._phase2_stop_reason == "cost_cap"
    assert client.usage.estimated_cost > cap
    assert all(cost <= cap for cost in fake.costs_at_send), fake.costs_at_send
    assert 1 < len(fake.of("phase2")) < len(shelf_script())
    _assert_usage_matches_script(client, model, fake)


# ----------------------------------------------------------------------
# OpenAI chat/completions
# ----------------------------------------------------------------------

def _openai_system(request):
    return request["messages"][0]


@pytest.mark.parametrize("streaming", [False, True])
def test_an_openai_run_keeps_its_prefix_and_pairs_every_tool_call(tmp_path, streaming):
    orchestrator, model, fake, client = _run(tmp_path, _openai("gpt-4.1"), SHORT_REQUEST, streaming=streaming)
    phase2 = fake.of("phase2")
    _assert_scripted_run_completed(orchestrator, phase2)
    _assert_cache_prefix_stable(phase2, orchestrator._compaction_count,
                                system=_openai_system, tools=lambda r: r["tools"])
    assert plain(_openai_system(phase2[0]))["role"] == "system"
    for index, request in enumerate(phase2):
        assert _pairing_problems_openai(request) == [], index
    forced = [request["tool_choice"] for request in phase2 if "tool_choice" in request]
    assert forced == [{"type": "function", "function": {"name": "read_file"}}]
    _assert_usage_matches_script(client, model, fake, net_cached=True)


@pytest.mark.parametrize("streaming", [False, True])
def test_openai_usage_is_estimated_when_the_endpoint_reports_none(tmp_path, streaming):
    orchestrator, model, fake, client = _run(tmp_path, _openai("gpt-4.1"), SHORT_REQUEST,
                                             streaming=streaming, usage=lambda n: False)
    assert model.scripted_usage == []
    assert client.usage.api_calls == len(fake.requests)
    sent = fake.costs_at_send
    assert all(before < after for before, after in zip(sent, sent[1:])), sent
    assert client.usage.estimated_cost > sent[-1]

