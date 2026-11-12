"""Every history rewrite the agent applies must leave a request both providers accept.

The Phase 2 / Phase 3 loops rewrite the conversation in several places:
keep-tail compaction, stale-body eviction, the request-only projection of
rejected edit drafts, the thinking-block strip, the checkpoint JSON round trip,
and then each client's request shaping (Anthropic cache breakpoints and the
forced-tool instruction; the OpenAI chat translation). These tests drive
seeded random runs through the REAL functions in the order the orchestrator
calls them and check every request against ``message_validity``:

* structural validity for the Anthropic Messages API and OpenAI Chat
  Completions;
* the preserved-thinking rule: a thinking block replayed to Claude must sit
  behind exactly the messages it was produced after (otherwise the current
  models answer 400, "bound to a different conversation");
* prompt-cache reuse: a turn that rewrote nothing must read the previous
  request's breakpoint, i.e. the shaping never changes an already-sent message.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic.types as at
import pytest

from besser.spec_driven_agent.agent import compaction
from besser.spec_driven_agent.agent.compaction import maybe_compact
from besser.spec_driven_agent.agent.history_eviction import (
    evict_stale_file_bodies,
    without_rejected_edit_drafts,
    without_thinking_blocks,
)
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers import llm_client
from besser.spec_driven_agent.providers.llm_client import (
    ClaudeLLMClient,
    _openai_messages_to_api,
    _openai_response_to_common,
    _with_message_cache,
)
from besser.spec_driven_agent.state.checkpoint import _to_wire

from tests.spec_driven_agent.message_validity import (
    ANTHROPIC_MAX_CACHE_BREAKPOINTS,
    anthropic_violations,
    block_get,
    block_type,
    count_cache_breakpoints,
    openai_violations,
)

SYSTEM = "You customise a generated app."
TOOLS = [
    {"name": name, "description": name, "input_schema": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"},
        "old_text": {"type": "string"}, "new_text": {"type": "string"},
        "action": {"type": "string"}, "id": {"type": "integer"},
        "command": {"type": "string"}}}}
    for name in ("read_file", "write_file", "modify_file", "task_list", "run_command")
]
_TOOL_NAMES = [tool["name"] for tool in TOOLS]
_PATHS = ["backend/main.py", "backend/models.py", "frontend/src/App.tsx", "README.md"]
# Thresholds (tokens, counted as characters / 4 below) the compaction check
# runs at: tight enough that most runs compact, several times.
_THRESHOLDS = (8_000, 12_000, 20_000, 40_000)

SEEDS = range(60)


# ----------------------------------------------------------------------
# Canonical wire form (what the provider renders, cache markers aside)
# ----------------------------------------------------------------------

def _strip_cache_control(value):
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(v) for v in value]
    return value


def _wire_blocks(message: dict) -> list:
    content = message.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    return [_strip_cache_control(_to_wire(block)) for block in blocks or []]


def _flat(messages: list[dict]) -> list[str]:
    """One JSON string per block, tagged with its message index and role."""
    flat = []
    for index, message in enumerate(messages):
        for block in _wire_blocks(message):
            flat.append(json.dumps([index, message.get("role"), block], sort_keys=True))
    return flat


def _thinking_prefix(messages: list[dict]) -> str:
    """The conversation a thinking block is bound to: every earlier message,
    earlier thinking blocks and cache markers excluded."""
    return json.dumps([
        [message.get("role"), [b for b in _wire_blocks(message)
                               if b.get("type") not in ("thinking", "redacted_thinking")]]
        for message in messages
    ], sort_keys=True)


def _breakpoints(messages: list[dict]) -> list[int]:
    """Flat positions carrying a cache_control marker."""
    positions, position = [], 0
    for message in messages:
        content = message.get("content")
        for block in content if isinstance(content, list) else [content]:
            if isinstance(block, dict) and block.get("cache_control"):
                positions.append(position)
            position += 1
    return positions


# ----------------------------------------------------------------------
# A run: the orchestrator's loop with a scripted model
# ----------------------------------------------------------------------

def _body(rng: random.Random) -> str:
    size = rng.choice((40, 300, 900, 4_000, 15_000, 30_000))
    line = "def handler(request):  # %d\n" % rng.randrange(10_000)
    return (line * (size // len(line) + 1))[:size]


def _tool_input(rng: random.Random, name: str) -> dict:
    path = rng.choice(_PATHS)
    if name == "read_file":
        return {"path": path}
    if name == "write_file":
        return {"path": path, "content": _body(rng)}
    if name == "modify_file":
        return {"path": path, "old_text": _body(rng), "new_text": _body(rng)}
    if name == "task_list":
        return {"action": rng.choice(["done", "list"]), "id": rng.randrange(1, 9)}
    return {"command": "pytest -q"}


def _tool_result(rng: random.Random, name: str, args: dict) -> str:
    if name == "read_file":
        return json.dumps({"path": args["path"], "content": _body(rng)})
    if name in ("write_file", "modify_file"):
        roll = rng.random()
        if roll < 0.3:
            return json.dumps({"error": "old_text not found in the file", "rejection_kind": "no_match"})
        if roll < 0.45:
            return json.dumps({"error": "syntax error at line 3", "rejection_kind": "syntax_error",
                               "would_write": _body(rng)})
        return json.dumps({"success": True, "path": args["path"]})
    if name == "task_list":
        if rng.random() < 0.3:
            return json.dumps({"error": "evidence required"})
        return json.dumps({"ok": True, "open": rng.randrange(5)})
    return json.dumps({"exit_code": rng.choice([0, 1]), "output": _body(rng)[:2_000]})


class _ScriptedRun:
    """Plays the model: emits turns and remembers which request each
    thinking block was produced for."""

    def __init__(self, rng: random.Random, provider: str, *, reuse_ids: bool = False):
        self.rng = rng
        self.provider = provider
        # An OpenAI-compatible gateway that numbers its call ids per reply
        # (call_0, call_1, ...): legal, results pair with the call before them.
        self.reuse_ids = reuse_ids
        self.ids = 0
        self.thinking: dict[str, tuple[str, str | None]] = {}
        self.force: str | None = None

    def _id(self) -> str:
        self.ids += 1
        return f"toolu_{self.ids:04d}" if self.provider == "anthropic" else f"call_{self.ids:04d}"

    def turn_plan(self) -> tuple[str, bool, bool, list[str]]:
        """(stop_reason, thinking?, text?, tool names)."""
        rng = self.rng
        roll = rng.random()
        thinking = self.provider == "anthropic" and rng.random() < 0.6
        if self.force or roll < 0.72:
            names = [rng.choice(_TOOL_NAMES) for _ in range(rng.choice((1, 1, 2, 3, 5, 12)))]
            if self.force and rng.random() < 0.85:
                names[0] = self.force
            return "tool_use", thinking, rng.random() < 0.5, names
        if roll < 0.9:
            # end_turn: sometimes prose, sometimes nothing at all.
            return "end_turn", thinking and rng.random() < 0.5, rng.random() < 0.6, []
        return "max_tokens", thinking, rng.random() < 0.5, []

    def anthropic_response(self, request: dict):
        stop, thinking, text, names = self.turn_plan()
        content: list = []
        if thinking:
            signature = f"sig-{len(self.thinking)}"
            previous = [block_get(b, "signature") for m in request["messages"]
                        if m.get("role") == "assistant" and isinstance(m.get("content"), list)
                        for b in m["content"] if block_type(b) == "thinking"]
            self.thinking[signature] = (_thinking_prefix(request["messages"]),
                                        previous[-1] if previous else None)
            content.append(at.ThinkingBlock(type="thinking", thinking="", signature=signature))
        if text:
            content.append(at.TextBlock(type="text", text="Working on it."))
        for name in names:
            content.append(at.ToolUseBlock(type="tool_use", id=self._id(), name=name,
                                           input=_tool_input(self.rng, name)))
        return SimpleNamespace(stop_reason=stop, content=content, stop_details=None,
                               usage=at.Usage(input_tokens=1, output_tokens=1))

    def openai_response(self):
        stop, _, text, names = self.turn_plan()
        calls = [SimpleNamespace(id=f"call_{i}" if self.reuse_ids else self._id(), function=SimpleNamespace(
            name=name, arguments=json.dumps(_tool_input(self.rng, name)))) for i, name in enumerate(names)]
        finish = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}[stop]
        message = SimpleNamespace(content="Working on it." if text else None, tool_calls=calls or None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])
        return _openai_response_to_common(response, TOOLS)


class _CompactionHost:
    """The orchestrator's own ``_maybe_compact`` and thinking strip, on a stub."""

    _maybe_compact = LLMOrchestrator._maybe_compact
    _drop_stale_thinking = LLMOrchestrator._drop_stale_thinking

    def __init__(self, output_dir: str):
        self.client = SimpleNamespace(model="scripted", max_tokens=0)
        self.tool_calls_log: list[dict] = []
        self.output_dir = output_dir
        self.domain_model = self.gui_model = self.agent_model = None
        self.state_machines = self.object_model = self.quantum_circuit = None
        self.bpmn_model = self.nn_model = self.primary_kind = None
        self._compaction_count = 0
        self.executor = SimpleNamespace(forget_shown_results=lambda: None)

    def _compaction_work_state(self) -> dict:
        return {"tasks": [{"id": 1, "text": "Add login"}]}


@pytest.fixture
def fast_tokens(monkeypatch):
    """Count tokens as characters / 4: the fuzz sends megabytes through it."""
    monkeypatch.setattr(compaction, "_TOKENIZER", None)
    monkeypatch.setattr(compaction, "_TOKENIZER_LOADED", True)


@pytest.fixture
def threshold(monkeypatch):
    holder = {"value": _THRESHOLDS[0]}

    def fixed(model, threshold=0, reserve=0):
        return holder["value"]

    monkeypatch.setattr(compaction, "effective_threshold", fixed)
    monkeypatch.setattr(orchestrator_module, "effective_threshold", fixed)
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    return holder


class _Checker:
    """Checks every request a run sends; collects violations."""

    def __init__(self, script: _ScriptedRun):
        self.script = script
        self.failures: list[str] = []
        self.previous: list[dict] | None = None
        self.rewritten = False
        self.requests = 0
        self.requests_turn = -1
        self.cache_reads = 0
        self.compactions = 0
        self.snapshots: list[list[dict]] = []
        self.previous_projection: list[dict] | None = None

    def fail(self, what: str, problems: list[str], messages: list[dict]) -> None:
        if problems:
            roles = [m.get("role") + ":" + ",".join(str(block_type(b)) for b in
                     (m["content"] if isinstance(m.get("content"), list) else ["str"]))
                     for m in messages]
            self.failures.append(f"request {self.requests} {what}: {problems[:3]} roles={roles}")

    def anthropic_request(self, kwargs: dict, *, retry: bool) -> None:
        self.requests += 1
        messages = kwargs["messages"]
        self.fail("anthropic", anthropic_violations(kwargs), messages)
        self._check_thinking_binding(messages)
        if (kwargs.get("tool_choice") or {}).get("type") == "tool":
            # A native forced tool_choice neither reads nor writes the
            # conversation cache: the next turn reads the entry before it.
            if _breakpoints(messages):
                self.fail("cache", ["forced tool_choice request marks the conversation"], messages)
            return
        if not retry:
            self._check_cache_reuse(messages)
            self.previous = messages
            self.rewritten = False

    def projection(self, history: list[dict], projected: list[dict], *, rewritten: bool) -> None:
        """The request-only projection never changes an already-sent message:
        all of the previous projection, but at most its last message, must
        open this one unchanged."""
        self.snapshots.append(list(history))
        # A copy: with nothing to project the history list itself comes
        # back, and the loop appends to it.
        previous, self.previous_projection = self.previous_projection, list(projected)
        if previous is None or rewritten:
            return
        stable = _flat(previous[:-1])
        if _flat(projected)[:len(stable)] != stable:
            self.fail("projection", ["the projection rewrote an already-sent message"], projected)

    def _check_thinking_binding(self, messages: list[dict]) -> None:
        present: list[str] = []
        for index, message in enumerate(messages):
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if block_type(block) != "thinking":
                    continue
                signature = block_get(block, "signature")
                bound_prefix, predecessor = self.script.thinking[signature]
                if _thinking_prefix(messages[:index]) != bound_prefix:
                    self.fail("thinking", [f"messages.{index}: thinking {signature} replayed after a "
                                           "different conversation than it was produced in"], messages)
                if predecessor is not None and predecessor not in present and present:
                    self.fail("thinking", [f"messages.{index}: thinking {signature} kept after its "
                                           f"predecessor {predecessor} was removed"], messages)
                present.append(signature)

    def _check_cache_reuse(self, messages: list[dict]) -> None:
        previous = self.previous
        if previous is None or self.rewritten or not _breakpoints(previous):
            # Nothing to read: the history was rewritten, or the previous
            # request was a forced call on a single message, which is sent
            # unmarked on purpose (a one-shot planning prompt, see
            # ``ClaudeLLMClient.chat``).
            return
        before, after = _flat(previous), _flat(messages)
        shared = 0
        while shared < min(len(before), len(after)) and before[shared] == after[shared]:
            shared += 1
        readable = [p for p in _breakpoints(previous) if p < shared]
        last_message_start = len(_flat(previous[:-1]))
        if not readable or readable[-1] + 1 < last_message_start:
            self.fail("cache", [f"no breakpoint of the previous request is reusable (shared "
                                f"prefix {shared} of {len(before)} blocks)"], messages)
        else:
            self.cache_reads += 1


def _run(seed: int, provider: str, tmp_path, threshold, *, evict: bool, monkeypatch,
         reuse_ids: bool = False) -> _Checker:
    rng = random.Random(seed)
    script = _ScriptedRun(rng, provider, reuse_ids=reuse_ids)
    checker = _Checker(script)
    host = _CompactionHost(str(tmp_path))
    monkeypatch.setattr(orchestrator_module, "_HISTORY_EVICTION_ENABLED", evict)
    threshold["value"] = rng.choice(_THRESHOLDS)

    model = rng.choice(["claude-opus-5-5", "claude-sonnet-5"])
    claude = ClaudeLLMClient(api_key="sk-ant-test", model=model, max_retries=0)
    claude._client = MagicMock()
    sent: list[dict] = []

    def create(**kwargs):
        retry = bool(sent) and sent[-1].get("_turn") == checker.requests_turn
        kwargs = dict(kwargs)
        checker.anthropic_request(kwargs, retry=retry)
        sent.append({"_turn": checker.requests_turn})
        return script.anthropic_response(kwargs)

    claude._client.messages.create.side_effect = create

    first = "Build a library app." if rng.random() < 0.5 else "x" * 3_000
    messages: list[dict] = [{"role": "user", "content": first}]
    if rng.random() < 0.3:  # Phase 3 shape: the toolchain reminder follows the prompt
        messages.append({"role": "user", "content": [{"type": "text", "text": "<system-reminder>tsc failed</system-reminder>"}]})
    inspection_nudged = resumed = False

    for turn in range(rng.randrange(15, 45)):
        checker.requests_turn = turn
        before = len(json.dumps(_to_wire(messages)))
        compacted = host._maybe_compact(messages)
        rewritten = compacted is not messages or resumed
        resumed = False
        if compacted is not messages:
            checker.rewritten = True
            assert len(json.dumps(_to_wire(compacted))) <= before
        messages = compacted

        request = without_rejected_edit_drafts(messages)
        checker.projection(messages, request, rewritten=rewritten)
        api = _openai_messages_to_api(SYSTEM, request)
        checker.fail("openai", openai_violations(api), request)
        if not reuse_ids:  # ids repeated across turns never reach Anthropic
            checker.fail("anthropic-shape", anthropic_violations(
                {"system": SYSTEM, "tools": TOOLS, "messages": request}), request)

        force = script.force = ("read_file" if rng.random() < 0.15 else None)
        if provider == "anthropic":
            response = claude.chat(system=SYSTEM, messages=request, tools=TOOLS, force_tool=force)
        else:
            response = script.openai_response()
        script.force = None

        stop = response["stop_reason"]
        if stop == "end_turn":
            if rng.random() < 0.15:
                break
            if response["content"]:
                messages.append({"role": "assistant", "content": response["content"]})
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                "You ended the turn, but these checklist items are still OPEN:\n  1. Add login")}]})
            continue
        if stop in ("max_tokens", "length"):
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                "Your previous response was CUT OFF at the output token limit.")}]})
            continue
        assert stop == "tool_use", stop
        messages.append({"role": "assistant", "content": response["content"]})
        blocks = [b for b in response["content"]
                  if hasattr(b, "type") and b.type == "tool_use" and getattr(b, "name", None)]
        results = []
        for block in blocks:
            host.tool_calls_log.append({"turn": turn, "tool": block.name, "input": block.input})
            result = {"type": "tool_result", "tool_use_id": block.id,
                      "content": _tool_result(rng, block.name, block.input)}
            results.append(result)
        messages.append({"role": "user", "content": results})
        if not inspection_nudged and rng.random() < 0.1:
            inspection_nudged = True
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                "<system-reminder>You have inspected without changing source.</system-reminder>")}]})
        for _ in range(rng.choice((0, 0, 0, 1, 2))):
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                "<system-reminder>modify_file on `backend/main.py` was rejected 3 times.</system-reminder>")}]})
        if rng.random() < 0.06:
            # Crash and resume: the checkpoint is a JSON round trip and the
            # resumed system prompt is rebuilt, so all thinking is dropped.
            messages = json.loads(json.dumps(_to_wire(messages), default=str))
            messages, _ = without_thinking_blocks(messages)
            checker.rewritten = resumed = True
    checker.compactions = host._compaction_count
    return checker


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_every_request_of_a_run_is_valid_and_cache_friendly(
        seed, provider, tmp_path, fast_tokens, threshold, monkeypatch):
    checker = _run(seed, provider, tmp_path, threshold, evict=bool(seed % 2), monkeypatch=monkeypatch)
    assert not checker.failures, "\n".join(checker.failures[:5])


@pytest.mark.parametrize("seed", range(20))
def test_a_gateway_that_reuses_call_ids_gets_valid_requests(
        seed, tmp_path, fast_tokens, threshold, monkeypatch):
    checker = _run(seed, "openai", tmp_path, threshold, evict=True, monkeypatch=monkeypatch,
                   reuse_ids=True)
    assert not checker.failures, "\n".join(checker.failures[:5])


def test_the_fuzz_reaches_the_interesting_shapes(tmp_path, fast_tokens, threshold, monkeypatch):
    """Guard against a generator that silently stopped producing what it is for."""
    compactions = cache_reads = 0
    for seed in range(10):
        checker = _run(seed, "anthropic", tmp_path, threshold, evict=True, monkeypatch=monkeypatch)
        compactions += checker.compactions
        cache_reads += checker.cache_reads
    assert compactions >= 10
    assert cache_reads >= 100


# ----------------------------------------------------------------------
# Direct properties of each rewrite, over the histories the runs produced
# ----------------------------------------------------------------------

_HISTORIES: list[list[dict]] = []


def _recorded_histories(tmp_path_factory) -> list[list[dict]]:
    """Every third history a request was built from in a few anthropic and
    openai runs (consecutive ones share most of their messages)."""
    if not _HISTORIES:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(compaction, "_TOKENIZER", None)
            patch.setattr(compaction, "_TOKENIZER_LOADED", True)
            patch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
            holder = {"value": 0}

            def fixed(model, threshold=0, reserve=0):
                return holder["value"]

            patch.setattr(compaction, "effective_threshold", fixed)
            patch.setattr(orchestrator_module, "effective_threshold", fixed)
            for seed in range(8):
                for provider in ("anthropic", "openai"):
                    checker = _run(seed, provider, str(tmp_path_factory.mktemp("run")), holder,
                                   evict=bool(seed % 2), monkeypatch=patch)
                    _HISTORIES.extend(checker.snapshots[::3])
    return _HISTORIES


def _wire_snapshot(messages: list[dict]) -> str:
    return json.dumps(_to_wire(messages), sort_keys=True, default=str)


def _assert_valid(messages: list[dict], where: str) -> None:
    request = without_rejected_edit_drafts(messages)
    shaped = {"system": SYSTEM, "tools": llm_client._with_tool_cache(TOOLS),
              "messages": _with_message_cache(request)}
    assert not anthropic_violations(shaped), (where, anthropic_violations(shaped)[:3])
    api = _openai_messages_to_api(SYSTEM, request)
    assert not openai_violations(api), (where, openai_violations(api)[:3])


def test_no_rewrite_mutates_the_history_it_is_given(tmp_path_factory, fast_tokens, monkeypatch):
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    output_dir = str(tmp_path_factory.getbasetemp())
    histories = _recorded_histories(tmp_path_factory)
    assert len(histories) > 100
    for history in histories:
        before = _wire_snapshot(history)
        _with_message_cache(history)
        without_rejected_edit_drafts(history)
        evict_stale_file_bodies(history, preserve_recent=0)
        without_thinking_blocks(history)
        maybe_compact(history, [], output_dir, threshold=1, preserve_tokens=500)
        assert _wire_snapshot(history) == before


def test_the_rolling_cache_adds_exactly_one_marker(tmp_path_factory, monkeypatch):
    """With the system and tools markers that makes 3 of the API's 4."""
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    for history in _recorded_histories(tmp_path_factory):
        shaped = _with_message_cache(without_rejected_edit_drafts(history))
        request = {"system": llm_client._with_cache_control(SYSTEM),
                   "tools": llm_client._with_tool_cache(TOOLS), "messages": shaped}
        assert len(_breakpoints(shaped)) == 1
        assert count_cache_breakpoints(request) == 3 <= ANTHROPIC_MAX_CACHE_BREAKPOINTS


@pytest.mark.parametrize("preserve_recent", [0, 2, 6])
def test_eviction_is_idempotent(tmp_path_factory, monkeypatch, preserve_recent):
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    for history in _recorded_histories(tmp_path_factory):
        once, _ = evict_stale_file_bodies(history, preserve_recent=preserve_recent)
        twice, count = evict_stale_file_bodies(once, preserve_recent=preserve_recent)
        assert count == 0 and twice is once
        _assert_valid(once, "evicted")


@pytest.mark.parametrize("threshold_tokens", [1, 2_000, 8_000])
@pytest.mark.parametrize("preserve", [0, 1, 500, 4_000, 20_000])
def test_compaction_at_any_budget_stays_valid_and_keeps_its_tail_verbatim(
        tmp_path_factory, fast_tokens, monkeypatch, threshold_tokens, preserve):
    """Including a second compaction of an already compacted history, and
    eviction first (the orchestrator's order)."""
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    output_dir = str(tmp_path_factory.getbasetemp())
    for history in _recorded_histories(tmp_path_factory):
        for start in (history, evict_stale_file_bodies(history, preserve_recent=2)[0]):
            current = start
            for _ in range(2):
                compacted, did = maybe_compact(current, [], output_dir, threshold=threshold_tokens,
                                               preserve_tokens=preserve)
                if not did:
                    assert compacted is current
                    break
                assert compacted[0]["content"].startswith("[Earlier work summarized]")
                kept = [m for m in compacted[1:] if not (
                    m.get("role") == "assistant" and m["content"] == [{"type": "text", "text": "Continuing."}])]
                assert all(a is b for a, b in zip(kept, current[len(current) - len(kept):]))
                compacted, _ = without_thinking_blocks(compacted)
                _assert_valid(compacted, f"compacted at {threshold_tokens}/{preserve}")
                current = compacted


def test_an_empty_end_turn_reply_leaves_two_user_turns_both_providers_accept(monkeypatch):
    """The orchestrator does not replay an empty reply (no text, no calls), so
    the nudge follows the tool results directly: consecutive user messages,
    which the Messages API merges (results first) and Chat Completions takes
    as tool messages then a user message. Deliberate, and valid."""
    monkeypatch.setattr(llm_client, "_ROLLING_MESSAGE_CACHE", True)
    use = at.ToolUseBlock(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"})
    history = [
        {"role": "user", "content": "build"},
        {"role": "assistant", "content": [use]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]},
        {"role": "user", "content": [{"type": "text", "text": "You ended the turn, but ..."}]},
    ]
    _assert_valid(history, "empty end_turn")
    api = _openai_messages_to_api(SYSTEM, history)
    assert [m["role"] for m in api] == ["system", "user", "assistant", "tool", "user"]
