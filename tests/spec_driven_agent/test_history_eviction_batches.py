"""Stale file bodies leave the conversation in batches, never the text the model edits from.

Every turn re-sends the whole history. Without eviction, each read result and
each written body stays in it until the compaction threshold (80k tokens or
more), long after a later read or write made it stale. Evicting rewrites the
cached prompt prefix from the first stubbed block on, so it happens in batches:
nothing until the stale bodies are a large share of the history, then all of
them at once. The newest read of lines of a file not edited since is never
stubbed: the model quotes from it and edits with its read_id.
"""

import json

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.agent.history_eviction import (
    DEFAULT_STALE_AFTER_TURNS,
    EVICTED_EDIT_BODY,
    evict_superseded_file_bodies,
)
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import dumps_for_model
from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.settings import AgentSettings

BODY = "".join(f"    total_{n} = compute(order_{n})\n" for n in range(400))   # ~13k chars
ALL_AT_ONCE = {"batch_share": 0, "min_batch_chars": 0}


def _read(call_id, path, *, start=1, end=400, offset=None, limit=None, content=BODY):
    args = {"path": path, **({"offset": offset} if offset is not None else {}),
            **({"limit": limit} if limit is not None else {})}
    result = {"content": content, "read_id": f"e:{call_id}", "start_line": start, "end_line": end}
    return ("read_file", call_id, args, result)


def _edit(call_id, path, tool="modify_file", error=None):
    args = {"path": path, "old_text": BODY[:700], "new_text": BODY[:800]} if tool == "modify_file" \
        else {"path": path, "content": BODY}
    return (tool, call_id, args, {"error": error} if error else {"status": "modified", "path": path})


def _history(*turns, idle_turns=0):
    """A conversation: one assistant message per turn, each answered by its tool results."""
    messages = [{"role": "user", "content": "Build the order service."}]
    for turn in turns:
        calls = turn if isinstance(turn, list) else [turn]
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": name, "input": args}
            for name, call_id, args, _ in calls]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": json.dumps(result)}
            for _, call_id, _, result in calls]})
    for n in range(idle_turns):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"checking {n}"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})
    return messages


def _shown(messages, call_id) -> str:
    return next(json.dumps(block) for message in messages if isinstance(message["content"], list)
                for block in message["content"]
                if call_id in (block.get("id"), block.get("tool_use_id")) and block["type"] == "tool_result")


def _input(messages, call_id) -> dict:
    return next(block["input"] for message in messages if isinstance(message["content"], list)
                for block in message["content"] if block.get("id") == call_id)


# ----------------------------------------------------------------------
# What is stale
# ----------------------------------------------------------------------

def test_the_newest_read_of_a_file_not_edited_since_is_kept_however_old():
    messages = _history(_read("r1", "app/orders.py"), idle_turns=40)
    out, stubbed = evict_superseded_file_bodies(messages, **ALL_AT_ONCE)
    assert (out, stubbed) == (messages, 0)
    assert out is messages


def test_a_read_shown_again_by_a_later_read_is_stubbed_and_the_later_one_kept():
    messages = _history(_read("r1", "app/orders.py"), _read("r2", "app/orders.py"), idle_turns=4)
    out, stubbed = evict_superseded_file_bodies(messages, **ALL_AT_ONCE)
    assert stubbed == 1
    assert BODY[:200] not in _shown(out, "r1") and "elided" in _shown(out, "r1")
    assert json.loads(_shown(out, "r2"))["content"] == json.dumps({
        "content": BODY, "read_id": "e:r2", "start_line": 1, "end_line": 400})


def test_windows_of_a_file_are_kept_until_a_later_read_shows_their_lines():
    first = _read("w1", "app/orders.py", start=1, end=200, offset=0, limit=200, content=BODY[:6000])
    second = _read("w2", "app/orders.py", start=201, end=400, offset=200, limit=200, content=BODY[6000:])
    assert evict_superseded_file_bodies(_history(first, second, idle_turns=20), **ALL_AT_ONCE)[1] == 0

    whole = _history(first, second, _read("w3", "app/orders.py"), idle_turns=4)
    out, stubbed = evict_superseded_file_bodies(whole, **ALL_AT_ONCE)
    assert stubbed == 2 and "compute(order_0)" in _shown(out, "w3")


def test_a_read_of_a_file_edited_since_goes_once_it_is_old():
    young = _history(_read("r1", "app/orders.py"), _edit("e1", "app/orders.py"),
                     idle_turns=DEFAULT_STALE_AFTER_TURNS - 2)
    assert evict_superseded_file_bodies(young, **ALL_AT_ONCE)[1] == 0

    old = _history(_read("r1", "app/orders.py"), _edit("e1", "app/orders.py"),
                   idle_turns=DEFAULT_STALE_AFTER_TURNS)
    out, stubbed = evict_superseded_file_bodies(old, **ALL_AT_ONCE)
    assert stubbed == 2    # the read, and the edit's own body
    assert "elided" in _shown(out, "r1")
    assert _input(out, "e1") == {"path": "app/orders.py", "old_text": EVICTED_EDIT_BODY,
                                 "new_text": EVICTED_EDIT_BODY}


def test_a_read_replaced_by_a_later_write_file_is_stale_at_once():
    messages = _history(_read("r1", "app/orders.py"), _edit("e1", "app/orders.py", tool="write_file"),
                        idle_turns=3)
    out, stubbed = evict_superseded_file_bodies(messages, **ALL_AT_ONCE)
    assert stubbed == 1 and "elided" in _shown(out, "r1")
    assert _input(out, "e1")["content"] == BODY   # the newest text of the file stays


def test_a_refused_edit_is_neither_an_edit_nor_counted():
    messages = _history(_read("r1", "app/orders.py"), _edit("e1", "app/orders.py", error="old_text not found"),
                        idle_turns=DEFAULT_STALE_AFTER_TURNS + 2)
    assert evict_superseded_file_bodies(messages, **ALL_AT_ONCE) == (messages, 0)


def test_the_recent_turns_are_never_touched():
    messages = _history(_read("r1", "app/orders.py"), _read("r2", "app/orders.py"))
    assert evict_superseded_file_bodies(messages, **ALL_AT_ONCE) == (messages, 0)


# ----------------------------------------------------------------------
# Batches: the cached prefix is rewritten once per batch
# ----------------------------------------------------------------------

def _rereads(count, *, idle_turns=4):
    return _history(*(_read(f"r{n}", "app/orders.py") for n in range(count)), idle_turns=idle_turns)


def test_stale_bodies_wait_until_they_are_a_third_of_the_history():
    # One stale read in a history of six reads plus padding: under a third.
    few = _rereads(2) + _history(*(_read(f"o{n}", f"app/other{n}.py") for n in range(4)))[1:]
    assert evict_superseded_file_bodies(few) == (few, 0)

    many = _rereads(6)
    out, stubbed = evict_superseded_file_bodies(many)
    assert stubbed == 5


def test_after_a_batch_the_next_turns_are_left_alone_until_the_next_batch():
    batch, _ = evict_superseded_file_bodies(_rereads(6))
    # Two more turns, one of them a re-read that makes the previous newest read stale.
    later = batch + _history(_read("r6", "app/orders.py"), idle_turns=1)[1:]
    out, stubbed = evict_superseded_file_bodies(later)
    assert (stubbed, out is later) == (0, True), "a second rewrite of the prefix one turn later"


# ----------------------------------------------------------------------
# A placeholder is never written as code
# ----------------------------------------------------------------------

STYLES = ".card {\n  color: teal;\n}\n"


@pytest.mark.parametrize("tool", ["write_file", "modify_file"])
def test_the_evicted_body_placeholder_is_refused_as_file_content(tmp_path, tool):
    (tmp_path / "styles.css").write_text(STYLES, encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.execute_typed("read_file", {"path": "styles.css"})
    args = ({"path": "styles.css", "content": EVICTED_EDIT_BODY} if tool == "write_file"
            else {"path": "styles.css", "old_text": "  color: teal;", "new_text": EVICTED_EDIT_BODY})
    result = executor.execute_typed(tool, args).payload
    assert result.get("rejection_kind") == "placeholder", result
    assert (tmp_path / "styles.css").read_text(encoding="utf-8") == STYLES


# ----------------------------------------------------------------------
# In the loop: opt-in, recorded, and edits keep working
# ----------------------------------------------------------------------

def test_eviction_is_off_unless_turned_on(monkeypatch):
    """Replays of 13 recorded runs: 7.5% less input, about 2% more cost on a caching provider."""
    monkeypatch.delenv("BESSER_LLM_HISTORY_EVICTION", raising=False)
    assert AgentSettings().llm_history_eviction is False
    for off in ("0", "false", "no"):
        monkeypatch.setenv("BESSER_LLM_HISTORY_EVICTION", off)
        assert AgentSettings().llm_history_eviction is False
    for on in ("1", "true"):
        monkeypatch.setenv("BESSER_LLM_HISTORY_EVICTION", on)
        assert AgentSettings().llm_history_eviction is True


class _Client:
    model = "some-byok-model"
    max_tokens = 16_000

    def __init__(self):
        self.usage = UsageTracker("some-byok-model")


def _orchestrator(tmp_path):
    item = Class(name="Order")
    item.attributes = {Property(name="name", type=PrimitiveDataType("str"))}
    return LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={item}),
                           output_dir=str(tmp_path))


def _call(executor, messages, *calls):
    """One turn through the real executor: the model's calls and their results."""
    messages.append({"role": "assistant", "content": [
        {"type": "tool_use", "id": call_id, "name": name, "input": args} for call_id, name, args in calls]})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id,
         "content": dumps_for_model(executor.execute_typed(name, args).payload)}
        for call_id, name, args in calls]})
    return json.loads(messages[-1]["content"][-1]["content"])


def test_after_a_batch_the_kept_read_id_edits_and_a_stubbed_read_is_read_again_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "_HISTORY_EVICTION_ENABLED", True)
    (tmp_path / "orders.py").write_text(BODY, encoding="utf-8")
    (tmp_path / "customers.py").write_text(BODY.replace("order", "customer"), encoding="utf-8")
    orchestrator = _orchestrator(tmp_path)
    executor = orchestrator.executor
    messages = [{"role": "user", "content": "Build the order service."}]
    for n in range(5):    # the model reads the same window again and again
        _call(executor, messages, (f"o{n}", "read_file", {"path": "orders.py", "offset": 0, "limit": 300 + n}))
    kept = _call(executor, messages, ("c1", "read_file", {"path": "customers.py"}))
    for n in range(4):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"planning {n}"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})

    messages = orchestrator._maybe_compact(messages)

    assert all("elided" in _shown(messages, f"o{n}") for n in range(4))
    assert "compute(order_0)" in _shown(messages, "o4") and kept["read_id"] in _shown(messages, "c1")
    # The kept read's read_id still authorizes a range edit.
    edit = executor.execute_typed("replace_file_lines", {
        "path": "customers.py", "read_id": kept["read_id"], "start_line": 1, "end_line": 1,
        "new_text": "    total_0 = compute(customer_0, rounded=True)"}).payload
    assert edit.get("status") == "modified", edit
    # A window whose earlier copies were stubbed comes back in full, not as "unchanged since".
    again = executor.execute_typed("read_file", {"path": "orders.py", "offset": 0, "limit": 304}).payload
    assert again.get("content", "").startswith("1|") and not again.get("unchanged")

    records = [json.loads(line) for line in (tmp_path / ".besser_trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    [event] = [r["payload"] for r in records if r["event"] == "eviction"]
    assert event["reason"] == "superseded" and event["stubbed"] == 4
    assert event["tokens_after"] < event["tokens_before"]


def _idle(messages, turns):
    for n in range(turns):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"checking {n}"}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})


def test_at_the_compaction_threshold_the_newest_read_of_a_file_not_edited_is_kept(tmp_path, monkeypatch):
    """Over the threshold, eviction stubs what is stale even if young, but not the read the model
    works from: a read five turns old of a file never edited stays verbatim through the
    compaction that follows, as it does with eviction off."""
    monkeypatch.setattr(orchestrator_module, "_HISTORY_EVICTION_ENABLED", True)
    orchestrator = _orchestrator(tmp_path)
    customers = BODY.replace("order", "customer")
    messages = [{"role": "user", "content": "Build the order service."}]
    for n in range(60):    # earlier work, enough to pass the threshold
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": f"step {n} " + " ".join(f"w{n}q{i}" for i in range(700))}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": "go on"}]})
    messages += _history(_read("r1", "orders.py"), _edit("e1", "orders.py"),
                         _read("c1", "customers.py", content=customers))[1:]
    _idle(messages, 4)

    out = orchestrator._maybe_compact(messages)

    assert "compute(customer_200)" in _shown(out, "c1")
    assert "elided" in _shown(out, "r1"), "a read of a file edited since is stale at the threshold"
    records = [json.loads(line) for line in (tmp_path / ".besser_trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [r["payload"]["reason"] for r in records if r["event"] == "eviction"] == ["threshold"]


def test_a_file_whose_only_copy_was_stubbed_must_be_read_again_before_a_rewrite(tmp_path, monkeypatch):
    """write_file may replace an existing file only after the model has seen all of it. Once a
    batch stubs the last copy the conversation held, it has not: a rewrite would be from memory."""
    monkeypatch.setattr(orchestrator_module, "_HISTORY_EVICTION_ENABLED", True)
    body = "".join(f"def f_{n}(order):\n    return order.total_{n}\n\n" for n in range(500))
    (tmp_path / "orders.py").write_text(body, encoding="utf-8")
    (tmp_path / "customers.py").write_text(body.replace("order", "customer"), encoding="utf-8")
    orchestrator = _orchestrator(tmp_path)
    executor = orchestrator.executor
    messages = [{"role": "user", "content": "Build the order service."}]
    _call(executor, messages, ("r1", "read_file", {"path": "orders.py"}))
    _call(executor, messages, ("e1", "modify_file", {"path": "orders.py", "old_text": "def f_0(order):",
                                                     "new_text": "def f_0(order, *, rounded=False):"}))
    # customers.py: an older copy is stubbed, but a later whole read of it is still shown.
    _call(executor, messages, ("c1", "read_file", {"path": "customers.py"}))
    _call(executor, messages, ("e2", "modify_file", {"path": "customers.py", "old_text": "def f_0(customer):",
                                                     "new_text": "def f_0(customer, *, rounded=False):"}))
    _call(executor, messages, ("c2", "read_file", {"path": "customers.py"}))
    _idle(messages, 9)

    messages = orchestrator._maybe_compact(messages)
    assert "elided" in _shown(messages, "r1") and "elided" in _shown(messages, "c1")

    rewrite = executor.execute_typed("write_file", {"path": "orders.py", "content": "def f_0(order):\n    return 0\n"})
    assert "error" in rewrite.payload and "read_file" in rewrite.payload["error"], rewrite.payload
    assert "return order.total_250" in (tmp_path / "orders.py").read_text(encoding="utf-8")
    shown = executor.execute_typed("write_file", {"path": "customers.py", "content": "def g(customer):\n    return 0\n"})
    assert shown.payload.get("status") == "written", shown.payload


def test_a_rewrite_refused_after_eviction_says_the_whole_file_is_no_longer_shown(tmp_path):
    """The model wrote the file, so it never read part of it: the refusal names the real reason."""
    executor = ToolExecutor(workspace=str(tmp_path))
    assert executor.execute_typed("write_file", {"path": "orders.py", "content": BODY}).payload["status"] == "written"
    executor.forget_full_views(["orders.py"])

    error = executor.execute_typed("write_file", {"path": "orders.py", "content": "x = 1\n"}).payload["error"]

    assert "read only part" not in error
    assert "no longer shows" in error and "read_file" in error


def test_the_reread_rule_holds_whatever_spelling_the_history_used_for_the_path(tmp_path):
    """The history names the file as the model spelled it (``./orders.py``); the executor keys it
    by its workspace path."""
    executor = ToolExecutor(workspace=str(tmp_path))
    assert executor.execute_typed("write_file", {"path": "orders.py", "content": BODY}).payload["status"] == "written"
    executor.forget_full_views(["./orders.py"])

    refused = executor.execute_typed("write_file", {"path": "orders.py", "content": "x = 1\n"}).payload

    assert "no longer shows" in refused.get("error", ""), refused


def test_in_a_whole_run_the_prefix_is_rewritten_only_at_a_recorded_batch(tmp_path, monkeypatch):
    """The real loop on a scripted model that re-reads two scaffold files: batches stub the four
    superseded copies, the requests at those batches are the only ones that are not the previous
    request plus new messages, and an edit quoting the kept read still lands."""
    from tests.spec_driven_agent.run_transport import plain, without_cache_control
    from tests.spec_driven_agent.scripted_rig import ScriptedAgent, run_scripted

    monkeypatch.setattr(orchestrator_module, "_HISTORY_EVICTION_ENABLED", True)
    monkeypatch.setattr(LLMOrchestrator, "_runtime_probe_issues", lambda self: [])
    router, other = "backend/routers/book.py", "backend/routers/library.py"
    rereads = [("read_file", {"path": router, "offset": 0, "limit": 1000 + n}) for n in range(4)]
    rereads += [("read_file", {"path": other, "offset": 0, "limit": 1000 + n}) for n in range(2)]
    edit = ("modify_file", {"path": router, "old_text": "from fastapi import",
                            "new_text": "# Book routes.\nfrom fastapi import"})
    agent = ScriptedAgent(str(tmp_path), [("list_files", {}), *rereads, ("list_files", {}),
                                          ("list_files", {}), ("list_files", {}), edit],
                          ["Add a comment to the book router"])
    run_scripted(tmp_path, agent)

    records = [json.loads(line) for line in (tmp_path / ".besser_trace.jsonl").read_text(
        encoding="utf-8").splitlines()]
    evictions = [r["payload"] for r in records if r["event"] == "eviction"]
    assert {e["reason"] for e in evictions} == {"superseded"} and sum(e["stubbed"] for e in evictions) == 4
    assert len(evictions) < 4, "one prefix rewrite per stale body, not per batch"
    requests = [plain(without_cache_control(kwargs["messages"])) for kwargs in agent.transport.of("phase2")]
    rewritten = [index + 1 for index in range(1, len(requests))
                 if requests[index][:len(requests[index - 1]) - 1] != requests[index - 1][:-1]]
    assert rewritten == [e["turn"] for e in evictions]
    assert "# Book routes." in (tmp_path / router).read_text(encoding="utf-8")
