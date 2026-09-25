"""When the executor rejects the same edit again, the orchestrator changes
what the model can DO, in steps, instead of repeating what it is told.

Two recorded runs (13 identical misses; 16 identical no-op calls) alternated read_file / modify_file for ~30 turns each
while every guard either never fired or was ignored. Aider's answer is
``max_reflections = 3`` and then the human decides; headless, the runtime
has to offer a different editing strategy, then stop only if the model
continues ignoring recovery. A failed quotation must not freeze the file.
"""
from __future__ import annotations

import json
import os

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator


@pytest.fixture
def simple_model():
    user = Class(name="User")
    user.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="name", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="TestModel", types={user})


class MockBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class StuckClient:
    """Sends the same no-op modify_file forever; honours nothing it is told."""
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def __init__(self):
        self.turns = []          # (turn, force_tool)
        self.texts_seen = []
        self.tool_results = []   # (turn, tool_result JSON as the model sees it)

    def chat(self, system, messages, tools, force_tool=None, model_override=None):
        n = len(self.turns) + 1
        self.turns.append((n, force_tool))
        for m in messages:
            c = m.get("content")
            if m.get("role") == "user" and isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        self.texts_seen.append(b["text"])
                    elif isinstance(b, dict) and b.get("type") == "tool_result":
                        self.tool_results.append((n, b.get("content")))
        if force_tool == "task_list":
            return {"stop_reason": "tool_use", "content": [
                MockBlock("tool_use", name="task_list", id=f"t{n}", input={"action": "list"}),
            ]}
        return {"stop_reason": "tool_use", "content": [
            MockBlock("tool_use", name="modify_file", id=f"m{n}",
                      input={"path": "app.py", "old_text": "x = 1\n", "new_text": "x = 1\n"}),
        ]}


def _run(simple_model, tmp_path, max_turns=20):
    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    client = StuckClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_model,
                           output_dir=str(tmp_path), max_turns=max_turns)
    orch.run("Build an app")
    return orch, client


def test_two_refusals_escalate_to_a_rewrite_then_force_a_fresh_read(simple_model, tmp_path):
    """Renamed from ``test_two_refusals_force_a_fresh_read_on_the_next_turn``.

    Two refusals still change the strategy on the very next turn, and the
    new strategy still starts with a read - but it is now carried as the
    tool result's ``edit_recovery`` rather than as a forced ``read_file``.
    The orchestrator only forces a tool for ``read_file`` /
    ``replace_file_lines`` / ``modify_file`` (see the allowlist beside
    ``_force_tool_next``), and ``write_file`` is deliberately not forceable
    - forcing it would order a rewrite before the read the instruction
    requires.

    So the hard steer arrives one refusal later, from the orchestrator's
    own ``_REPEAT_FORCE_AT = 3``: turn 4 instead of turn 3. Both halves are
    asserted, because dropping either one is how this escalation could
    silently stop happening. The surviving ``replace_file_lines``
    assertion is that orchestrator reminder's wording, which the ladder
    change did not update; it is left in place deliberately so that
    re-pointing it at the rewrite shows up here as a failing test rather
    than as two guards quietly disagreeing.
    """
    orch, client = _run(simple_model, tmp_path)

    escalated = [n for n, content in client.tool_results
                 if isinstance(content, str) and '"next_tool": "write_file"' in content]
    assert escalated, client.tool_results
    assert escalated[0] == 3, "refusals on turns 1 and 2 -> rewrite hint delivered on turn 3"
    hint = next(c for n, c in client.tool_results
                if isinstance(c, str) and '"next_tool": "write_file"' in c)
    assert "read_file on the WHOLE file" in hint

    forced = [n for n, f in client.turns if f == "read_file"]
    assert forced, client.turns
    assert forced[0] == 4, "the hard steer is _REPEAT_FORCE_AT, one refusal later"
    assert any("replace_file_lines" in t and "app.py" in t for t in client.texts_seen)


def test_ignoring_recovery_is_bounded_but_does_not_freeze_the_file(simple_model, tmp_path):
    orch, client = _run(simple_model, tmp_path)
    assert orch._phase2_stop_reason == "stuck_edit_loop"
    assert len(client.turns) < 12, f"the loop ran {len(client.turns)} turns"
    assert not orch.executor._frozen("app.py")
    assert any("remains editable" in t and "app.py" in t for t in client.texts_seen)
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_successful_insertion_then_replay_is_bounded(simple_model, tmp_path):
    class RepeatingInsertion(StuckClient):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            for block in result["content"]:
                if block.name == "modify_file":
                    block.input["new_text"] = "y = 2\nx = 1\n"
            return result

    (tmp_path / "app.py").write_text("x = 1\n")
    client = RepeatingInsertion()
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_model,
                           output_dir=str(tmp_path), max_turns=100)
    orch.run("Build an app")
    assert orch._phase2_stop_reason == "stuck_edit_loop"
    assert len(client.turns) < 13
    assert (tmp_path / "app.py").read_text() == "y = 2\nx = 1\n"


def test_a_client_without_force_tool_still_gets_the_message_and_the_stop(simple_model, tmp_path):
    class PlainClient(StuckClient):
        def chat(self, system, messages, tools):        # no force_tool parameter
            return super().chat(system, messages, tools)

    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    client = PlainClient()
    orch = LLMOrchestrator(llm_client=client, domain_model=simple_model,
                           output_dir=str(tmp_path), max_turns=20)
    orch.run("Build an app")
    assert orch._phase2_stop_reason == "stuck_edit_loop"
    assert any("replace_file_lines" in t for t in client.texts_seen)


def test_untruncated_edit_inputs_are_kept_beside_the_trace(simple_model, tmp_path):
    """Item 1's other half: the trace/checkpoint/recipe are bounded at 500
    chars, so the full bytes of every edit go to a sidecar the fixtures can
    be cut from."""
    long_block = "x = 1\n" + ("# padding line to push this input over the log budget\n" * 12)
    with open(os.path.join(tmp_path, "app.py"), "w", encoding="utf-8") as f:
        f.write(long_block)

    class OneEdit(StuckClient):
        def chat(self, system, messages, tools, force_tool=None, model_override=None):
            n = len(self.turns) + 1
            self.turns.append((n, force_tool))
            if n > 1:
                return {"stop_reason": "end_turn", "content": [MockBlock("text", text="done")]}
            return {"stop_reason": "tool_use", "content": [
                MockBlock("tool_use", name="modify_file", id="m1",
                          input={"path": "app.py", "old_text": long_block,
                                 "new_text": long_block + "y = 2\n"}),
            ]}

    LLMOrchestrator(llm_client=OneEdit(), domain_model=simple_model,
                    output_dir=str(tmp_path), max_turns=5).run("Build an app")
    sidecar = tmp_path / ".besser_tool_inputs.jsonl"
    assert sidecar.exists(), sorted(os.listdir(tmp_path))
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[0]["tool"] == "modify_file"
    assert rows[0]["input"]["old_text"] == long_block, "must be the untruncated bytes"
    assert len(long_block) > 500


# -- the same TARGET refused again, whatever the draft --------------------

SOURCE = "# keep\nasync def action():\n    return False\n"


def _executor(tmp_path):
    from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    return ToolExecutor(workspace=str(tmp_path))


def _miss(ex, new_text):
    """A modify_file that cannot match, so the anchor is the only constant."""
    return ex.execute_typed("modify_file", {
        "path": "app.py", "old_text": "def nowhere():\n    pass\n",
        "new_text": new_text}).payload


def test_alternating_drafts_at_one_target_escalate(tmp_path):
    """Run se7k3zbx alternated two byte-identical drafts at booking.py
    301-400 across t24/26/28/30/32. Keyed on exact text the counter only
    reached 3 on the fifth call; keyed on the target it reaches 4 on the
    fourth."""
    ex = _executor(tmp_path)
    for draft in ("# A\n", "# B\n", "# A\n"):
        _miss(ex, draft)
        assert (ex.last_repeat or ("app.py", 0))[1] < 3, draft

    _miss(ex, "# B\n")

    assert ex.last_repeat == ("app.py", 4)
    assert ex.repeat_rejections("app.py") == 4


def test_three_genuinely_different_drafts_do_not_escalate(tmp_path):
    """Run mbzbzhq9 sent three different drafts to one range at t34/36/40 and
    SUCCEEDED on the third, so the target threshold sits above the exact one."""
    ex = _executor(tmp_path)
    for draft in ("# A\n", "# B\n", "# C\n"):
        _miss(ex, draft)
        assert (ex.last_repeat or ("app.py", 0))[1] < 3, draft


def test_a_successful_edit_clears_the_target_counter(tmp_path):
    ex = _executor(tmp_path)
    for draft in ("# A\n", "# B\n", "# C\n"):
        _miss(ex, draft)
    landed = ex.execute_typed("modify_file", {
        "path": "app.py", "old_text": "    return False",
        "new_text": "    return True"}).payload
    assert landed["status"] == "modified", landed

    _miss(ex, "# D\n")
    _miss(ex, "# E\n")

    assert ex.last_repeat is None, "a landed edit must reset the target count"


def test_a_range_edit_target_counts_across_redrafts(tmp_path):
    """A range edit carries no old_text, so its target is the selected lines."""
    ex = _executor(tmp_path)
    read = ex.execute_typed("read_file", {"path": "app.py"}).payload
    for n in range(4):
        refused = ex.execute_typed("replace_file_lines", {
            "path": "app.py", "read_id": read["read_id"], "start_line": 2,
            "end_line": 3, "new_text": f"async def broken(\n    # {n}\n"}).payload
        assert refused["rejection_kind"] == "syntax_error", refused

    assert ex.last_repeat == ("app.py", 4)


# -- two refusals -> rewrite the whole file -------------------------------


def test_two_refusals_on_one_path_escalate_to_a_whole_file_rewrite(tmp_path):
    """The top of the ladder is now ``read_file`` whole, then ``write_file``.

    Adopted for robustness, not for throughput. The 96-run A/B (48 per arm,
    qwen30b, three cases, arms run concurrently) moved the pass rate not at
    all: 12/48 control vs 16/48 treatment, p=0.501, CI [-9.7, +25.7], and a
    first wave's 3/24 vs 9/24 did not replicate. What it did move is lost
    scaffold code: 27 items across 6 apps in the control (one app lost four
    ORM classes, another eight routes, another a spec-required `ship`
    endpoint) against 0 in the treatment, p=0.027. Those losses were traced
    by hand to SUCCESSFUL ``replace_file_lines`` edits on a drifted view
    overwriting neighbouring classes - damage done by the tool the old
    ladder escalated toward.

    The escalation is only safe because the rewrite it names is itself
    gated: ``_write_file`` refuses a file this run has not read, so the
    "read the WHOLE file first" instruction is backed by an executor check
    and not by the model's goodwill. Both halves are asserted here.
    """
    ex = _executor(tmp_path)

    assert "edit_recovery" not in _miss(ex, "# A\n"), "one refusal is not an escalation"
    recovery = _miss(ex, "# B\n")["edit_recovery"]

    assert recovery["next_tool"] == "write_file"
    assert recovery["path"] == "app.py"
    assert "read_file on the WHOLE file (no offset/limit)" in recovery["instruction"]
    assert "do not summarise, elide, or drop code" in recovery["instruction"]
    assert "No rejected edit was applied" in recovery["instruction"]

    # The tool it names refuses to run before that read happens.
    unread = ex.execute_typed("write_file", {"path": "app.py", "content": "x = 1\n"}).payload
    assert "error" in unread, unread
    assert "you have not read it this run" in unread["error"]

    ex.execute_typed("read_file", {"path": "app.py"})
    rewritten = ex.execute_typed("write_file", {"path": "app.py", "content": "x = 1\n"}).payload
    assert rewritten["status"] == "written", rewritten


def test_successful_edits_never_escalate_to_a_rewrite(tmp_path):
    """The 7cb06829 regression guard, at the behaviour level.

    The per-file rule deleted in 7cb06829 counted edits that WORKED and
    ordered a whole-file rewrite after three, so "three good edits to one
    router became a whole-file rewrite". The rewrite is back, but on the
    counter that replaced it: consecutive refusals, cleared by any landed
    edit. Five good edits in a row must arm nothing, and one landed edit
    between two misses must put the count back to zero - otherwise the new
    tier is the old bug with a new trigger word.
    """
    ex = _executor(tmp_path)
    source = (tmp_path / "app.py")
    source.write_text("".join(f"v{n} = {n}\n" for n in range(5)), encoding="utf-8")

    for n in range(5):
        landed = ex.execute_typed("modify_file", {
            "path": "app.py", "old_text": f"v{n} = {n}",
            "new_text": f"v{n} = {n + 100}"}).payload
        assert landed["status"] == "modified", landed
        assert "edit_recovery" not in landed, landed.get("edit_recovery")

    _miss(ex, "# A\n")
    cleared = ex.execute_typed("modify_file", {
        "path": "app.py", "old_text": "v0 = 100", "new_text": "v0 = 0"}).payload
    assert cleared["status"] == "modified", cleared

    assert "edit_recovery" not in _miss(ex, "# B\n"), (
        "a landed edit must reset the refusal count, or an ordinary "
        "miss-fix-miss rhythm escalates to a rewrite"
    )
