"""A run's full record can be kept server-side, apart from the size-bounded recipe the user downloads.

The recipe ships inside the app archive, so it is cut to a size budget: a long run
kept 5 of its 109 tool calls there, with their inputs shortened, and the trace
that held the rest lived in the run's workspace, which is deleted hours after the
run. When the operator sets ``BESSER_INCIDENT_LOG_DIR``, the backend appends every
trace record, redacted, to ``<incident log dir>/runs/<run id>.jsonl`` as it happens.
"""
import asyncio
import io
import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import dumps_for_model
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.state.tracing import EVENT_PHASE_ENTER, EVENT_TOOL_CALL, TraceWriter
from besser.spec_driven_agent.validation.issues import ValidationIssue
from besser.utilities.secret_redaction import redact_record_text
from besser.utilities.web_modeling_editor.backend.services import cleanup
from besser.utilities.web_modeling_editor.backend.services.spec_driven import incidents, run_evidence, run_record
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings
from tests.utilities.web_modeling_editor.backend._runtime import current_runtime
from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import (
    _FakeClient,
    _FakeOrchestrator,
    _build_request,
    _collect_frames,
    _parse_frame,
)

RUN = "0123456789abcdef0123456789abcdef"
# Keys with no known provider prefix: only the literal match finds them.
OWN_KEY = "Zq8mV3xK9pLr2Tw7Yb4Nc6Hd1Fg5Js0A"
SERVER_TOKEN = "Hq2Wn8Zr5Tx1Lc7Vb3Mk9Pd4Fs6Gj0Ya"
PROVIDER_KEY = "sk-proj-" + "A" * 40
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbm4ifQ.c2lnbmF0dXJlLXZhbHVl"
HEX = "8f3a9c1e5b7d2f4a6c8e0b1d3f5a7c9e"
CALLS = 150


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")


@pytest.fixture
def incident_dir(tmp_path, monkeypatch):
    path = tmp_path / "incidents"
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", str(path))
    return path


@pytest.fixture
def default_dir(tmp_path, monkeypatch):
    """Where the incident log goes when the variable is unset, kept under ``tmp_path``."""
    path = tmp_path / "default-incidents"
    monkeypatch.setattr(incidents, "_DEFAULT_INCIDENT_LOG_DIR", str(path))
    return path


def _orchestrator(output_dir, evidence=None) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
        output_dir=str(output_dir), enable_checkpointing=False, enable_requirements_ledger=False,
        enable_toolchain_validation=False, run_id=RUN, run_evidence=evidence)


def _records(incident_dir, run_id=RUN) -> list[dict]:
    path = incident_dir / "runs" / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tool_calls(records) -> list[dict]:
    return [r for r in records if r["event"] == EVENT_TOOL_CALL]


def _stored(incident_dir, tool, tool_input, result, secrets=(OWN_KEY,)) -> str:
    """What the record keeps of one call, as the orchestrator passes it."""
    record = run_evidence.for_run(RUN, secrets=secrets)
    record({"ts": 0, "run_id": RUN, "event": EVENT_TOOL_CALL,
            "payload": {"tool": tool, "input": tool_input, "result": result}})
    return (incident_dir / "runs" / f"{RUN}.jsonl").read_text(encoding="utf-8")


def _cut_recipe(workspace, evidence, monkeypatch) -> dict:
    monkeypatch.setitem(ToolExecutor._handlers, "modify_file",
                        lambda self, args: {"success": True, "path": args["path"], "log": "ok\n" * 3000})
    orchestrator = _orchestrator(workspace, evidence)
    orchestrator._trace.write(EVENT_PHASE_ENTER, phase="phase2")
    for turn in range(CALLS):
        block = SimpleNamespace(id=f"t{turn}", name="modify_file", input={
            "path": f"backend/routes_{turn % 5}.py", "old_text": f"old {turn} " + "o" * 900,
            "new_text": f"new {turn} " + "n" * 900})
        orchestrator._execute_single_tool(block, turn=turn)
    # Findings the recipe never cuts, so cutting the inputs is not enough and calls are dropped.
    orchestrator._validation_issues = [ValidationIssue("warning", f"{n} " + "w" * 1000) for n in range(115)]
    orchestrator._save_recipe("Track orders", 30.0)
    return json.loads((workspace / ".besser_recipe.json").read_text(encoding="utf-8"))


def test_a_cut_recipe_says_what_it_kept_and_the_server_record_keeps_every_call(tmp_path, incident_dir,
                                                                               monkeypatch):
    workspace = tmp_path / "besser_llm_run"
    workspace.mkdir()

    recipe = _cut_recipe(workspace, run_evidence.for_run(RUN, secrets=(OWN_KEY,)), monkeypatch)

    kept = len(recipe["tool_calls"])
    assert kept == 5 and recipe["tool_calls_count"] == CALLS
    assert recipe["recipe_shortened"]["note"].endswith(
        f"kept {kept} of {CALLS} tool calls; full record kept server-side under run {RUN}")
    calls = _tool_calls(_records(incident_dir))
    assert [c["payload"]["turn"] for c in calls] == list(range(1, CALLS + 1))
    first = calls[0]
    assert first["phase"] == "phase2" and first["run_id"] == RUN
    assert first["payload"]["input"]["new_text"] == "new 0 " + "n" * 900, "the input is whole"
    assert first["payload"]["success"] is True and first["payload"]["seconds"] >= 0
    result = first["payload"]["result"]
    assert len(result) > 9000 and result.startswith('{"success": true')
    assert "<<cut" not in result, "a result under the cap is whole"


def test_a_cut_recipe_without_a_server_record_names_none(tmp_path, monkeypatch):
    recipe = _cut_recipe(tmp_path, None, monkeypatch)

    note = recipe["recipe_shortened"]["note"]
    assert note.endswith(f"kept 5 of {CALLS} tool calls")
    assert "server" not in note and "record" not in note


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "blank", "spaces"])
def test_no_record_is_kept_unless_the_incident_dir_is_set(tmp_path, default_dir, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BESSER_INCIDENT_LOG_DIR", raising=False)
    else:
        monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", value)
    monkeypatch.chdir(tmp_path)

    assert run_evidence.for_run(RUN) is None
    assert incidents.incident_log_dir() == str(default_dir), "blank is unset"
    run_evidence.prune_run_evidence()
    assert not (tmp_path / "runs").exists() and not (default_dir / "runs").exists()


@pytest.mark.parametrize("days", ["0", "-3"])
def test_a_retention_of_zero_or_less_keeps_no_record(incident_dir, monkeypatch, days):
    monkeypatch.setenv("BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS", days)

    assert BackendSettings().llm_run_evidence_retention_days == 0
    assert run_evidence.for_run(RUN) is None


def test_an_ordinary_run_keeps_no_record_by_default(tmp_path, default_dir, monkeypatch):
    monkeypatch.delenv("BESSER_INCIDENT_LOG_DIR", raising=False)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _CrashesMidRun)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    runner = SmartGenerationRunner(_build_request(api_key=OWN_KEY), runtime=current_runtime())
    asyncio.run(_collect_frames(runner))

    assert runner._t_orchestrator.evidence is None
    assert not (default_dir / "runs").exists()


def test_a_long_argument_and_result_are_cut_to_their_caps(incident_dir):
    content = "a" * (run_evidence.VALUE_CHARS * 2)
    output = "HEAD" + "x" * (run_evidence.RESULT_CHARS * 4) + "TAIL"

    [call] = _tool_calls([json.loads(_stored(incident_dir, "write_file", {"content": content}, output))])

    written, result = call["payload"]["input"]["content"], call["payload"]["result"]
    assert len(written) < run_evidence.VALUE_CHARS + 100 and f"of {len(content)} chars" in written
    assert result.startswith("HEAD") and result.endswith("TAIL")
    assert len(result) < run_evidence.RESULT_CHARS + 100


@pytest.mark.parametrize("where", ["head", "tail"])
def test_a_secret_straddling_a_cut_never_survives(incident_dir, where):
    half = run_evidence.VALUE_CHARS // 2
    if where == "head":
        content = "x" * (half - 10) + OWN_KEY + "y" * (run_evidence.VALUE_CHARS * 2)
    else:
        content = "y" * (run_evidence.VALUE_CHARS * 2) + OWN_KEY + "x" * (half - 10)
    result = "x" * (run_evidence.RESULT_CHARS // 2 - 10) + OWN_KEY + "y" * 20000

    stored = _stored(incident_dir, "write_file", {"path": "a.py", "content": content}, result)

    # Ten characters of the key are on the kept side of the cut.
    for fragment in (OWN_KEY[:10], OWN_KEY[-10:]):
        assert fragment not in stored


def test_many_secrets_before_a_cut_do_not_pull_an_unredacted_one_into_it(incident_dir):
    """Each redaction shortens the redacted window past the cut, pulling its far end into the kept part."""
    half = run_evidence.VALUE_CHARS // 2
    keys = (OWN_KEY + " ") * (half // (len(OWN_KEY) + 1))
    # A key across the far end of the window: only its first ten characters are in it.
    window_end = half + 1024
    content = keys + "z" * (window_end - 10 - len(keys)) + OWN_KEY + "y" * (run_evidence.VALUE_CHARS * 2)

    stored = _stored(incident_dir, "write_file", {"path": "a.py", "content": content}, "{}")

    assert OWN_KEY[:10] not in stored


def _long_literal_across_a_cut(value, at, earlier_key_at=None, total=100_000):
    """A file with ``value`` as a quoted credential literal at ``at``, and maybe a private key."""
    text = list("x" * total)

    def put(position, chunk):
        text[position:position + len(chunk)] = chunk
    if earlier_key_at is not None:
        put(earlier_key_at, "-----BEGIN PRIVATE KEY-----\n" + "\n".join("k" * 64 for _ in range(6))
            + "\n-----END PRIVATE KEY-----")
    put(at, f'\nconst sessionToken = "{value}";\n')
    return "".join(text)[:total]


@pytest.mark.parametrize("length, at, earlier_key_at", [
    (1000, run_evidence.VALUE_CHARS // 2 + 1024 - 800, 1000),
    (1000, 100_000 - run_evidence.VALUE_CHARS // 2 - 1024 - 200, 100_000 - 2000),
    (2000, run_evidence.VALUE_CHARS // 2 - 100, None),
], ids=["head-window-after-a-key", "tail-window-before-a-key", "across-the-cut"])
def test_a_long_credential_across_a_cut_or_window_never_survives(length, at, earlier_key_at):
    value = "".join(chr(ord("A") + (i * 7) % 26) + str(i % 10) for i in range(length // 2))
    text = _long_literal_across_a_cut(value, at, earlier_key_at)

    kept = run_evidence._redacted_cut(text, run_evidence.VALUE_CHARS, redact_record_text)

    assert [value[i:i + 24] for i in range(0, len(value) - 24, 24) if value[i:i + 24] in kept] == []


ENV_BODY = f"SECRET_KEY={HEX}\nJWT_SECRET=supersecretjwtvalue123\n"
CURL = ("curl -v -H 'X-API-Key: k9f8e7d6c5b4a3f2e1d0' -H 'Cookie: session=s3ss10nV4lu3QwErTy9876' "
        "https://api.example.com/x")
CURL_OUT = ("> X-API-Key: k9f8e7d6c5b4a3f2e1d0\r\n> Cookie: session=s3ss10nV4lu3QwErTy9876\r\n"
            "< Set-Cookie: sid=zyxwvu9876543210zyxw; HttpOnly\r\n")


@pytest.mark.parametrize("tool, tool_input, result, secrets", [
    ("read_file", {"path": ".env"}, dumps_for_model({"success": True, "content": ENV_BODY}),
     [HEX, "supersecretjwtvalue123"]),
    ("write_file", {"path": "src/config.js", "content": (
        'const JWT_SECRET = "s3cr3t-value-from-user-9Xk2";\nexport const apiKey = "wk_4f8a9b2c7d1e6f3a";\n')},
     dumps_for_model({"success": True}), ["s3cr3t-value-from-user-9Xk2", "wk_4f8a9b2c7d1e6f3a"]),
    ("run_command", {"command": CURL}, dumps_for_model({"stdout": CURL_OUT, "exit_code": 0}),
     ["k9f8e7d6c5b4a3f2e1d0", "s3ss10nV4lu3QwErTy9876", "zyxwvu9876543210zyxw"]),
    ("run_command", {"command": "pip config list"}, dumps_for_model({"stdout": (
        "global.index-url='https://deploy:Pa55w0rdHere99@pypi.corp.example.com/simple'\n"), "exit_code": 0}),
     ["Pa55w0rdHere99"]),
    ("run_command", {"command": "printenv"}, dumps_for_model({"stdout": (
        f"MINIMAX_API_KEY={HEX}\nGITHUB_TOKEN=ghp_{'a' * 36}\nPATH=/usr/bin\n"), "exit_code": 0}),
     [HEX, "ghp_" + "a" * 36]),
], ids=["env-file-read", "js-literals", "curl-headers", "url-credentials", "printenv"])
def test_the_record_redacts_what_tools_read_write_and_print(incident_dir, tool, tool_input, result, secrets):
    stored = _stored(incident_dir, tool, tool_input, result)

    assert [secret for secret in secrets if secret in stored] == []
    assert "[REDACTED]" in stored


def test_the_record_redacts_secrets_in_tool_arguments_and_results(incident_dir, monkeypatch):
    monkeypatch.setitem(ToolExecutor._handlers, "run_command", lambda self, args: {
        "stdout": f'{{"access_token": "{JWT}"}}\nOPENAI={PROVIDER_KEY}\nkey {OWN_KEY}\n', "exit_code": 0})
    orchestrator = _orchestrator(incident_dir.parent, run_evidence.for_run(RUN, secrets=(OWN_KEY,)))
    orchestrator.executor.allow_shell = True
    command = f"curl -H 'Authorization: Bearer {JWT}' -d '{OWN_KEY}' http://localhost/x"

    orchestrator._execute_single_tool(SimpleNamespace(id="t1", name="run_command", input={"command": command}),
                                      turn=0)

    [call] = _tool_calls(_records(incident_dir))
    stored = json.dumps(call)
    for secret in (OWN_KEY, PROVIDER_KEY, JWT):
        assert secret not in stored
    assert "[REDACTED]" in call["payload"]["input"]["command"]
    assert "[REDACTED]" in call["payload"]["result"]


def test_a_key_partly_shaped_like_a_token_is_redacted_whole(incident_dir, monkeypatch):
    key = "AKIA" + "ABCDEFGHIJKLMNOP" + "tailpartofthekey123"
    monkeypatch.setitem(ToolExecutor._handlers, "write_file", lambda self, args: {"success": True})
    orchestrator = _orchestrator(incident_dir.parent, run_evidence.for_run(RUN, secrets=(key,)))

    orchestrator._execute_single_tool(
        SimpleNamespace(id="t1", name="write_file", input={"path": "a.txt", "content": f"key={key}"}), turn=0)

    assert "tailpartofthekey123" not in json.dumps(_records(incident_dir))


def test_a_failed_write_is_logged_once_per_run(incident_dir, caplog):
    incident_dir.mkdir()
    (incident_dir / "runs").write_text("not a folder", encoding="utf-8")
    record = run_evidence.for_run(RUN)

    with caplog.at_level(logging.DEBUG, logger=run_evidence.__name__):
        for _ in range(3):
            record({"ts": 0, "run_id": RUN, "event": EVENT_TOOL_CALL, "payload": {}})

    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1


def test_without_the_backend_a_run_writes_no_server_record(tmp_path, incident_dir, monkeypatch):
    monkeypatch.setitem(ToolExecutor._handlers, "modify_file", lambda self, args: {"success": True})
    orchestrator = _orchestrator(tmp_path / "out")

    orchestrator._execute_single_tool(
        SimpleNamespace(id="t1", name="modify_file", input={"path": "a.py", "old_text": "a", "new_text": "b"}),
        turn=0)

    assert not incident_dir.exists()
    assert (tmp_path / "out" / ".besser_trace.jsonl").is_file()


class _CrashesMidRun(_FakeOrchestrator):
    """Records two tool calls through the trace the runner wired, then fails."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evidence = kwargs["run_evidence"]
        self.trace = TraceWriter(self.output_dir, run_id=kwargs["run_id"], mirror=self.evidence)

    def run(self, instructions: str) -> str:
        self.on_text(f"model prose quoting {SERVER_TOKEN}")
        self.trace.write(EVENT_PHASE_ENTER, phase="phase2")
        for turn in (1, 2):
            self.trace.write(EVENT_TOOL_CALL, uncut={"input": {"command": f"echo {OWN_KEY}"},
                                                     "result": f"printed {SERVER_TOKEN}"},
                             turn=turn, tool="run_command", success=True)
        logging.getLogger("besser.spec_driven_agent.pipeline.orchestrator").warning("Phase 2: about to fail")
        raise RuntimeError("worker crashed")


def _run_id_handler() -> logging.Handler:
    [handler] = [h for h in logging.getLogger("besser").handlers if getattr(h, "is_run_id_log", False)]
    return handler


@pytest.fixture
def run_id_log():
    """The backend's handler on the ``besser`` logger, writing to a buffer; removed afterwards."""
    besser_logger = logging.getLogger("besser")
    before = list(besser_logger.handlers)
    run_record.enable_run_log()
    _run_id_handler().setStream(written := io.StringIO())
    yield written
    besser_logger.handlers[:] = before


def test_a_run_that_crashes_keeps_the_calls_it_made_and_logs_its_id(incident_dir, monkeypatch, run_id_log):
    monkeypatch.setenv("BESSER_FREE_LLM_TOKEN", SERVER_TOKEN)
    monkeypatch.setattr(runner_module, "LLMOrchestrator", _CrashesMidRun)
    monkeypatch.setattr(runner_module, "create_llm_client", lambda **_: _FakeClient())

    runner = SmartGenerationRunner(_build_request(api_key=OWN_KEY), runtime=current_runtime())
    frames = asyncio.run(_collect_frames(runner))

    calls = _tool_calls(_records(incident_dir, runner.run_id))
    assert [c["payload"]["turn"] for c in calls] == [1, 2]
    stored = json.dumps(calls)
    assert OWN_KEY not in stored and SERVER_TOKEN not in stored
    assert "[REDACTED]" in calls[0]["payload"]["input"]["command"]
    assert not [frame for frame in map(_parse_frame, frames) if SERVER_TOKEN in json.dumps(frame)]
    assert f"WARNING:     run_id={runner.run_id} Phase 2: about to fail" in run_id_log.getvalue().splitlines()


def test_tool_calls_run_in_parallel_log_with_their_run_id(tmp_path, monkeypatch, run_id_log):
    def noisy_read(self, args):
        logging.getLogger("besser.spec_driven_agent.agent.tool_executor").warning("read %s", args["path"])
        return {"content": ""}

    monkeypatch.setitem(ToolExecutor._handlers, "read_file", noisy_read)
    orchestrator = _orchestrator(tmp_path)
    blocks = [SimpleNamespace(id=f"t{n}", name="read_file", input={"path": f"f{n}.py"}) for n in range(3)]

    run_record.with_run_id(RUN, orchestrator._execute_tool_blocks)(blocks, 0)
    logging.getLogger("besser.spec_driven_agent").warning("outside any run")

    lines = run_id_log.getvalue().splitlines()
    assert sorted(lines[:3]) == [f"WARNING:     run_id={RUN} read f{n}.py" for n in range(3)]
    assert lines[3] == "WARNING:     outside any run"


def test_the_hourly_cleanup_removes_a_record_past_its_retention(tmp_path, incident_dir, monkeypatch):
    monkeypatch.setenv("BESSER_LLM_RUN_EVIDENCE_RETENTION_DAYS", "30")
    runs = incident_dir / "runs"
    runs.mkdir(parents=True)
    old, recent = runs / "old.jsonl", runs / "recent.jsonl"
    for path in (old, recent):
        path.write_text("{}\n", encoding="utf-8")
    month_ago = time.time() - 31 * 86400
    os.utime(old, (month_ago, month_ago))
    # The cleanup's own folders, never the machine's.
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(cleanup.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    settings = get_settings().model_copy(update={"llm_run_workspace_root": None})
    monkeypatch.setattr(cleanup, "get_settings", lambda: settings)

    cleanup.cleanup_old_temp_files()

    assert not old.exists() and recent.exists()
