"""A validation collector must never report clean when it did not run.

Every one of these collectors returned ``[]`` on a timeout, a tool that failed
to launch, or an internal error — which the Phase 3 verdict renders as
"0 blockers". That is the exact shape of the pilot failure where backends that
NameError on import shipped as verified: the check had not run, and nothing in
the result said so.

The contract is: a skipped check produces a visible *warning* (we can't prove
the code is broken, only that we didn't look), never silence.
"""
import json
import subprocess

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import (
    LLMOrchestrator, _check_did_not_run, _classify_issue,
)
from besser.spec_driven_agent.validation.issues import is_completion_issue, required_check_unverified


def _model():
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="TestModel", types={user})


class _MockClient:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, system, messages, tools):
        return {"stop_reason": "end_turn", "content": []}


@pytest.fixture
def orch(tmp_path):
    return LLMOrchestrator(
        llm_client=_MockClient(), domain_model=_model(), output_dir=str(tmp_path),
    )


def _said_it_skipped(issues):
    return any("did not run" in i and "SKIPPED" in i for i in issues)


# --------------------------------------------------------------------------- #
# The note itself
# --------------------------------------------------------------------------- #
def test_a_skipped_check_is_a_warning_not_a_blocker(orch, tmp_path, monkeypatch):
    note = _check_did_not_run("ruff", "timed out after 30s")
    assert _classify_issue(note).severity == "warning"
    # It must not look like a passing result to a reader either.
    assert "SKIPPED" in note
    assert not is_completion_issue(_classify_issue(note)), "optional linters do not block completion"
    required = _classify_issue(required_check_unverified("frontend build [.]", "shell tools are disabled"))
    assert required.severity == "warning" and is_completion_issue(required)
    orch.auto_fix_issues = True
    orch._phase2_exited_cleanly = True
    orch._checkpoint_phase = "phase3"
    monkeypatch.setattr(orch, "_collect_validation_issues", lambda: [required])
    monkeypatch.setattr(orch.client, "chat", lambda *a, **kw: pytest.fail("environment gaps must not spend repair turns"))
    orch._run_phase3_validation()
    orch._finish_checkpoint()
    from besser.spec_driven_agent.checkpoint import load_checkpoint
    assert load_checkpoint(str(tmp_path)).phase == "phase3"
    orch._validation_issues = [_classify_issue(note)]
    orch._finish_checkpoint()
    assert load_checkpoint(str(tmp_path)) is None

    loss = {"diagram_id": "gui-1", "diagram_type": "GUIModel", "diagnostic": "processor_failed: ValueError"}
    assembled = LLMOrchestrator(llm_client=orch.client, domain_model=_model(), output_dir=str(tmp_path), assembly_issues=[loss])
    loss["diagnostic"] = "changed after construction"
    findings = [item for item in assembled._collect_validation_issues() if "model assembly" in item.message]
    assert len(findings) == 1 and findings[0].severity == "warning" and is_completion_issue(findings[0])
    assert "GUIModel [gui-1]" in findings[0].message and "processor_failed: ValueError" in findings[0].message
    assembled._validation_issues = findings
    assembled._save_checkpoint_for_turn(1, [], "Create the GUI")
    assert load_checkpoint(str(tmp_path)).validation_issues[0]["message"] == findings[0].message
    assembled._save_recipe("Create the GUI", 0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["model_assembly_issues"][0]["diagnostic"] == "processor_failed: ValueError"


def test_requirement_judgment_is_stable_until_source_changes(orch, tmp_path, monkeypatch):
    from besser.spec_driven_agent import requirements_ledger as ledger

    orch._requirements = [{"id": 1, "text": "An action works", "kind": "action"}]
    calls = []

    def judge(*args, **kwargs):
        assert kwargs.get("original_spec") == orch._instructions
        calls.append(args)
        return [{"id": 1, "text": "An action works", "kind": "action", "status": "missing", "evidence": ""}]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    first = orch._collect_requirement_issues()
    (tmp_path / ".besser_trace.jsonl").write_text("trace noise\n", encoding="utf-8")
    assert orch._collect_requirement_issues() == first
    assert len(calls) == 1
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    orch._collect_requirement_issues()
    assert len(calls) == 2

    # Citation-only mistakes can recover once, without editing a working app
    # or rerolling the already-missing requirement.
    orch._requirements.append({"id": 2, "text": "Value is stored", "kind": "attribute"})
    original = {"id": 1, "text": "An action works", "kind": "action", "status": "missing", "evidence": ""}
    rounds = []

    def citation_judge(requirements, *args, **kwargs):
        assert kwargs.get("original_spec") == orch._instructions
        rounds.append([item["id"] for item in requirements])
        if len(rounds) > 1:
            feedback = kwargs["previous_verdicts"]
            assert [item["id"] for item in feedback] == [2]
            assert feedback[0]["evidence"] == "missing.py:value = 2"
            assert "missing" in feedback[0]["note"]
        evidence = "missing.py:value = 2" if len(rounds) == 1 else "app.py:value = 2"
        answer = {"id": 2, "text": "Value is stored", "kind": "attribute", "status": "implemented", "evidence": evidence}
        return [original, answer] if len(rounds) == 1 else [answer]

    monkeypatch.setattr(ledger, "judge_coverage", citation_judge)
    orch._collect_requirement_issues()
    assert orch._requirement_verdicts[1]["status"] == "unverified"
    orch._collect_requirement_issues()
    orch._collect_requirement_issues()
    assert rounds == [[1, 2], [2]]
    assert [item["status"] for item in orch._requirement_verdicts] == ["missing", "implemented"]

    # Stopping between the initial judgment and citation retry cannot spend a
    # second request, consume the retry, or convert unknown evidence to success.
    (tmp_path / "extra.py").write_text("other = 1\n", encoding="utf-8")
    rounds.clear()
    orch._collect_requirement_issues()
    assert len(rounds) == 1 and orch._requirement_verdicts[1]["status"] == "unverified"
    monkeypatch.setattr(orch, "max_cost_usd", 0)
    assert orch._collect_requirement_issues() and len(rounds) == 1
    monkeypatch.setattr(orch, "max_cost_usd", 10)
    monkeypatch.setattr(orch, "_should_continue", lambda: False)
    assert orch._collect_requirement_issues() and len(rounds) == 1
    monkeypatch.setattr(orch, "_should_continue", None)
    orch._collect_requirement_issues()
    assert rounds == [[1, 2], [2]]


def test_failed_requirement_extraction_stays_unknown_and_retries_once(orch, monkeypatch):
    from besser.spec_driven_agent import requirements_ledger as ledger

    calls = []
    monkeypatch.setattr(ledger, "_is_real_provider", lambda client: True)
    monkeypatch.setattr(ledger, "extract_requirements", lambda *args: calls.append(args))
    monkeypatch.setattr(orch, "_should_continue", lambda: False)
    orch._planner_instructions("Build a booking app")
    assert calls == [] and orch._requirement_extraction_attempts == 0
    monkeypatch.setattr(orch, "_should_continue", None)
    orch._planner_instructions("Build a booking app")
    assert orch._requirements is None
    for _ in range(3):
        issues = orch._collect_requirement_issues()
        assert issues and _classify_issue(issues[0]).severity == "blocker"
    assert len(calls) == 2


def test_conversion_losses_remain_obligations_until_current_code_evidence(orch, tmp_path, monkeypatch):
    from besser.spec_driven_agent import requirements_ledger as ledger

    expression = "context User inv nonnegative: self.missing >= 0"
    diagnostic = {
        "id": "ocl-conversion-test", "name": "nonnegative", "context": "User",
        "expression": expression, "original_text": expression,
        "reason": "Property 'missing' not found in context 'User'",
        "source": {"diagram_title": "Users", "element_id": "ocl-1"},
    }
    orch.domain_model.conversion_issues = [diagnostic]
    monkeypatch.setattr(ledger, "extract_requirements", lambda *_: [])
    planned = orch._planner_instructions("Users must have a nonnegative count.")
    assert expression in planned
    assert orch._requirements == []  # recovery does not pretend extraction found the rule
    prompt = orch._build_system_prompt("Users must have a nonnegative count.")
    assert expression in prompt
    assert "Where the model disagrees with your specification" in prompt
    tasks = orch._deterministic_gap_tasks()
    assert len(tasks) == 1 and expression in tasks[0]
    orch.executor.set_tasks(tasks)

    # Turning off the paid judge cannot silently turn a lost model rule green.
    orch.enable_requirements_ledger = False
    disabled = orch._collect_requirement_issues()
    assert disabled and expression in disabled[0]
    assert all(_classify_issue(item).severity == "warning" and is_completion_issue(_classify_issue(item))
               for item in disabled)
    orch.enable_requirements_ledger = True

    def judge(requirements, digest, client, **kwargs):
        assert len(requirements) == 1
        req = requirements[0]
        return [{**req, "status": "implemented", "note": "",
                 "evidence": 'app.py:raise ValueError("count must be nonnegative")'}]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    source = tmp_path / "app.py"
    source.write_text("count = 0\n", encoding="utf-8")
    assert orch._collect_requirement_issues()  # invented evidence cannot close it
    orch.executor._task_list({"action": "drop", "id": 1, "reason": "already modeled"})
    assert orch._collect_requirement_issues()  # nor can a dropped checklist item

    source.write_text('def validate(count):\n    if count < 0:\n'
                      '        raise ValueError("count must be nonnegative")\n', encoding="utf-8")
    orch.executor.set_tasks(tasks)
    assert orch._collect_requirement_issues() == []
    assert orch._collect_task_issues() == []  # current verified recovery closes the task
    assert orch.executor.task_snapshot()[0]["verification"] == "evidence_checked"
    orch._save_recipe("Users must have a nonnegative count.", elapsed=0)
    recipe = json.loads((tmp_path / ".besser_recipe.json").read_text(encoding="utf-8"))
    assert recipe["model_conversion_issues"] == [diagnostic]
    assert recipe["requirements"][0]["status"] == "implemented"

    source.write_text("count = 0\n", encoding="utf-8")
    assert orch._collect_requirement_issues()  # later regressions invalidate the evidence

    # A postcondition is a promised result, not an input rejection. An actual
    # state assignment must remain eligible evidence for its action contract.
    orch.domain_model.conversion_issues = [{
        **diagnostic, "kind": "postcondition", "method": "activate",
        "expression": "context User::activate() post active: self.active = true",
    }]
    post = orch._requirements_for_validation()[0]
    assert post["kind"] == "action" and "postcondition on User::activate" in post["text"]
    assert "reject violations" not in post["text"]
    source.write_text("def activate(user):\n    user.active = True\n", encoding="utf-8")
    verdict = ledger.verify_evidence([
        {**post, "status": "implemented", "evidence": "app.py:user.active = True"},
    ], str(tmp_path))
    assert verdict[0]["status"] == "implemented"

    # Exhausted runtime must not bypass the completion gate silently, either.
    orch._start_time = 0
    orch.max_runtime_seconds = 1
    orch._run_phase3_validation()
    assert any(item.severity == "blocker" and "validation did not run" in item.message
               for item in orch._validation_issues)


def test_core_rejects_oversized_spec_before_work(orch):
    from besser.spec_driven_agent.specification import MAX_SPECIFICATION_CHARS

    for entry in (orch.run, orch.resume, orch.modify):
        with pytest.raises(ValueError, match="specification was not truncated"):
            entry("x" * (MAX_SPECIFICATION_CHARS + 1))
    assert orch._instructions == ""
    assert not orch.tool_calls_log


# --------------------------------------------------------------------------- #
# ruff
# --------------------------------------------------------------------------- #
def test_ruff_timeout_is_reported(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=30)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert _said_it_skipped(orch._collect_ruff_issues())


def test_ruff_failing_to_launch_is_reported(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    def _boom(*a, **kw):
        raise OSError("Exec format error")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert _said_it_skipped(orch._collect_ruff_issues())


def test_ruff_erroring_out_is_reported(orch, monkeypatch):
    """--exit-zero means findings never set a non-zero status, so a non-zero
    code is ruff itself failing. That used to read as a clean workspace."""
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    class _R:
        returncode = 2
        stdout = ""
        stderr = "ruff failed to read pyproject.toml"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    issues = orch._collect_ruff_issues()
    assert _said_it_skipped(issues)
    assert any("pyproject.toml" in i for i in issues)


def test_a_clean_ruff_run_stays_clean(orch, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/ruff")

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    assert orch._collect_ruff_issues() == []


# --------------------------------------------------------------------------- #
# tsc
# --------------------------------------------------------------------------- #
def _with_tsconfig(tmp_path):
    (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")


def test_tsc_failing_with_no_parseable_errors_is_reported(orch, tmp_path, monkeypatch):
    """A frontend whose toolchain won't even start is not a passing frontend."""
    _with_tsconfig(tmp_path)
    orch.enable_toolchain_validation = True
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc")

    class _R:
        returncode = 1
        stdout = ""
        stderr = "Cannot find module 'typescript'"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())
    issues = orch._collect_tsc_issues()
    assert any("typescript" in i for i in issues)


def test_tsc_timeout_is_reported(orch, tmp_path, monkeypatch):
    _with_tsconfig(tmp_path)
    orch.enable_toolchain_validation = True
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="tsc", timeout=60)

    monkeypatch.setattr(subprocess, "run", _timeout)
    assert _said_it_skipped(orch._collect_tsc_issues())


def test_a_clean_tsc_run_stays_clean(orch, tmp_path, monkeypatch):
    _with_tsconfig(tmp_path)
    orch.enable_toolchain_validation = True
    (tmp_path / "package.json").write_text(json.dumps({"devDependencies": {"typescript": "^5"}}))
    orch.allow_shell_tools = True
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/npm" if n.startswith("npm") else None)
    calls = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(command, **kwargs):
        calls.append(command)
        return _R()

    monkeypatch.setattr(subprocess, "run", run)
    findings = orch._collect_tsc_issues()
    assert len(findings) == 1 and findings[0].startswith("verification setup:")
    assert _classify_issue(findings[0]).severity == "blocker" and calls == []
    monkeypatch.setattr("shutil.which", lambda n: None)
    assert _classify_issue(orch._collect_tsc_issues()[0]).severity == "warning", "missing system npm must not trigger futile setup"

    import os
    local_tsc = tmp_path / "node_modules" / ".bin" / ("tsc.cmd" if os.name == "nt" else "tsc")
    local_tsc.parent.mkdir(parents=True)
    local_tsc.write_text("mock compiler, never executed")
    orch.allow_shell_tools = False
    assert _classify_issue(orch._collect_tsc_issues()[0]).severity == "warning" and calls == []
    orch.allow_shell_tools = True
    assert orch._collect_tsc_issues() == []
    assert calls[-1][0] == str(local_tsc), "authorized project compiler works without a global tsc"
    monkeypatch.setattr("shutil.which", lambda n: "/global/tsc")
    assert orch._collect_tsc_issues() == []
    assert calls[-1][0] == str(local_tsc), "prefer project-pinned compiler over host version"


def test_required_frontend_build_respects_permissions_and_current_source(orch, tmp_path, monkeypatch):
    from besser.spec_driven_agent.validation.frontend_build import collect_frontend_build_issues

    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"build": "vite build"}}))
    (tmp_path / "index.html").write_text("<div id='root'></div>")
    source = tmp_path / "app.js"
    source.write_text("export const value = 1;")
    # An unrelated library package is not an application requiring a build.
    library = tmp_path / "library"
    library.mkdir()
    (library / "package.json").write_text(json.dumps({"scripts": {"build": "echo library"}}))
    calls, cache = [], {}
    mode = {"outcome": "pass"}

    def build(command, **kwargs):
        calls.append((command, kwargs))
        assert command[-2:] == ["run", "build"] and "OPENAI_API_KEY" not in kwargs["env"]
        for path in ("dist/assets/app.js", "build/manifest.json", ".next/build-manifest.json", "tsconfig.tsbuildinfo"):
            output = tmp_path / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("generated build artifact")
        if mode["outcome"] == "timeout":
            raise subprocess.TimeoutExpired(command, timeout=120)
        if mode["outcome"] == "mutate":
            source.write_text("export const value = 3;")
        return subprocess.CompletedProcess(command, 1 if mode["outcome"] == "fail" else 0, "build result", "")

    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    monkeypatch.setattr(subprocess, "run", build)
    monkeypatch.setenv("OPENAI_API_KEY", "never-pass-this-to-build")
    options = dict(source_revision=orch._workspace_revision, successful_builds=cache, can_run=lambda: True)
    for enabled, allowed in ((False, True), (True, False), (True, True)):
        findings = collect_frontend_build_issues(str(tmp_path), enabled=enabled, allow_shell=allowed, **options)
        assert len(findings) == 1 and is_completion_issue(_classify_issue(findings[0]))
        assert _classify_issue(findings[0]).severity == ("blocker" if enabled and allowed else "warning")
        if enabled and allowed:
            assert findings[0].startswith("verification setup:") and "install_dependencies" in findings[0]
    assert calls == [], "missing permission/dependencies must never launch npm or install"
    monkeypatch.setattr("shutil.which", lambda name: None)
    findings = collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options)
    assert _classify_issue(findings[0]).severity == "warning" and "npm is unavailable" in findings[0]
    monkeypatch.setattr("shutil.which", lambda name: "/tools/npm")
    (tmp_path / "node_modules").mkdir()
    assert collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options) == []
    assert len(calls) == 1
    assert collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options) == []
    assert len(calls) == 1, "only current source may reuse a successful check"
    source.write_text("export const value = 2;")
    mode["outcome"] = "mutate"
    findings = collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options)
    assert any("source changed during build" in item for item in findings)
    mode["outcome"] = "fail"
    findings = collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options)
    assert _classify_issue(findings[0]).severity == "blocker"
    mode["outcome"] = "timeout"
    findings = collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options)
    assert _classify_issue(findings[0]).severity == "warning" and "timed out" in findings[0]
    (tmp_path / "package.json").write_text("invalid JSON")
    findings = collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options)
    assert _classify_issue(findings[0]).severity == "blocker" and "package.json" in findings[0]

    # Model an escaped junction without requiring Windows symlink privileges.
    import os
    realpath = os.path.realpath
    escape = str(tmp_path / "escape")
    def walk(root):
        children = ["escape", "node_modules"]
        yield root, children, []
        assert children == [], "prune escaped directories before descent, not after reading manifests"
    monkeypatch.setattr(os, "walk", walk)
    monkeypatch.setattr(os.path, "realpath", lambda path: str(tmp_path.parent) if str(path) == escape else realpath(path))
    assert collect_frontend_build_issues(str(tmp_path), enabled=True, allow_shell=True, **options) == []


# --------------------------------------------------------------------------- #
# data contract
# --------------------------------------------------------------------------- #
def test_a_failed_contract_build_is_reported(orch, monkeypatch):
    import besser.spec_driven_agent.contract_checks as cc

    def _boom(_model):
        raise RuntimeError("contract build exploded")

    monkeypatch.setattr(cc, "build_data_contract", _boom)
    issues = orch._collect_data_contract_issues()
    assert _said_it_skipped(issues)
    assert any("exploded" in i for i in issues)
