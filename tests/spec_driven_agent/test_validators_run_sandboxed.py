"""Phase 3 checks that execute generated code run inside the run sandbox.

The import smoke check (on by default), the constructibility and API probes,
``tsc`` (a project-local compiler is package code), ``cargo check`` (build.rs,
proc-macros) and ``npm run build`` all executed model-authored code with a
plain ``subprocess.run``: outside bubblewrap, so sibling runs, ``/proc/1`` and
the network were in reach even on the hosted worker where ``run_command`` is
confined. They now go through ``run_confined``: sandboxed with no network,
skipped (never run unconfined) where the sandbox is mandatory but unavailable,
and run with a warning on a platform that has no namespaces.

The same review found the sandbox itself sharing writable ``/usr/local`` and
``/root`` between runs (one run's ``pip install`` lands in every later run and
in the worker) and exposing every user's telemetry under ``/app/telemetry``.

bwrap never exists on the Windows test host, so the Linux side is asserted on
the command it would run and on the mount plan, with the filesystem faked.
"""

import os
import subprocess
import sys

import pytest

from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution.process import run_bounded
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SandboxUnavailable,
    run_confined,
)
from besser.spec_driven_agent.validation import toolchain as tc
from besser.spec_driven_agent.validation.issues import _classify_issue


class _Ok:
    returncode = 0
    stdout = ""
    stderr = ""


@pytest.fixture
def no_plain_subprocess(monkeypatch):
    """Executing generated code outside the sandbox helper fails the test."""
    def _refuse(*args, **_kwargs):
        pytest.fail(f"ran outside the sandbox: {args[0] if args else ''}")
    monkeypatch.setattr(subprocess, "run", _refuse)


@pytest.fixture
def confined(monkeypatch, no_plain_subprocess):
    calls = []

    def _record(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Ok()

    monkeypatch.setattr(sandbox_mod, "run_confined", _record)
    return calls


@pytest.fixture
def sandbox_gone(monkeypatch, no_plain_subprocess):
    """Linux worker whose sandbox cannot start: nothing may execute at all."""
    def _unavailable(*_a, **_k):
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

    monkeypatch.setattr(sandbox_mod, "sandboxed_command", _unavailable)
    monkeypatch.setattr(sandbox_mod, "run_bounded",
                        lambda *a, **k: pytest.fail("ran without a sandbox"))


def _orm_backend(root):
    backend = root / "backend"
    backend.mkdir()
    (backend / "sql_alchemy.py").write_text("x = 1\n", encoding="utf-8")
    (backend / "main_api.py").write_text("app = None\n", encoding="utf-8")
    return backend


def _tsc_project(root):
    (root / "tsconfig.json").write_text("{}", encoding="utf-8")
    (root / "node_modules").mkdir()


def _crate(root):
    (root / "Cargo.toml").write_text('[package]\nname = "x"\n', encoding="utf-8")


def _frontend(root):
    import json
    (root / "package.json").write_text(json.dumps({"scripts": {"build": "vite build"}}))
    (root / "index.html").write_text("<div id='root'></div>")
    (root / "node_modules").mkdir()


def _frontend_issues(root):
    from besser.spec_driven_agent.validation.frontend_build import collect_frontend_build_issues
    return collect_frontend_build_issues(
        str(root), enabled=True, allow_shell=True, source_revision=lambda: "r",
        successful_builds={}, can_run=lambda: True)


# --------------------------------------------------------------------------- #
# Every check that executes generated code goes through the sandbox helper
# --------------------------------------------------------------------------- #
def test_the_import_smoke_check_runs_confined(tmp_path, confined):
    from besser.spec_driven_agent.validation.python_imports import _import_smoke_issues

    backend = _orm_backend(tmp_path)
    assert _import_smoke_issues(str(tmp_path)) == []
    [(argv, kwargs)] = confined
    assert argv[0] == sys.executable and "configure_mappers" in argv[-1]
    assert kwargs["workspace"] == str(tmp_path) and kwargs["cwd"] == str(backend)


def test_the_constructibility_probe_runs_confined_on_its_scratch_copy(tmp_path, confined):
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    _orm_backend(tmp_path)
    collect_constructibility_report(str(tmp_path))
    [(_argv, kwargs)] = confined
    assert kwargs["workspace"] == str(tmp_path)
    # Only the scratch copy is writable; the run workspace stays out of reach.
    [scratch] = kwargs["writable"]
    assert kwargs["cwd"].startswith(scratch) and not scratch.startswith(str(tmp_path))


def test_the_api_probe_runs_confined_and_still_gets_its_scenario(tmp_path, confined):
    from besser.spec_driven_agent.validation import api_probe

    _orm_backend(tmp_path)
    api_probe.probe_api_scenario(str(tmp_path), [{"method": "GET", "path": "/room/"}])
    [(argv, kwargs)] = confined
    assert argv[-1] == "--worker"
    assert '"/room/"' in kwargs["input"]
    [scratch] = kwargs["writable"]
    assert kwargs["cwd"].startswith(scratch)


def test_tsc_runs_confined(tmp_path, monkeypatch, confined):
    _tsc_project(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc" if n == "tsc" else None)
    assert tc._collect_tsc_issues(str(tmp_path), False, True, set()) == []
    [(argv, kwargs)] = confined
    assert argv[:2] == ["/usr/bin/tsc", "--noEmit"]
    assert kwargs["workspace"] == str(tmp_path)


def test_cargo_check_runs_confined_with_a_per_run_target_dir(tmp_path, monkeypatch, confined):
    _crate(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/cargo" if n == "cargo" else None)
    assert tc._collect_cargo_issues(str(tmp_path)) == []
    [(argv, kwargs)] = confined
    assert argv[:2] == ["/usr/bin/cargo", "check"]
    # Not a host-wide cache that one run's build outputs could poison.
    assert kwargs["env"]["CARGO_TARGET_DIR"].startswith(sandbox_mod.sandbox_home(str(tmp_path)))


def test_the_frontend_build_runs_confined(tmp_path, monkeypatch, confined):
    _frontend(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/npm")
    assert _frontend_issues(tmp_path) == []
    [(argv, kwargs)] = confined
    assert argv == ["/usr/bin/npm", "run", "build"]
    assert kwargs["workspace"] == os.path.realpath(str(tmp_path))


def test_kotlinc_uses_the_tree_killing_runner(tmp_path, monkeypatch, no_plain_subprocess):
    """subprocess.run's timeout hangs on Windows when a grandchild holds a pipe."""
    src = tmp_path / "svc" / "src" / "main" / "kotlin"
    src.mkdir(parents=True)
    (tmp_path / "svc" / "build.gradle.kts").write_text("plugins { }\n")
    (src / "App.kt").write_text("fun main() {}\n")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/kotlinc" if n == "kotlinc" else None)
    calls = []
    monkeypatch.setattr(tc, "run_bounded", lambda argv, **k: calls.append(argv) or _Ok())

    assert tc._collect_kotlinc_issues(str(tmp_path)) == []
    assert calls and calls[0][0] == "/usr/bin/kotlinc"


# --------------------------------------------------------------------------- #
# Sandbox mandatory but unavailable: the check is skipped, never run unconfined
# --------------------------------------------------------------------------- #
def _skipped(issues):
    assert len(issues) == 1, issues
    assert "sandbox is unavailable" in issues[0] and "SKIPPED" in issues[0], issues
    # "We did not look" must never read as a defect in the generated code.
    assert _classify_issue(issues[0]).severity == "warning", issues
    return issues[0]


def test_the_import_smoke_check_is_skipped_without_a_sandbox(tmp_path, sandbox_gone):
    from besser.spec_driven_agent.validation.python_imports import _import_smoke_issues

    _orm_backend(tmp_path)
    _skipped(_import_smoke_issues(str(tmp_path)))


def test_the_constructibility_probe_is_skipped_without_a_sandbox(tmp_path, sandbox_gone):
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    _orm_backend(tmp_path)
    report = collect_constructibility_report(str(tmp_path))
    _skipped(report["issues"])


def test_the_api_probe_is_refused_without_a_sandbox(tmp_path, sandbox_gone):
    from besser.spec_driven_agent.validation import api_probe

    _orm_backend(tmp_path)
    report = api_probe.probe_api_scenario(str(tmp_path), [{"method": "GET", "path": "/"}])
    assert report["status"] == "error" and "sandbox is unavailable" in report["error"]


def test_tsc_is_unverified_without_a_sandbox(tmp_path, monkeypatch, sandbox_gone):
    _tsc_project(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/tsc" if n == "tsc" else None)
    issues = tc._collect_tsc_issues(str(tmp_path), False, True, set())
    assert any("sandbox is unavailable" in i for i in issues), issues


def test_cargo_is_skipped_without_a_sandbox(tmp_path, monkeypatch, sandbox_gone):
    _crate(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/cargo" if n == "cargo" else None)
    _skipped(tc._collect_cargo_issues(str(tmp_path)))


def test_the_frontend_build_is_unverified_without_a_sandbox(tmp_path, monkeypatch, sandbox_gone):
    _frontend(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/npm")
    issues = _frontend_issues(tmp_path)
    assert len(issues) == 1 and "sandbox is unavailable" in issues[0], issues


def test_an_unfetched_crate_dependency_is_not_reported_as_a_compile_error(tmp_path, monkeypatch):
    """No network in the sandbox: cargo cannot download, the code is not wrong."""
    _crate(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/cargo" if n == "cargo" else None)

    class _Offline:
        returncode = 101
        stdout = ""
        stderr = ("error: failed to get `serde` as a dependency of package `x v0.1.0`\n"
                  "Caused by:\n  Couldn't resolve host name (Could not resolve host: index.crates.io)\n")

    monkeypatch.setattr(sandbox_mod, "run_confined", lambda *a, **k: _Offline())
    [issue] = tc._collect_cargo_issues(str(tmp_path))
    assert "SKIPPED" in issue and _classify_issue(issue).severity == "warning"


def test_a_real_cargo_compile_error_is_still_reported(tmp_path, monkeypatch):
    _crate(tmp_path)
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/cargo" if n == "cargo" else None)

    class _Broken:
        returncode = 101
        stdout = ""
        stderr = "src/main.rs:3:5: error[E0308]: mismatched types\n"

    monkeypatch.setattr(sandbox_mod, "run_confined", lambda *a, **k: _Broken())
    issues = tc._collect_cargo_issues(str(tmp_path))
    assert issues == ["cargo [.]: src/main.rs:3:5: error[E0308]: mismatched types"]


# --------------------------------------------------------------------------- #
# run_confined itself
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_linux_bwrap(monkeypatch):
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: True)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _n: "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox_mod, "_selftest", lambda _b: (True, ""))
    monkeypatch.setattr(sandbox_mod, "_mount_args", lambda ws, writable=None: ["<mounts>"])
    ran = []

    def _run(argv, **kwargs):
        ran.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "out", "")

    monkeypatch.setattr(sandbox_mod, "run_bounded", _run)
    return ran


def test_run_confined_unshares_the_network_and_runs_the_argv_without_a_shell(
    tmp_path, fake_linux_bwrap,
):
    argv = [sys.executable, "-c", "import x; print('a b')"]
    result = run_confined(argv, workspace=str(tmp_path), cwd=str(tmp_path),
                          timeout=5, env={"PATH": "/usr/bin"}, input="{}")

    assert result.stdout == "out"
    [(bwrap_argv, kwargs)] = fake_linux_bwrap
    assert bwrap_argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in bwrap_argv and "--unshare-pid" in bwrap_argv
    assert bwrap_argv[bwrap_argv.index("--") + 1:] == argv, "argv must not be re-parsed by a shell"
    assert kwargs["shell"] is False and kwargs["input"] == "{}" and kwargs["timeout"] == 5
    assert os.path.isdir(sandbox_mod.sandbox_home(str(tmp_path)))


def test_run_command_keeps_its_network(tmp_path, fake_linux_bwrap):
    """pip / npm installs in run_command need it; only validators lose it."""
    plan = sandbox_mod.sandboxed_command("pip install x", workspace=str(tmp_path), cwd=str(tmp_path))
    assert "--unshare-net" not in plan.argv
    assert plan.argv[-3:] == ["/bin/sh", "-c", "pip install x"]


def test_a_sandbox_that_fails_to_start_is_a_refusal_not_a_check_result(tmp_path, monkeypatch, fake_linux_bwrap):
    monkeypatch.setattr(
        sandbox_mod, "run_bounded",
        lambda argv, **k: subprocess.CompletedProcess(argv, 1, "", "bwrap: No permissions to create new namespace"))
    with pytest.raises(SandboxUnavailable, match="bwrap: No permissions"):
        run_confined(["python"], workspace=str(tmp_path), cwd=str(tmp_path), timeout=5)


def test_a_platform_without_namespaces_runs_the_check_with_a_warning(tmp_path, monkeypatch):
    """A developer's Windows/macOS box: no sandbox exists, so it runs as before."""
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: False)
    result = run_confined([sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                          workspace=str(tmp_path), cwd=str(tmp_path), timeout=60, input="piped")
    assert result.returncode == 0 and result.stdout.strip() == "piped"


def test_run_bounded_feeds_input_on_stdin(tmp_path):
    result = run_bounded([sys.executable, "-c", "import sys; print(sys.stdin.read()[::-1])"],
                         timeout=60, input="abc")
    assert result.stdout.strip() == "cba"


# --------------------------------------------------------------------------- #
# The mount plan, on a faked Linux root
# --------------------------------------------------------------------------- #
_ROOT = {"app": "dir", "bin": "usr/bin", "etc": "dir", "proc": "dir", "root": "dir",
         "tmp": "dir", "usr": "dir", "workspace": "dir"}


@pytest.fixture
def fake_root(monkeypatch):
    real_isdir = os.path.isdir
    monkeypatch.setattr(sandbox_mod.os, "listdir", lambda _p: list(_ROOT))
    monkeypatch.setattr(sandbox_mod.os.path, "islink",
                        lambda p: _ROOT.get(p.lstrip("/"), "dir") != "dir")
    monkeypatch.setattr(sandbox_mod.os, "readlink", lambda p: _ROOT[p.lstrip("/")])
    monkeypatch.setattr(sandbox_mod.os.path, "realpath", lambda p: p)
    monkeypatch.setattr(sandbox_mod.os.path, "isdir",
                        lambda p: p.startswith("/") or real_isdir(p))
    monkeypatch.setenv("BESSER_TELEMETRY_DIR", "/app/telemetry")
    monkeypatch.setenv("RUSTUP_HOME", "/root/.rustup")
    monkeypatch.setenv("HOME", "/root")


def _pairs(args, flag):
    return [tuple(args[i + 1:i + 3]) for i, a in enumerate(args) if a == flag]


def test_usr_local_and_root_are_no_longer_writable_or_shared(fake_root):
    args = sandbox_mod._mount_args("/workspace/runs/besser_llm_a_1")
    writable = [src for src, _ in _pairs(args, "--bind")]
    for shared in ("/usr/local", "/root", "/usr"):
        assert shared not in writable, f"{shared} is writable and shared between runs"
    assert ("/usr", "/usr") in _pairs(args, "--ro-bind")
    assert ("/root", "/root") in _pairs(args, "--ro-bind")


def test_each_run_gets_its_own_writable_home(fake_root):
    """Where pip's user install and the npm / cargo caches now go."""
    homes = []
    for run in ("/workspace/runs/besser_llm_a_1", "/workspace/runs/besser_llm_b_2"):
        args = sandbox_mod._mount_args(run)
        home = run + ".sandbox-home"
        assert (home, home) in _pairs(args, "--bind")
        assert ("HOME", home) in _pairs(args, "--setenv")
        homes.append(home)
    assert homes[0] != homes[1]
    # The run cleanup removes besser_llm_* entries; the home inherits it.
    assert os.path.basename(homes[0]).startswith("besser_llm_")


def test_rust_still_finds_its_toolchain_after_home_moved(fake_root):
    args = sandbox_mod._mount_args("/workspace/runs/besser_llm_a_1")
    assert ("RUSTUP_HOME", "/root/.rustup") in _pairs(args, "--setenv")


def test_telemetry_is_masked_not_exposed_read_only(fake_root):
    args = sandbox_mod._mount_args("/workspace/runs/besser_llm_a_1")
    assert ("/app", "/app") in _pairs(args, "--ro-bind")
    tmpfs = [args[i + 1] for i, a in enumerate(args) if a == "--tmpfs"]
    assert "/app/telemetry" in tmpfs
    # Mounted after /app, or the ro-bind would cover it again.
    assert args.index("/app/telemetry") > args.index("/app")


def test_a_probe_scratch_copy_is_writable_and_the_run_root_stays_hidden(fake_root):
    args = sandbox_mod._mount_args("/workspace/runs/besser_llm_a_1",
                                   writable=["/tmp/besser_probe_x"])
    binds = _pairs(args, "--bind")
    assert ("/tmp/besser_probe_x", "/tmp/besser_probe_x") in binds
    for src, _ in binds + _pairs(args, "--ro-bind"):
        if src.startswith("/workspace"):
            assert src == "/workspace/runs/besser_llm_a_1.sandbox-home", src



# --------------------------------------------------------------------------- #
# The write-time import probe and the pip dependency check
# --------------------------------------------------------------------------- #
_ORM = "from sqlalchemy.orm import declarative_base\nBase = declarative_base()\n"


def test_the_write_time_import_probe_runs_confined(tmp_path, monkeypatch, no_plain_subprocess):
    from besser.spec_driven_agent.agent import tool_executor as te

    path = tmp_path / "sql_alchemy.py"
    path.write_text(_ORM, encoding="utf-8")
    calls = []
    monkeypatch.setattr(te, "run_confined", lambda argv, **k: calls.append((argv, k)) or _Ok(),
                        raising=False)
    assert te._breaks_module_import(str(path), _ORM, _ORM + "x = 1\n", str(tmp_path)) is None
    assert len(calls) == 2  # before and after the edit
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable and kwargs["workspace"] == str(tmp_path)
    assert kwargs.get("network", False) is False


def test_the_write_time_import_probe_never_blocks_an_edit_without_a_sandbox(tmp_path, monkeypatch):
    from besser.spec_driven_agent.agent import tool_executor as te

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe ran unconfined"))

    def _unavailable(*_a, **_k):
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

    monkeypatch.setattr(te, "run_confined", _unavailable, raising=False)
    path = tmp_path / "sql_alchemy.py"
    path.write_text(_ORM, encoding="utf-8")
    # Even an edit that really breaks the import is let through: skipped, not refused.
    assert te._breaks_module_import(str(path), _ORM, "raise ImportError('x')\n", str(tmp_path)) is None
    assert path.read_text(encoding="utf-8") == _ORM


def _dependency_orch(tmp_path, monkeypatch):
    from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
    from besser.spec_driven_agent.providers.llm_client import UsageTracker

    (tmp_path / "requirements.txt").write_text("anyio\n")

    class _Client:
        model = "mock-model"
        usage = UsageTracker("mock-model")

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    model = DomainModel(name="M", types={Class(name="A", attributes={Property(name="n", type=StringType)})})
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=model, output_dir=str(tmp_path),
                           allow_shell_tools=True, enable_toolchain_validation=False,
                           enable_checkpointing=False)
    for name in ("_collect_frontend_contract_issues", "_collect_ruff_issues",
                 "_collect_execution_issues", "_collect_tsc_issues",
                 "_collect_requirement_issues", "_collect_task_issues",
                 "_collect_data_contract_issues", "_collect_missing_frontend_issue",
                 "_collect_framework_switch_issues"):
        monkeypatch.setattr(orch, name, lambda: [])
    return orch


def test_the_pip_dependency_check_runs_confined_with_network(tmp_path, monkeypatch):
    """pip may build an sdist (setup.py) to resolve; it must reach the index."""
    from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module

    orch = _dependency_orch(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(orchestrator_module, "run_bounded",
                        lambda *a, **k: pytest.fail("pip ran unconfined"), raising=False)
    monkeypatch.setattr(orchestrator_module, "run_confined",
                        lambda argv, **k: calls.append((argv, k)) or _Ok(), raising=False)
    orch._collect_validation_issues()
    pip = [(a, k) for a, k in calls if "pip" in a]
    assert pip, calls
    argv, kwargs = pip[0]
    assert "--dry-run" in argv
    assert kwargs["network"] is True and kwargs["workspace"] == str(tmp_path)


def test_the_pip_dependency_check_is_skipped_with_a_reason_without_a_sandbox(tmp_path, monkeypatch):
    from besser.spec_driven_agent.pipeline import orchestrator as orchestrator_module

    orch = _dependency_orch(tmp_path, monkeypatch)

    def _unavailable(*_a, **_k):
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")

    monkeypatch.setattr(orchestrator_module, "run_bounded",
                        lambda *a, **k: pytest.fail("pip ran unconfined"), raising=False)
    monkeypatch.setattr(orchestrator_module, "run_confined", _unavailable, raising=False)
    findings = [i for i in orch._collect_validation_issues() if "requirements.txt" in i.message]
    assert len(findings) == 1, findings
    assert findings[0].severity == "warning"
    assert "sandbox is unavailable" in findings[0].message and "SKIPPED" in findings[0].message
