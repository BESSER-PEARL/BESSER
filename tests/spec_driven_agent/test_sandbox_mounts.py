"""Mount rules a caller adds to a sandbox: read-only binds, masked paths, no workspace bind.

The lab confines its shell, checks and preview with one policy (BUILD_PLAN R2): the change's trees
read-write, the git root read-only, the workspace root and the lab's own state masked. The shared
sandbox applies it in one order: masked tmpfs first, then the read-write binds, then the read-only
binds, so a mask may be an ancestor of a bind but never inside one. Every default leaves the argv
byte-identical to the pre-change module (the golden test below, recorded from it).

All of it is asserted on a faked Linux root, so it runs on every OS.
"""

import os
from types import SimpleNamespace

import pytest

from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution.sandbox import (
    SANDBOX_POLICY_ENV,
    SHELL_NETWORK_ENV,
    SandboxMounts,
    SandboxUnavailable,
    run_confined,
    sandboxed_command,
)

_ROOT = {"app": "dir", "bin": "usr/bin", "etc": "dir", "proc": "dir", "root": "dir",
         "srv": "dir", "tmp": "dir", "usr": "dir", "workspace": "dir"}

_RUN = "/workspace/runs/besser_llm_a_1"
_HOME = _RUN + ".sandbox-home"


@pytest.fixture
def linux_box(monkeypatch):
    """A Linux host with a working bwrap and pasta, faked; records run_bounded's argv."""
    real_isdir = os.path.isdir
    monkeypatch.setattr(sandbox_mod.os, "listdir", lambda _p: list(_ROOT))
    monkeypatch.setattr(sandbox_mod.os.path, "islink",
                        lambda p: _ROOT.get(p.lstrip("/"), "dir") != "dir")
    monkeypatch.setattr(sandbox_mod.os, "readlink", lambda p: _ROOT[p.lstrip("/")])
    monkeypatch.setattr(sandbox_mod.os.path, "realpath", lambda p: p)
    monkeypatch.setattr(sandbox_mod.os.path, "isdir",
                        lambda p: (p.startswith("/") and "missing" not in p) or real_isdir(p))
    monkeypatch.setattr(sandbox_mod.os, "makedirs", lambda *_a, **_k: None)
    monkeypatch.setenv("BESSER_TELEMETRY_DIR", "/app/telemetry")
    monkeypatch.setenv("BESSER_INCIDENT_LOG_DIR", "/app/incidents")
    monkeypatch.setenv("RUSTUP_HOME", "/root/.rustup")
    monkeypatch.setenv(SANDBOX_POLICY_ENV, "auto")
    monkeypatch.setenv(SHELL_NETWORK_ENV, "auto")
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: True)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sandbox_mod, "_selftest", lambda _b: (True, ""))
    monkeypatch.setattr(sandbox_mod, "_network_selftest", lambda _b: (True, ""))
    monkeypatch.setattr(sandbox_mod, "_container_nameserver", lambda: "127.0.0.11")
    monkeypatch.setattr(sandbox_mod, "_sandbox_resolv_conf",
                        lambda: "/tmp/besser-sandbox-resolv.conf")
    ran = []
    monkeypatch.setattr(sandbox_mod, "run_bounded", lambda argv, **_k: ran.append(argv)
                        or SimpleNamespace(returncode=0, stdout="", stderr=""))
    return ran


def _pairs(args, flag):
    return [tuple(args[i + 1:i + 3]) for i, a in enumerate(args) if a == flag]


def _index(args, *triple):
    for i in range(len(args) - 2):
        if tuple(args[i:i + 3]) == triple:
            return i
    raise AssertionError(f"{triple} not in argv")


# --------------------------------------------------------------------------- #
# Golden: argv recorded from the pre-change module (D1 test 12)
# --------------------------------------------------------------------------- #
_ISOLATION = ["--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
              "--unshare-cgroup-try", "--die-with-parent", "--new-session"]
_SYSTEM = ["--ro-bind", "/app", "/app", "--symlink", "usr/bin", "/bin",
           "--ro-bind", "/etc", "/etc", "--ro-bind", "/root", "/root",
           "--ro-bind", "/srv", "/srv", "--ro-bind", "/usr", "/usr",
           "--proc", "/proc", "--dev", "/dev",
           "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", "/var/tmp", "--tmpfs", "/dev/shm",
           "--tmpfs", "/app/telemetry", "--tmpfs", "/app/incidents",
           "--bind", _HOME, _HOME, "--setenv", "HOME", _HOME,
           "--setenv", "RUSTUP_HOME", "/root/.rustup"]
_GOLDEN_SHELL = ["/usr/bin/bwrap", *_ISOLATION, *_SYSTEM, "--bind", _RUN, _RUN,
                 "--chdir", _RUN, "--", "/bin/sh", "-c", "echo hi"]
_GOLDEN_PROBE = ["/usr/bin/bwrap", "--unshare-net", *_ISOLATION, *_SYSTEM,
                 "--bind", "/tmp/besser_probe_x", "/tmp/besser_probe_x",
                 "--chdir", _RUN + "/app", "--", "python", "-c", "pass"]
_GOLDEN_SESSION = ["/usr/bin/bwrap", "--unshare-net", *_ISOLATION, *_SYSTEM, "--bind", _RUN, _RUN,
                   "--ro-bind", "/tmp/besser-sandbox-resolv.conf", "/etc/resolv.conf",
                   "--chdir", _RUN, "--", "sup"]
_GOLDEN_CONFINED = ["/usr/bin/bwrap", "--unshare-net", *_ISOLATION, *_SYSTEM, "--bind", _RUN, _RUN,
                    "--chdir", _RUN, "--", "python", "-c", "pass"]


def test_default_arguments_give_the_pre_change_argv(linux_box):
    assert sandboxed_command("echo hi", workspace=_RUN, cwd=_RUN).argv == _GOLDEN_SHELL
    assert sandboxed_command(["python", "-c", "pass"], workspace=_RUN, cwd=_RUN + "/app",
                             network=False, writable=["/tmp/besser_probe_x"]).argv == _GOLDEN_PROBE
    session = sandboxed_command(["sup"], workspace=_RUN, cwd=_RUN, shell_network=True)
    assert session.argv == _GOLDEN_SESSION and session.private_network is True
    run_confined(["python", "-c", "pass"], workspace=_RUN, cwd=_RUN, timeout=5)
    assert linux_box == [_GOLDEN_CONFINED]


def test_empty_mounts_change_nothing(linux_box):
    empty = SandboxMounts()
    plan = sandboxed_command("echo hi", workspace=_RUN, cwd=_RUN, writable=empty.writable,
                             read_only=empty.read_only, masked=empty.masked)
    assert plan.argv == _GOLDEN_SHELL


# --------------------------------------------------------------------------- #
# The lab's layout: trees read-write, git root read-only, the rest masked
# --------------------------------------------------------------------------- #
_LAB_ROOT = "/srv/lab/work"
_TREE = _LAB_ROOT + "/changes/c1/besser"
_GIT_ROOT = "/srv/lab/git"


def test_read_only_and_masked_binds(linux_box):
    """D1 test 1: the git root read-only after the last read-write bind; a mask under a visible
    top level becomes a tmpfs; one under a hidden top level is not mounted again."""
    argv = sandboxed_command("ls", workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                             read_only=(_GIT_ROOT,),
                             masked=("/srv/lab/state", "/app/lab-checkout")).argv

    assert ("/srv/lab/git", "/srv/lab/git") in _pairs(argv, "--ro-bind")
    last_bind = max(i for i, a in enumerate(argv) if a == "--bind")
    assert _index(argv, "--ro-bind", _GIT_ROOT, _GIT_ROOT) > last_bind
    tmpfs = [i for i, a in enumerate(argv) if a == "--tmpfs"]
    assert "/app/lab-checkout" in [argv[i + 1] for i in tmpfs]
    assert "/srv/lab/state" not in [argv[i + 1] for i in tmpfs], \
        "/srv is not mounted at all; a tmpfs there only creates the path"
    assert ("/srv", "/srv") not in _pairs(argv, "--ro-bind")
    assert (_TREE, _TREE) in _pairs(argv, "--bind")


def test_a_masked_ancestor_of_a_writable_path_is_mounted_before_the_bind(linux_box):
    """The mask hides the tree's siblings (other changes, base/, .cache/); bwrap creates the bind
    target inside the tmpfs, so the order matters."""
    argv = sandboxed_command("ls", workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                             masked=(_LAB_ROOT,)).argv
    assert _index(argv, "--tmpfs", _LAB_ROOT, "--bind") < _index(argv, "--bind", _TREE, _TREE)


def test_masks_come_before_every_bind_and_read_only_binds_after_them(linux_box):
    base = _LAB_ROOT + "/base/besser"
    argv = sandboxed_command("ls", workspace=_LAB_ROOT, cwd=_LAB_ROOT, writable=[_TREE],
                             read_only=(_GIT_ROOT, base), masked=(_LAB_ROOT, "/app/lab-checkout")).argv
    tmpfs =[i for i, a in enumerate(argv) if a == "--tmpfs" and argv[i + 1] in
             (_LAB_ROOT, "/app/lab-checkout")]
    binds = [i for i, a in enumerate(argv) if a == "--bind"]
    ro = [_index(argv, "--ro-bind", p, p) for p in (_GIT_ROOT, base)]
    assert len(tmpfs) == 2
    assert max(tmpfs) < min(binds) and max(binds) < min(ro)


@pytest.mark.parametrize("masked, read_only", [
    ((_TREE + "/.git",), ()),
    ((), (_TREE + "/.git",)),
    ((_TREE,), ()),
])
def test_a_mask_or_read_only_path_inside_a_writable_path_is_refused(linux_box, masked, read_only):
    """A later read-write bind would undo a mask under it; an in-place read-only bind is defeated
    by renaming an ancestor directory (D1 decision 5)."""
    with pytest.raises(ValueError) as exc:
        sandboxed_command("ls", workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                          read_only=read_only, masked=masked)
    message = str(exc.value)
    assert (masked or read_only)[0] in message and _TREE in message


def test_a_mask_inside_a_read_only_path_is_refused(linux_box):
    """The read-only bind comes after the mask and would show what it hides."""
    with pytest.raises(ValueError) as exc:
        sandboxed_command("ls", workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                          read_only=(_GIT_ROOT,), masked=("/srv/lab/git/hooks",))
    assert "/srv/lab/git/hooks" in str(exc.value) and _GIT_ROOT in str(exc.value)


def test_a_missing_masked_directory_is_skipped(linux_box):
    """Nothing to hide; a tmpfs would only create the path."""
    argv = sandboxed_command("echo hi", workspace=_RUN, cwd=_RUN, masked=("/app/missing",)).argv
    assert argv == _GOLDEN_SHELL


def test_writable_empty_binds_nothing_read_write_but_the_home(linux_box):
    """A draft's plan step: the base trees read-only, nothing writable."""
    base = _LAB_ROOT + "/base/besser"
    argv = sandboxed_command("ls", workspace=_LAB_ROOT, cwd=base, writable=(),
                             read_only=(base, _GIT_ROOT), masked=(_LAB_ROOT,)).argv
    home = _LAB_ROOT + ".sandbox-home"
    assert _pairs(argv, "--bind") == [(home, home)]
    assert (base, base) in _pairs(argv, "--ro-bind")
    assert ("/srv", "/srv") not in _pairs(argv, "--ro-bind")


def test_run_confined_passes_the_mounts(linux_box):
    run_confined(["pytest"], workspace=_LAB_ROOT, cwd=_TREE, timeout=5, writable=[_TREE],
                 read_only=(_GIT_ROOT,), masked=(_LAB_ROOT,))
    argv = linux_box[0]
    assert "--unshare-net" in argv
    assert _index(argv, "--tmpfs", _LAB_ROOT, "--bind") < _index(argv, "--bind", _TREE, _TREE) \
        < _index(argv, "--ro-bind", _GIT_ROOT, _GIT_ROOT)


# --------------------------------------------------------------------------- #
# The shell session and the tool executor carry the mounts
# --------------------------------------------------------------------------- #
_LAB_MOUNTS = SandboxMounts(writable=(_TREE,), read_only=(_GIT_ROOT,), masked=(_LAB_ROOT,))


def _record_plans(monkeypatch, module):
    plans = []

    def _refuse(command, **kwargs):
        plans.append(kwargs)
        raise SandboxUnavailable("recorded")

    monkeypatch.setattr(module, "sandboxed_command", _refuse)
    return plans


def test_a_shell_session_plans_its_sandbox_with_the_mounts(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import shell_session

    plans = _record_plans(monkeypatch, shell_session)
    with pytest.raises(SandboxUnavailable):
        shell_session.ShellSession(str(tmp_path), mounts=_LAB_MOUNTS).run(
            "true", working_dir=None, timeout=5)
    assert plans[0]["writable"] == (_TREE,)
    assert plans[0]["read_only"] == (_GIT_ROOT,) and plans[0]["masked"] == (_LAB_ROOT,)
    assert plans[0]["shell_network"] is True


def test_a_shell_session_without_mounts_plans_as_before(tmp_path, monkeypatch):
    from besser.spec_driven_agent.execution import shell_session

    plans = _record_plans(monkeypatch, shell_session)
    with pytest.raises(SandboxUnavailable):
        shell_session.ShellSession(str(tmp_path)).run("true", working_dir=None, timeout=5)
    assert plans[0]["writable"] is None
    assert plans[0]["read_only"] == () and plans[0]["masked"] == ()


def test_the_tool_executor_hands_the_mounts_to_its_shell_and_its_import_probe(tmp_path, monkeypatch):
    from besser.spec_driven_agent.agent import tool_executor as te

    executor = te.ToolExecutor(workspace=str(tmp_path), allow_shell=True,
                               sandbox_mounts=_LAB_MOUNTS)
    assert executor._shell.mounts == _LAB_MOUNTS

    orm = "from sqlalchemy.orm import declarative_base\nBase = declarative_base()\n"
    path = tmp_path / "sql_alchemy.py"
    path.write_text(orm, encoding="utf-8")
    calls = []
    monkeypatch.setattr(te, "run_confined", lambda argv, **k: calls.append(k)
                        or SimpleNamespace(returncode=0, stdout="", stderr=""))
    result = executor._modify_file({"path": "sql_alchemy.py", "old_text": "Base = declarative_base()",
                                    "new_text": "Base = declarative_base()\nx = 1"})
    assert "error" not in result, result
    assert calls and all(k["writable"] == (_TREE,) and k["read_only"] == (_GIT_ROOT,)
                         and k["masked"] == (_LAB_ROOT,) for k in calls)


# --------------------------------------------------------------------------- #
# A private network on request (preview services, sandboxed installs)
# --------------------------------------------------------------------------- #
def test_private_network_unshares_whatever_the_shell_network_mode(linux_box, monkeypatch):
    monkeypatch.setenv(SHELL_NETWORK_ENV, "shared")
    plan = sandboxed_command(["svc"], workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                             private_network=True)
    assert plan.private_network is True and "--unshare-net" in plan.argv
    assert ("/tmp/besser-sandbox-resolv.conf", "/etc/resolv.conf") in _pairs(plan.argv, "--ro-bind")


def test_private_network_refuses_when_it_cannot_be_set_up(linux_box, monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_network_selftest",
                        lambda _b: (False, "pasta (passt) is not installed"))
    with pytest.raises(SandboxUnavailable) as exc:
        sandboxed_command(["svc"], workspace=_LAB_ROOT, cwd=_TREE, writable=[_TREE],
                          private_network=True)
    assert "pasta (passt) is not installed" in exc.value.detail


def test_private_network_on_a_platform_without_namespaces_runs_unconfined(monkeypatch):
    """Same rule as run_command (D1 decision 9): the caller reports it as unconfined."""
    monkeypatch.setattr(sandbox_mod, "sandbox_supported_platform", lambda: False)
    plan = sandboxed_command(["svc"], workspace=_LAB_ROOT, cwd=".", private_network=True)
    assert plan.mode == "unconfined-platform" and plan.private_network is False
