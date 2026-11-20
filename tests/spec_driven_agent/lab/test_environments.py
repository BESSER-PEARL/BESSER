"""Per-change environments from shared stores (``lab/environments.py``).

Installs run through a fake runner that records each command, so nothing here installs anything.
The node_modules clone and the snapshot are real (hard links in a temporary folder).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import stat

import pytest

from besser.spec_driven_agent.lab import environments
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    RepoSpec,
)
from besser.spec_driven_agent.lab.environments import (
    STAMP_FILENAME,
    Environments,
    EnvironmentStamp,
    LabEnvironmentError,
    probe_link_mode,
)

LOCK_A = '{"name": "editor", "lockfileVersion": 3, "packages": {"a": "1.0.0"}}\n'
LOCK_B = '{"name": "editor", "lockfileVersion": 3, "packages": {"a": "2.0.0"}}\n'
BAF_TARGET_PARTS = ("venv", "lib", "python3.11", "site-packages", "baf", "platforms", "websocket",
                    "websocket_platform.py")


def _write(path, text: str = "x") -> str:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _venv_python(tree) -> str:
    if os.name == "nt":
        return os.path.join(str(tree), "venv", "Scripts", "python.exe")
    return os.path.join(str(tree), "venv", "bin", "python")


def _spec(key: str, path) -> RepoSpec:
    return RepoSpec(key, str(path), "develop", "BESSER-PEARL", key, str(path) + "-git")


class FakeRunner:
    """Records every command; ``uv venv`` makes a venv whose interpreter lives in ``python_home``."""

    def __init__(self, python_home, *, fail_on: str = "", baf_target: str = ""):
        self.python_home = str(python_home)
        self.fail_on = fail_on
        self.baf_target = baf_target
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, argv, cwd):
        argv = list(argv)
        self.calls.append((argv, cwd))
        if self.fail_on and self.fail_on in argv:
            return 1, "error: resolution failed"
        if argv[:2] == ["uv", "venv"]:
            venv = argv[-1]
            for name in (("bin", "python"), ("Scripts", "python.exe")):
                _write(os.path.join(venv, *name), "")
            _write(os.path.join(venv, "pyvenv.cfg"), f"home = {self.python_home}\nversion_info = 3.12.8\n")
        if len(argv) > 1 and argv[1] == "-c":
            return 0, "a warning first\n" + self.baf_target + "\n"
        return 0, ""

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]


@pytest.fixture
def root(tmp_path):
    return tmp_path / "work"


@pytest.fixture
def cache(root):
    return str(root / ".cache")


@pytest.fixture
def runner(tmp_path):
    return FakeRunner(tmp_path / "interpreters" / "bin")


def _besser_tree(path):
    for name in ("setup.cfg", "pyproject.toml", "requirements.txt",
                 "besser/utilities/web_modeling_editor/backend/requirements.txt", "docs/requirements.txt"):
        _write(os.path.join(str(path), *name.split("/")), name)
    return _spec(REPO_BESSER, path)


def _agent_tree(path):
    for name in ("requirements-dev.txt", "pyproject.toml", "patches/websocket_platform.py"):
        _write(os.path.join(str(path), *name.split("/")), name)
    return _spec(REPO_MODELING_AGENT, path)


def _editor_tree(path, lock: str = LOCK_A):
    _write(os.path.join(str(path), "package-lock.json"), lock)
    for package in ("editor", "webapp"):
        _write(os.path.join(str(path), "packages", package, "package.json"), "{}")
    return _spec(REPO_FRONTEND, path)


def _base_editor_with_modules(root, lock: str = LOCK_A) -> RepoSpec:
    """A base editor after its ``npm ci``: hoisted modules, a package's own modules, a workspace symlink."""
    base = root / "base" / "besser" / "frontend"
    spec = _editor_tree(base, lock)
    _write(base / "node_modules" / "left-pad" / "index.js", "module.exports = 1;")
    _write(base / "node_modules" / ".bin" / "tool", "#!/usr/bin/env node")
    os.chmod(base / "node_modules" / ".bin" / "tool", 0o755)
    _write(base / "packages" / "webapp" / "node_modules" / "vite" / "index.js", "vite")
    os.makedirs(base / "node_modules" / "@besser")
    # npm workspaces: node_modules/@besser/webapp -> ../../packages/webapp
    os.symlink(os.path.join("..", "..", "packages", "webapp"), base / "node_modules" / "@besser" / "webapp",
               target_is_directory=True)
    return spec


# -- probe ------------------------------------------------------------------------------------------


def test_probe_returns_hardlink_when_reflink_fails(cache, monkeypatch, caplog):
    def no_reflink(src, dst):
        raise LabEnvironmentError("cp: failed to clone: Operation not supported")

    monkeypatch.setattr(environments, "_reflink_copy", no_reflink)
    with caplog.at_level(logging.WARNING, logger=environments.__name__):
        assert probe_link_mode(cache) == "hardlink"
    assert "can reach other changes" in caplog.text
    assert os.listdir(cache) == []  # the probe files are removed


def test_probe_returns_reflink_when_the_clone_works(cache, monkeypatch):
    monkeypatch.setattr(environments, "_reflink_copy", lambda src, dst: shutil.copy2(src, dst))
    assert probe_link_mode(cache) == "reflink"
    assert os.listdir(cache) == []


def test_uv_link_mode_follows_the_link_mode(cache, tmp_path):
    tree = tmp_path / "work" / "changes" / "a" / "besser"
    for mode, uv_mode in (("hardlink", "hardlink"), ("reflink", "clone")):
        fake = FakeRunner(tmp_path / "interpreters" / "bin")
        if os.path.exists(tree):
            shutil.rmtree(tree)
        Environments(cache, mode=mode, runner=fake).ensure(_besser_tree(tree))
        install = fake.argvs()[1]
        assert install[install.index("--link-mode") + 1] == uv_mode
        assert install[install.index("--cache-dir") + 1] == os.path.join(cache, "uv")


# -- node_modules -----------------------------------------------------------------------------------


def _file_modes(folder) -> dict[str, int]:
    modes = {}
    for current, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(current, name)
            if not os.path.islink(path):  # Windows lists a copied directory link among the files
                modes[os.path.relpath(path, folder)] = stat.S_IMODE(os.lstat(path).st_mode)
    return modes


def test_hardlink_snapshot_files_are_read_only(root, cache):
    base = _base_editor_with_modules(root)
    snapshot = Environments(cache, mode="hardlink").snapshot_node(base)

    assert os.path.basename(snapshot) == environments._sha256_file(os.path.join(base.path, "package-lock.json"))
    modes = _file_modes(snapshot)
    assert set(modes) == {os.path.join("node_modules", "left-pad", "index.js"),
                          os.path.join("node_modules", ".bin", "tool"),
                          os.path.join("packages", "webapp", "node_modules", "vite", "index.js")}
    assert all(mode & 0o222 == 0 for mode in modes.values()), modes
    if os.name != "nt":  # the execute bit survives, so .bin scripts still run
        assert modes[os.path.join("node_modules", ".bin", "tool")] & 0o111
    # Directories stay writable: Vite writes its cache under node_modules/.vite.
    assert os.access(os.path.join(snapshot, "node_modules"), os.W_OK)
    assert os.path.islink(os.path.join(snapshot, "node_modules", "@besser", "webapp"))


def test_reflink_snapshot_stays_writable(root, cache, monkeypatch):
    monkeypatch.setattr(environments, "_reflink_copy", lambda src, dst: shutil.copytree(src, dst, symlinks=True))
    snapshot = Environments(cache, mode="reflink").snapshot_node(_base_editor_with_modules(root))
    assert all(mode & 0o200 for mode in _file_modes(snapshot).values())


def test_snapshot_is_taken_once_per_lock(root, cache):
    envs = Environments(cache, mode="hardlink")
    base = _base_editor_with_modules(root)
    first = envs.snapshot_node(base)
    _write(os.path.join(first, "marker"), "kept")
    assert envs.snapshot_node(base) == first
    assert os.path.isfile(os.path.join(first, "marker"))


def test_snapshot_refuses_a_base_without_node_modules(root, cache):
    base = _editor_tree(root / "base" / "besser" / "frontend")
    with pytest.raises(LabEnvironmentError, match="npm ci"):
        Environments(cache, mode="hardlink").snapshot_node(base)


def test_node_modules_cloned_into_two_trees_resolve_inside_each(root, cache, runner):
    envs = Environments(cache, mode="hardlink", runner=runner)
    snapshot = envs.snapshot_node(_base_editor_with_modules(root))
    trees = [root / "changes" / change / "besser" / "frontend" for change in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")]

    for tree in trees:
        stamp = envs.ensure(_editor_tree(tree))
        assert stamp.mode == "linked"

    assert runner.calls == []  # linking runs no command in the tree
    for tree in trees:
        link = os.path.join(str(tree), "node_modules", "@besser", "webapp")
        assert os.path.islink(link)
        assert os.path.realpath(link) == os.path.realpath(os.path.join(str(tree), "packages", "webapp"))
        assert os.path.isfile(os.path.join(link, "package.json"))
        cloned = os.path.join(str(tree), "packages", "webapp", "node_modules", "vite", "index.js")
        assert os.path.samefile(cloned, os.path.join(snapshot, "packages", "webapp", "node_modules", "vite",
                                                     "index.js"))


def test_changed_package_lock_runs_npm_ci_instead_of_linking(root, cache, runner, caplog):
    envs = Environments(cache, mode="hardlink", runner=runner)
    envs.snapshot_node(_base_editor_with_modules(root, LOCK_A))
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser" / "frontend"

    with caplog.at_level(logging.WARNING, logger=environments.__name__):
        stamp = envs.ensure(_editor_tree(tree, LOCK_B))

    assert stamp.mode == "installed"
    assert runner.calls == [(["npm", "ci", "--no-audit", "--no-fund", "--loglevel=error"], str(tree))]
    assert not os.path.exists(os.path.join(str(tree), "node_modules"))
    assert "npm ci" in caplog.text


def test_link_replaces_a_node_modules_symlink_without_following_it(root, cache, tmp_path):
    envs = Environments(cache, mode="hardlink")
    envs.snapshot_node(_base_editor_with_modules(root))
    elsewhere = tmp_path / "other-change-modules"
    _write(elsewhere / "kept.js", "theirs")
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser" / "frontend"
    spec = _editor_tree(tree)
    os.symlink(str(elsewhere), tree / "node_modules", target_is_directory=True)

    envs.ensure(spec)

    assert not os.path.islink(tree / "node_modules")
    assert os.path.isfile(tree / "node_modules" / "left-pad" / "index.js")
    assert os.listdir(elsewhere) == ["kept.js"]


def test_link_skips_a_package_folder_that_leaves_the_tree(root, cache, tmp_path, caplog):
    envs = Environments(cache, mode="hardlink")
    envs.snapshot_node(_base_editor_with_modules(root))
    elsewhere = tmp_path / "other-change-webapp"
    os.makedirs(elsewhere)
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser" / "frontend"
    spec = _editor_tree(tree)
    shutil.rmtree(tree / "packages" / "webapp")
    os.symlink(str(elsewhere), tree / "packages" / "webapp", target_is_directory=True)

    with caplog.at_level(logging.WARNING, logger=environments.__name__):
        envs.ensure(spec)

    assert os.listdir(elsewhere) == []
    assert "packages/webapp/node_modules" in caplog.text


def test_without_a_snapshot_the_tree_gets_npm_ci(root, cache, runner):
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser" / "frontend"
    stamp = Environments(cache, mode="hardlink", runner=runner).ensure(_editor_tree(tree))
    assert stamp.mode == "installed"
    assert runner.argvs() == [["npm", "ci", "--no-audit", "--no-fund", "--loglevel=error"]]


# -- Python environments ----------------------------------------------------------------------------


def test_besser_commands_are_provisions_with_the_trees_venv(root, cache, runner):
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser"
    stamp = Environments(cache, mode="hardlink", runner=runner).ensure(_besser_tree(tree))

    venv, install = runner.argvs()
    assert venv[:2] == ["uv", "venv"] and venv[-1] == os.path.join(str(tree), "venv")
    assert venv[venv.index("--python") + 1] == "3.12"
    assert "--seed" in venv and "--no-python-downloads" in venv
    assert install[:3] == ["uv", "pip", "install"]
    assert install[install.index("--python") + 1] == _venv_python(tree)
    assert install[install.index("-e") + 1] == f"{tree}[dev]"
    requirements = [install[i + 1] for i, arg in enumerate(install) if arg == "-r"]
    assert requirements == [os.path.join(str(tree), "besser", "utilities", "web_modeling_editor", "backend",
                                         "requirements.txt"),
                            os.path.join(str(tree), "docs", "requirements.txt")]
    assert "import-linter==2.15" in install
    assert all(cwd == str(tree) for _, cwd in runner.calls)
    assert stamp.mode == "installed" and stamp.repo == REPO_BESSER


def test_agent_commands_and_the_patch_copy_unlinks_its_target_first(root, cache, tmp_path):
    tree = root / "changes" / "aaaaaaaaaaaa" / "modeling-agent"
    target = os.path.join(str(tree), *BAF_TARGET_PARTS)
    fake = FakeRunner(tmp_path / "interpreters" / "bin", baf_target=target)
    Environments(cache, mode="hardlink", runner=fake).ensure(_agent_tree(tree))

    argvs = fake.argvs()
    py = _venv_python(tree)
    assert argvs[0][argvs[0].index("--python") + 1] == "3.11"
    assert argvs[1][-2:] == ["-r", os.path.join(str(tree), "requirements-dev.txt")]
    assert argvs[2][-3:] == ["--no-deps", "-e", str(tree)]
    assert argvs[3][:2] == [py, "-c"]
    assert argvs[4:] == [["rm", "-f", target],
                         ["cp", os.path.join(str(tree), "patches", "websocket_platform.py"), target]]


def test_agent_patch_target_outside_the_venv_is_refused(root, cache, tmp_path):
    tree = root / "changes" / "aaaaaaaaaaaa" / "modeling-agent"
    fake = FakeRunner(tmp_path / "interpreters" / "bin", baf_target=str(tmp_path / "elsewhere.py"))
    with pytest.raises(LabEnvironmentError, match="outside"):
        Environments(cache, mode="hardlink", runner=fake).ensure(_agent_tree(tree))
    assert not any(argv[0] in ("rm", "cp") for argv in fake.argvs())
    assert not os.path.exists(os.path.join(str(tree), STAMP_FILENAME))


def test_interpreter_under_the_workspace_root_is_refused(root, cache):
    fake = FakeRunner(root / "base" / "besser" / "venv" / "bin")
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser"
    with pytest.raises(LabEnvironmentError, match="workspace root"):
        Environments(cache, mode="hardlink", runner=fake).ensure(_besser_tree(tree))
    assert not any(argv[:3] == ["uv", "pip", "install"] for argv in fake.argvs())


def test_a_failed_install_raises_and_writes_no_stamp(root, cache, tmp_path):
    fake = FakeRunner(tmp_path / "interpreters" / "bin", fail_on="install")
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser"
    envs = Environments(cache, mode="hardlink", runner=fake)
    with pytest.raises(LabEnvironmentError, match="exit 1"):
        envs.ensure(_besser_tree(tree))
    assert envs.stamp(_spec(REPO_BESSER, tree)) is None


# -- stamp and runner -------------------------------------------------------------------------------


def test_the_stamp_skips_a_second_ensure(root, cache, runner):
    envs = Environments(cache, mode="hardlink", runner=runner)
    spec = _besser_tree(root / "changes" / "aaaaaaaaaaaa" / "besser")
    first = envs.ensure(spec)
    commands = len(runner.calls)

    assert envs.ensure(spec) == first
    assert len(runner.calls) == commands
    with open(os.path.join(spec.path, STAMP_FILENAME), encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored == {"repo": "besser", "inputs_sha256": first.inputs_sha256, "created_at": first.created_at,
                      "seconds": first.seconds, "mode": "installed"}
    assert envs.stamp(spec) == first


def test_changed_requirements_reinstall_into_the_existing_venv(root, cache, runner):
    envs = Environments(cache, mode="hardlink", runner=runner)
    spec = _besser_tree(root / "changes" / "aaaaaaaaaaaa" / "besser")
    first = envs.ensure(spec)
    _write(os.path.join(spec.path, "requirements.txt"), "numpy==2.3.0\n")
    runner.calls.clear()

    second = envs.ensure(spec)

    assert second.inputs_sha256 != first.inputs_sha256
    assert [argv[:3] for argv in runner.argvs()] == [["uv", "pip", "install"]]


def test_a_deleted_environment_is_made_again(root, cache, runner):
    envs = Environments(cache, mode="hardlink", runner=runner)
    spec = _besser_tree(root / "changes" / "aaaaaaaaaaaa" / "besser")
    envs.ensure(spec)
    shutil.rmtree(os.path.join(spec.path, "venv"))
    runner.calls.clear()
    envs.ensure(spec)
    assert runner.argvs()[0][:2] == ["uv", "venv"]


def test_the_runner_passed_to_ensure_is_the_one_used(root, cache, tmp_path):
    default = FakeRunner(tmp_path / "interpreters" / "bin")
    sandboxed = FakeRunner(tmp_path / "interpreters" / "bin")
    envs = Environments(cache, mode="hardlink", runner=default)

    envs.ensure(_besser_tree(root / "changes" / "aaaaaaaaaaaa" / "besser"), runner=sandboxed)
    assert default.calls == [] and len(sandboxed.calls) == 2

    envs.ensure(_besser_tree(root / "base" / "besser"))
    assert len(default.calls) == 2 and len(sandboxed.calls) == 2


def test_stamp_is_none_when_missing_or_unreadable(root, cache, caplog):
    envs = Environments(cache, mode="hardlink")
    spec = _besser_tree(root / "changes" / "aaaaaaaaaaaa" / "besser")
    assert envs.stamp(spec) is None
    _write(os.path.join(spec.path, STAMP_FILENAME), "{torn")
    with caplog.at_level(logging.WARNING, logger=environments.__name__):
        assert envs.stamp(spec) is None
    assert STAMP_FILENAME in caplog.text


def test_inputs_hash_covers_every_patch_file(root):
    tree = root / "changes" / "aaaaaaaaaaaa" / "modeling-agent"
    spec = _agent_tree(tree)
    before = environments._inputs_sha256(spec)
    _write(os.path.join(str(tree), "patches", "another.py"), "patched")
    assert environments._inputs_sha256(spec) != before


def test_stamp_round_trips(root, cache):
    stamp = EnvironmentStamp(repo="besser", inputs_sha256="ab" * 32, created_at=1.5, seconds=2.0, mode="linked")
    tree = root / "changes" / "aaaaaaaaaaaa" / "besser"
    _write(os.path.join(str(tree), STAMP_FILENAME), json.dumps(dataclasses.asdict(stamp)))
    assert Environments(cache, mode="hardlink").stamp(_spec(REPO_BESSER, tree)) == stamp
