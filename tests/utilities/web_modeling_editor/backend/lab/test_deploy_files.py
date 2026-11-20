"""The lab's deploy files (``scripts/lab/``, local mode) against the code they configure.

Settings names against :class:`BackendSettings`, the nginx site as ``provision.sh`` renders it against the
preview's port scheme, the base clones ``setup_workspace.sh`` makes against :meth:`LabWorkspace.ensure_ready`, and
the local container's published ports. The shell scripts run in bash (Git for Windows' on Windows); without one,
those tests skip.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
import yaml

from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, repo_specs_from_settings
from besser.utilities.web_modeling_editor.backend.services.lab.preview import (
    PREVIEW_AGENT_PORT,
    PREVIEW_BACKEND_PORT,
    PREVIEW_FRONTEND_PORT,
    SLOT_PORT_STEP,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings
from tests.spec_driven_agent.lab.conftest import git, write

REPO = Path(__file__).resolve().parents[5]
SCRIPTS = REPO / "scripts" / "lab"
LAB_ENV = SCRIPTS / "lab.env.example"
LOCAL_ENV = REPO / ".env.lab.example"
SHELL_SCRIPTS = (SCRIPTS / "provision.sh", SCRIPTS / "setup_workspace.sh", REPO / "docker" / "lab" / "entrypoint.sh")

# provision.sh replaces this name with LAB_HOST in lab.env.example and nginx-lab.conf.
EXAMPLE_HOST = "lab.example.org"
LAB_BACKEND_PORT = 9000
LAB_EDITOR_PORT = 8080

_ASSIGNMENT = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=(.*)$")


def _assignments(path: Path) -> dict[str, str]:
    """``NAME=value`` lines of an env file, commented-out ones included (they name settings too)."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT.match(line)
        if match:
            found.setdefault(match.group(1), match.group(2).strip())
    return found


def _active_assignments(path: Path) -> dict[str, str]:
    """The uncommented ``NAME=value`` lines: what the file sets."""
    return {name: value for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#") and "=" in line
            for name, _, value in [line.strip().partition("=")]}


def _lab_setting_names() -> set[str]:
    return {field.validation_alias for field in BackendSettings.model_fields.values()
            if isinstance(field.validation_alias, str) and field.validation_alias.startswith("BESSER_LAB_")}


def _bash() -> Optional[str]:
    """A bash that runs these scripts: Git for Windows' on Windows (never WSL's launcher), else the PATH's."""
    if sys.platform == "win32":
        git_exe = shutil.which("git")
        if not git_exe:
            return None
        # git.exe is in <Git>/cmd, <Git>/bin or <Git>/mingw64/bin; bash.exe in <Git>/bin.
        for root in Path(git_exe).resolve().parents[:3]:
            if (root / "bin" / "bash.exe").is_file() and (root / "usr" / "bin").is_dir():
                return str(root / "bin" / "bash.exe")
        return None
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash to run the deploy scripts with")


def _posix(path: Path | str) -> str:
    """A path as the scripts' bash takes it (forward slashes; Git for Windows reads C:/...)."""
    return str(path).replace("\\", "/")


def _run_bash(script: str, *, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=env, timeout=300)


# -- settings ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("env_file", [LAB_ENV, LOCAL_ENV], ids=["lab.env.example", ".env.lab.example"])
def test_every_lab_variable_of_an_env_example_is_a_backend_setting(env_file):
    named = {name for name in _assignments(env_file) if name.startswith("BESSER_LAB_")}
    assert named, f"{env_file.name} names no BESSER_LAB_ setting"
    assert named <= _lab_setting_names(), f"not BackendSettings fields: {sorted(named - _lab_setting_names())}"


def test_lab_env_example_sets_every_lab_setting():
    missing = _lab_setting_names() - set(_active_assignments(LAB_ENV))
    assert not missing, f"lab.env.example lacks {sorted(missing)}"


def test_lab_env_example_carries_the_decided_values():
    values = _active_assignments(LAB_ENV)
    assert values["BESSER_LAB_PLAN_LLM_MODEL"] == "claude-opus-5-5"
    assert values["BESSER_LLM_SHELL_NETWORK"] == "private"
    assert values["BESSER_LAB_ADMIN_LOGINS"] == ""
    # The comment above the admins line says to fill it in (BUILD_PLAN decision 28).
    text = LAB_ENV.read_text(encoding="utf-8")
    comment = text[:text.index("BESSER_LAB_ADMIN_LOGINS=")].rstrip().rsplit("\n", 1)[-1]
    assert comment.startswith("#") and "maintainer" in comment.lower()


def test_lab_env_example_is_valid_for_the_backend(monkeypatch):
    """What provision.sh installs (the host name replaced) is accepted by BackendSettings as the lab expects."""
    for name, value in _active_assignments(LAB_ENV).items():
        monkeypatch.setenv(name, value.replace(EXAMPLE_HOST, "lab.test"))
    settings = BackendSettings()
    assert settings.lab_enabled and settings.lab_preview_enabled
    assert settings.lab_preview_urls == ("https://preview.lab.test", "https://preview-2.lab.test")
    assert settings.deployment_url == "https://lab.test"
    for folder in (settings.lab_git_root, settings.lab_state_dir):  # beside the workspace, never inside it
        assert not folder.startswith(settings.lab_workspace_root + "/")


# -- nginx ------------------------------------------------------------------------------------------


def _server_blocks(conf: str) -> list[str]:
    """The text of each top-level ``server { ... }`` block (comments dropped)."""
    text = "\n".join(line.split("#", 1)[0] for line in conf.splitlines())
    blocks, depth, start = [], 0, None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0 and re.search(r"\bserver\s*$", text[:index]):
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start:index + 1])
                start = None
    assert depth == 0, "unbalanced braces in the rendered site"
    return blocks


def _upstream_ports(block: str) -> set[int]:
    return {int(port) for port in re.findall(r"proxy_pass\s+http://127\.0\.0\.1:(\d+)", block)}


@pytest.fixture(scope="module")
def rendered_site() -> str:
    """nginx-lab.conf rendered by provision.sh's own function for lab.test, two preview hosts, two networks."""
    if BASH is None:
        pytest.skip("no bash to run provision.sh with")
    result = _run_bash(
        f'set -e; source "{_posix(SCRIPTS / "provision.sh")}"; '
        f'render_nginx_conf "{_posix(SCRIPTS / "nginx-lab.conf")}" lab.test "192.0.2.0/24 2001:db8::/32" '
        '$(preview_hosts "https://preview.lab.test,https://preview-2.lab.test/")')
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_rendered_site_has_the_lab_block_and_one_block_per_preview_slot(rendered_site):
    blocks = _server_blocks(rendered_site)
    tls = [block for block in blocks if re.search(r"listen\s+443\b", block)]
    names = [re.search(r"server_name\s+([^;]+);", block).group(1).split() for block in tls]
    assert names == [["lab.test"], ["preview.lab.test"], ["preview-2.lab.test"]]
    # The plain-HTTP block only answers the certificate challenge and redirects, for every name.
    plain = [block for block in blocks if block not in tls]
    assert len(plain) == 1 and "return 301 https://" in plain[0]
    assert set(re.search(r"server_name\s+([^;]+);", plain[0]).group(1).split()) == {
        "lab.test", "preview.lab.test", "preview-2.lab.test"}
    assert EXAMPLE_HOST not in rendered_site and "__" not in rendered_site


def test_every_rendered_block_admits_only_the_allowed_networks(rendered_site):
    for block in _server_blocks(rendered_site):
        assert "allow 192.0.2.0/24;" in block and "allow 2001:db8::/32;" in block
        assert "deny all;" in block


def test_lab_block_sends_the_api_to_the_lab_backend_and_the_rest_to_its_editor(rendered_site):
    lab = _server_blocks(rendered_site)[1]
    assert _upstream_ports(lab) == {LAB_BACKEND_PORT, LAB_EDITOR_PORT}
    api = re.search(r"location /besser_api/ \{([^}]*)\}", lab).group(1)
    assert f"127.0.0.1:{LAB_BACKEND_PORT}" in api and "proxy_buffering off;" in api
    assert re.search(r'add_header X-Frame-Options "DENY" always;', lab)
    assert "frame-ancestors" not in lab


def test_preview_blocks_use_the_slot_ports_and_never_reach_the_lab_backend(rendered_site):
    previews = _server_blocks(rendered_site)[2:]
    assert len(previews) == 2
    for index, block in enumerate(previews):
        offset = index * SLOT_PORT_STEP
        assert _upstream_ports(block) == {PREVIEW_BACKEND_PORT + offset, PREVIEW_FRONTEND_PORT + offset,
                                          PREVIEW_AGENT_PORT + offset}
        assert LAB_BACKEND_PORT not in _upstream_ports(block)
        assert 'Content-Security-Policy "frame-ancestors https://lab.test"' in block
        assert "X-Frame-Options" not in block


def test_preview_hosts_refuses_what_is_not_a_host_name():
    if BASH is None:
        pytest.skip("no bash to run provision.sh with")
    result = _run_bash(f'source "{_posix(SCRIPTS / "provision.sh")}"; preview_hosts "https://bad_host;x"')
    assert result.returncode != 0 and "not a preview origin" in result.stderr


# -- scripts ----------------------------------------------------------------------------------------


@needs_bash
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_parse(script):
    result = subprocess.run([BASH, "-n", _posix(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def _make_upstreams(home: Path) -> dict[str, str]:
    """Three local "upstream" repositories, BESSER holding the editor as a submodule (an empty folder in a clone)."""
    upstream = {key: str(home / "upstream" / name) for key, name in
                ((REPO_FRONTEND, "editor"), (REPO_BESSER, "besser"), (REPO_MODELING_AGENT, "modeling-agent"))}
    branches = {key: DEFAULT_UPSTREAMS[key].rsplit("@", 1)[1] for key in upstream}
    for key, path in upstream.items():
        os.makedirs(path)
        git(path, "init", "-q", "-b", branches[key])
        write(os.path.join(path, "README.md"), f"{key}\n")
        git(path, "add", "-A")
        git(path, "commit", "-q", "-m", "initial")
    git(upstream[REPO_BESSER], "-c", "protocol.file.allow=always", "submodule", "add", "-q",
        upstream[REPO_FRONTEND], FRONTEND_SUBMODULE_PATH)
    git(upstream[REPO_BESSER], "commit", "-q", "-m", "add the editor")
    return upstream


def _clone_bases(root: Path, git_root: Path, upstream: dict[str, str]) -> subprocess.CompletedProcess:
    """setup_workspace.sh's ensure_base_clone for each repository, in the script's order."""
    specs = {spec.key: spec for spec in repo_specs_from_settings(str(root), str(git_root), DEFAULT_UPSTREAMS)}
    calls = "; ".join(
        f'ensure_base_clone {key} "{_posix(specs[key].path)}" "{_posix(specs[key].git_dir)}" '
        f'"{_posix(upstream[key])}" {specs[key].base_branch}'
        for key in (REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT))
    return _run_bash(f'set -e; source "{_posix(SCRIPTS / "setup_workspace.sh")}"; {calls}')


@needs_bash
def test_setup_workspace_clones_the_bases_the_lab_accepts(tmp_path):
    upstream = _make_upstreams(tmp_path)
    root, git_root, state = tmp_path / "work", tmp_path / "git", tmp_path / "state"
    result = _clone_bases(root, git_root, upstream)
    assert result.returncode == 0, result.stderr

    specs = repo_specs_from_settings(str(root), str(git_root), DEFAULT_UPSTREAMS)
    workspace = LabWorkspace(str(root), specs, state_dir=str(state), git_root=str(git_root))
    workspace.ensure_ready()  # base branches checked out, git directories outside the workspace
    assert not [folder for folder, names, _ in os.walk(root) if ".git" in names], "a .git directory in the workspace"
    editor = next(spec for spec in specs if spec.key == REPO_FRONTEND)
    assert Path(editor.git_dir) == git_root / "besser" / "modules" / Path(*FRONTEND_SUBMODULE_PATH.split("/"))
    assert (Path(editor.path) / "README.md").is_file()


@needs_bash
def test_setup_workspace_runs_again_and_brings_the_bases_to_upstream(tmp_path):
    upstream = _make_upstreams(tmp_path)
    root, git_root = tmp_path / "work", tmp_path / "git"
    assert _clone_bases(root, git_root, upstream).returncode == 0
    write(os.path.join(upstream[REPO_MODELING_AGENT], "agent.py"), "AGENT = 2\n")
    git(upstream[REPO_MODELING_AGENT], "add", "-A")
    git(upstream[REPO_MODELING_AGENT], "commit", "-q", "-m", "upstream work")

    result = _clone_bases(root, git_root, upstream)
    assert result.returncode == 0, result.stderr
    agent = next(spec for spec in repo_specs_from_settings(str(root), str(git_root), DEFAULT_UPSTREAMS)
                 if spec.key == REPO_MODELING_AGENT)
    assert (Path(agent.path) / "agent.py").read_text(encoding="utf-8") == "AGENT = 2\n"
    assert git(str(tmp_path), f"--git-dir={agent.git_dir}", "rev-parse", f"refs/remotes/upstream/{agent.base_branch}") \
        == git(upstream[REPO_MODELING_AGENT], "rev-parse", "HEAD")


@needs_bash
def test_setup_workspace_refuses_a_git_directory_inside_the_workspace(tmp_path):
    upstream = _make_upstreams(tmp_path)
    root, git_root = tmp_path / "work", tmp_path / "git"
    agent_tree = root / "base" / "modeling-agent"
    git(str(tmp_path), "clone", "-q", upstream[REPO_MODELING_AGENT], str(agent_tree))  # phase 1's layout
    result = _clone_bases(root, git_root, upstream)
    assert result.returncode != 0
    assert "inside the workspace" in result.stderr


# -- local mode -------------------------------------------------------------------------------------


def _lab_service() -> dict:
    compose = yaml.safe_load((REPO / "docker-compose.lab.yml").read_text(encoding="utf-8"))
    (service,) = compose["services"].values()
    return service


def test_local_container_publishes_its_ports_on_loopback_only():
    ports = _lab_service()["ports"]
    assert all(str(port).startswith("127.0.0.1:") for port in ports)
    published = {int(str(port).rsplit(":", 1)[1]) for port in ports}
    assert published == {LAB_EDITOR_PORT, LAB_BACKEND_PORT, PREVIEW_BACKEND_PORT, PREVIEW_FRONTEND_PORT,
                         PREVIEW_AGENT_PORT}


def test_local_container_can_sandbox_and_keeps_its_workspace_in_a_volume():
    service = _lab_service()
    assert {"seccomp=unconfined", "apparmor=unconfined"} <= set(service["security_opt"])
    assert any("/dev/net/tun" in device for device in service["devices"])
    assert any(str(volume).endswith(":/srv/besser-lab") for volume in service["volumes"])
    assert service["build"]["target"] == "lab"
    environment = service["environment"]
    assert environment["BESSER_LAB_PREVIEW_BIND"] == "0.0.0.0"  # pasta forwards into the container's interface
    for name in environment:
        if name.startswith("BESSER_LAB_"):
            assert name in _lab_setting_names()


def test_the_lab_editor_listens_on_loopback_only():
    """The editor's Express server binds HOST (every interface without it); on the host only nginx reaches it."""
    unit = (SCRIPTS / "besser-lab-editor.service").read_text(encoding="utf-8").splitlines()
    assert "Environment=HOST=127.0.0.1" in unit
