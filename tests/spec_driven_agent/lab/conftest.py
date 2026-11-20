"""Shared fixtures of the lab agent's tests: a scripted LLM client and a real lab workspace.

Scripted client: ``scripted_client(responses, stream=False, cost_per_call=0.0)`` (a fixture)
builds a client that answers each call with the next scripted response; a scripted
exception is raised instead. ``tool_use`` and ``end_turn`` build responses; import
them with ``from tests.spec_driven_agent.lab.conftest import end_turn, tool_use``.

Workspace: ``lab_layout`` (a fixture, :class:`LabLayout`) is a lab disk layout as
``scripts/lab/setup_workspace.sh`` makes it, under ``tmp_path``: three local "upstream"
repositories (BESSER with the editor as a submodule, the editor, the modeling agent),
their base clones under ``<root>/base`` with every git directory in a separate git root,
and an empty state folder. ``workspace.upstream_url`` is patched to the local upstreams,
so ``refresh_bases`` and ``open_change`` fetch without a network. ``lab_workspace`` is a
:class:`LabWorkspace` on it. Tests outside this folder import both fixtures (and
``lab_layout_template``, which they depend on) from this module. ``git``, ``write`` and
``read`` are the helpers the workspace tests share.
"""

from __future__ import annotations

import itertools
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import pytest

from besser.spec_driven_agent.lab import workspace as workspace_module
from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    RepoSpec,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, repo_specs_from_settings

_ids = itertools.count(1)


class Block:
    """A content block shaped like the provider SDKs' (attributes, not keys)."""

    def __init__(self, block_type: str, **fields: Any):
        self.type = block_type
        for name, value in fields.items():
            setattr(self, name, value)


def tool_use(name: str, input: dict, *more: tuple[str, dict], text: str = "") -> dict:
    """A reply calling ``name`` with ``input``, then each ``(name, input)`` in ``more``, in one turn."""
    content = [Block("text", text=text)] if text else []
    for call_name, call_input in ((name, input), *more):
        content.append(Block("tool_use", id=f"toolu_{next(_ids)}", name=call_name, input=call_input))
    return {"stop_reason": "tool_use", "content": content}


def end_turn(text: str = "Done.") -> dict:
    """A final reply with ``text``."""
    return {"stop_reason": "end_turn", "content": [Block("text", text=text)]}


class _Usage:
    def __init__(self):
        self.estimated_cost = 0.0


class ScriptedClient:
    """Answers ``chat`` with the scripted responses, in order, and records every call."""

    model = "scripted-model"
    max_tokens = 4096

    def __init__(self, responses: list, *, cost_per_call: float = 0.0):
        self.responses = list(responses)
        self.cost_per_call = cost_per_call
        self.usage = _Usage()
        #: (system, a copy of the messages, the tool names) of each call
        self.calls: list[tuple[str, list[dict], list[str]]] = []
        #: the ``model_override`` of each call (None: the client's own model)
        self.models: list[str | None] = []

    def _next(self, system: str, messages: list[dict], tools: list[dict], model_override: str | None = None) -> dict:
        self.calls.append((system, list(messages), [tool["name"] for tool in tools]))
        self.models.append(model_override)
        if not self.responses:
            raise AssertionError(f"the scripted client has no response left for call {len(self.calls)}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        self.usage.estimated_cost += self.cost_per_call
        return response

    def chat(self, system: str, messages: list[dict], tools: list[dict], *, force_tool: str | None = None,
             model_override: str | None = None) -> dict:
        return self._next(system, messages, tools, model_override)


class StreamingScriptedClient(ScriptedClient):
    """Also streams: each text block of a response arrives as one ``text_delta``."""

    def __init__(self, responses: list, *, cost_per_call: float = 0.0):
        super().__init__(responses, cost_per_call=cost_per_call)
        self.streamed_calls = 0

    def chat_stream(self, system: str, messages: list[dict], tools: list[dict], model_override: str | None = None):
        response = self._next(system, messages, tools, model_override)
        self.streamed_calls += 1
        for block in response["content"]:
            if block.type == "text":
                yield {"type": "text_delta", "text": block.text}
        yield {"type": "message_done", **response}


@pytest.fixture
def scripted_client():
    """Factory: ``scripted_client(responses, stream=False, cost_per_call=0.0)``."""
    def make(responses: list, *, stream: bool = False, cost_per_call: float = 0.0) -> ScriptedClient:
        cls = StreamingScriptedClient if stream else ScriptedClient
        return cls(responses, cost_per_call=cost_per_call)
    return make


# -- a real lab workspace ---------------------------------------------------------------------------

SUBMODULE_PARTS = tuple(FRONTEND_SUBMODULE_PATH.split("/"))


def git(cwd: str, *args: str) -> str:
    """``git <args>`` in ``cwd`` as the test's own user (the lab's commits are authored by a login)."""
    result = subprocess.run(
        ["git", "-c", "user.name=setup", "-c", "user.email=setup@example.com", "-c", "core.autocrlf=false",
         "-c", "protocol.file.allow=always", *args],
        cwd=cwd, check=True, capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _init_repo(path: str, branch: str, files: dict[str, str]) -> None:
    os.makedirs(path)
    git(path, "init", "-q", "-b", branch)
    for name, text in files.items():
        write(os.path.join(path, *name.split("/")), text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")


@dataclass
class LabLayout:
    """A lab's disk layout under one temporary folder (see the module docstring)."""

    #: the folder holding everything below
    home: str
    #: repository key -> its local "upstream" repository (a working tree on the base branch)
    upstream: dict[str, str]
    #: BESSER_LAB_WORKSPACE_ROOT
    root: str
    #: BESSER_LAB_GIT_ROOT
    git_root: str
    #: BESSER_LAB_STATE_DIR
    state_dir: str

    def specs(self) -> tuple[RepoSpec, ...]:
        """The base repositories, as the backend builds them from its settings."""
        return repo_specs_from_settings(self.root, self.git_root, DEFAULT_UPSTREAMS)

    def workspace(self) -> LabWorkspace:
        """A new :class:`LabWorkspace` on this layout."""
        return LabWorkspace(self.root, self.specs(), state_dir=self.state_dir, git_root=self.git_root)

    def base_path(self, repo: str, *parts: str) -> str:
        """A path in ``repo``'s base clone."""
        return os.path.join(next(s.path for s in self.specs() if s.key == repo), *parts)

    def git_dir(self, repo: str) -> str:
        """``repo``'s git directory in the git root."""
        return next(s.git_dir for s in self.specs() if s.key == repo)

    def git_in(self, repo: str, *args: str) -> str:
        """``git <args>`` on ``repo``'s git directory (its shared refs), whatever tree is checked out."""
        return git(self.home, f"--git-dir={self.git_dir(repo)}", *args)

    def push_upstream(self, repo: str, path: str, text: str, message: str = "upstream work") -> str:
        """Commit ``path`` = ``text`` on ``repo``'s upstream base branch; returns the new commit."""
        write(os.path.join(self.upstream[repo], *path.split("/")), text)
        git(self.upstream[repo], "add", "-A")
        git(self.upstream[repo], "commit", "-q", "-m", message)
        return git(self.upstream[repo], "rev-parse", "HEAD")


def build_lab_layout(home: str) -> LabLayout:
    """Make the upstreams and the base clones under ``home``, as the setup script does on a lab host."""
    upstream = {
        REPO_FRONTEND: os.path.join(home, "upstream", "editor"),
        REPO_BESSER: os.path.join(home, "upstream", "besser"),
        REPO_MODELING_AGENT: os.path.join(home, "upstream", "modeling-agent"),
    }
    _init_repo(upstream[REPO_FRONTEND], "develop", {"packages/webapp/src/app.ts": "export const app = 1;\n"})
    _init_repo(upstream[REPO_BESSER], "development", {
        "README.md": "BESSER\n",
        "docs/old.rst": "Old page\n",
        "besser/old_name.py": "VALUE = 1\n",
    })
    git(upstream[REPO_BESSER], "submodule", "add", "-q", upstream[REPO_FRONTEND], FRONTEND_SUBMODULE_PATH)
    git(upstream[REPO_BESSER], "commit", "-q", "-m", "add the editor")
    _init_repo(upstream[REPO_MODELING_AGENT], "develop", {"agent.py": "AGENT = 1\n"})

    layout = LabLayout(home=home, upstream=upstream, root=os.path.join(home, "work"),
                       git_root=os.path.join(home, "git"), state_dir=os.path.join(home, "state"))
    os.makedirs(layout.state_dir)
    specs = {spec.key: spec for spec in layout.specs()}
    os.makedirs(layout.git_root)
    for key in (REPO_BESSER, REPO_MODELING_AGENT):
        git(home, "clone", "-q", f"--separate-git-dir={specs[key].git_dir}", "-b", specs[key].base_branch,
            upstream[key], specs[key].path)
    # The editor as a submodule of BESSER's base: git puts its git directory under BESSER's, in the git root.
    git(specs[REPO_BESSER].path, "submodule", "update", "--init", "-q", FRONTEND_SUBMODULE_PATH)
    git(specs[REPO_FRONTEND].path, "checkout", "-q", specs[REPO_FRONTEND].base_branch)
    assert os.path.isdir(specs[REPO_FRONTEND].git_dir)
    return layout


def _relocate(layout: LabLayout, home: str) -> LabLayout:
    """``layout`` copied to ``home``: the base trees' ``.git`` files name the copy's git directories."""
    def move(path: str) -> str:
        return os.path.join(home, os.path.relpath(path, layout.home))

    moved = LabLayout(home=home, upstream={key: move(path) for key, path in layout.upstream.items()},
                      root=move(layout.root), git_root=move(layout.git_root), state_dir=move(layout.state_dir))
    shutil.copytree(layout.home, home)
    for spec in moved.specs():
        os.remove(os.path.join(spec.path, ".git"))  # Git for Windows may mark it hidden, which open(..., "w") refuses
        write(os.path.join(spec.path, ".git"), f"gitdir: {spec.git_dir}\n")
    return moved


@pytest.fixture(scope="session")
def lab_layout_template(tmp_path_factory) -> LabLayout:
    """One layout built per test session; ``lab_layout`` copies it."""
    return build_lab_layout(str(tmp_path_factory.mktemp("lab_layout") / "lab"))


@pytest.fixture
def lab_layout(lab_layout_template, tmp_path, monkeypatch) -> LabLayout:
    """A fresh lab layout under ``tmp_path``, its upstreams reachable as ``workspace.upstream_url``."""
    layout = _relocate(lab_layout_template, str(tmp_path / "lab"))
    monkeypatch.setattr(workspace_module, "upstream_url", lambda spec: layout.upstream[spec.key])
    return layout


@pytest.fixture
def lab_workspace(lab_layout) -> LabWorkspace:
    """A :class:`LabWorkspace` on ``lab_layout``, ready (``ensure_ready`` passed)."""
    workspace = lab_layout.workspace()
    workspace.ensure_ready()
    return workspace
