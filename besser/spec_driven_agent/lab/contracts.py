"""Shared data types of the lab agent: the agent that changes BESSER's own repositories.

Every lab module exchanges these types, so they live in one place and import
nothing from the rest of the package.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: Repository keys, in the order pull requests are opened (the editor first:
#: BESSER's submodule pointer and API schema check depend on its commit).
REPO_FRONTEND = "frontend"
REPO_BESSER = "besser"
REPO_MODELING_AGENT = "modeling_agent"
REPO_ORDER = (REPO_FRONTEND, REPO_BESSER, REPO_MODELING_AGENT)

#: Path of the editor submodule inside the BESSER checkout.
FRONTEND_SUBMODULE_PATH = "besser/utilities/web_modeling_editor/frontend"

#: Branch prefix of every change the lab makes.
LAB_BRANCH_PREFIX = "lab/"

#: Folder under the workspace root holding the changes' state and notes.
LAB_STATE_DIRNAME = ".lab"


class ChangeKind(str, enum.Enum):
    """What a change adds; selects its checklist and its guidance."""

    GENERATOR = "generator"
    METAMODEL = "metamodel"
    DIAGRAM_TYPE = "diagram_type"
    EDITOR_ELEMENT = "editor_element"
    BACKEND = "backend"
    AGENT = "agent"
    BUGFIX = "bugfix"
    OTHER = "other"


@dataclass(frozen=True)
class RepoSpec:
    """One repository of the lab workspace."""

    key: str
    path: str
    base_branch: str
    github_owner: str
    github_name: str


@dataclass(frozen=True)
class FileChange:
    path: str
    #: "added", "modified", "deleted" or "renamed"
    status: str


@dataclass(frozen=True)
class RepoChange:
    """What a change touched in one repository, against its base branch."""

    repo: str
    branch: str
    base: str
    files: tuple[FileChange, ...] = ()


@dataclass
class LabChange:
    """One change in progress: a branch per repository plus its notes."""

    id: str
    slug: str
    title: str
    user: str
    kind: ChangeKind = ChangeKind.OTHER
    #: "active" (checked out) or "parked"
    status: str = "active"
    created_at: float = 0.0
    #: repository key -> branch name, for the repositories the change has a branch in
    branches: dict[str, str] = field(default_factory=dict)
    #: True once the user confirmed the plan, so later messages build
    plan_approved: bool = False


@dataclass(frozen=True)
class LabLimits:
    max_turns: int = 150
    max_cost_usd: float = 5.0
    max_runtime_seconds: float = 2400.0


def _no_progress(turn: int, tool: str, status: str, detail: Optional[str]) -> None:
    return None


def _no_text(delta: str) -> None:
    return None


def _no_phase(phase: str, message: str) -> None:
    return None


def _always() -> bool:
    return True


@dataclass
class LabCallbacks:
    """How a run reports progress. Called from the worker thread."""

    #: (turn, tool name, "executing" | "done" | "error", short detail)
    on_progress: Callable[[int, str, str, Optional[str]], None] = _no_progress
    on_text: Callable[[str], None] = _no_text
    #: (phase, message); phases: "plan", "build", "check"
    on_phase: Callable[[str, str], None] = _no_phase
    #: False once the user stopped the run
    should_continue: Callable[[], bool] = _always


@dataclass
class LoopResult:
    #: "completed", "cancelled", "timeout", "cost_cap", "max_turns" or "api_error"
    stop_reason: str
    detail: str = ""
    turns: int = 0
    cost_usd: float = 0.0
    final_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtraTool:
    """A tool the lab adds to the file and shell tools."""

    #: {"name", "description", "input_schema"}
    schema: dict[str, Any]
    #: receives the tool arguments, returns the text the model reads
    handler: Callable[[dict[str, Any]], str]
    #: False for a tool that changes nothing, so the plan step may use it
    writes: bool = True


@dataclass(frozen=True)
class CheckSpec:
    """One of a repository's own checks, as CI runs it."""

    id: str
    repo: str
    title: str
    argv: tuple[str, ...]
    #: working directory, relative to the repository
    cwd: str = "."
    timeout_seconds: float = 900.0
    #: "tests", "lint", "types", "docs", "i18n" or "schema"
    kind: str = "tests"


@dataclass(frozen=True)
class CheckResult:
    spec_id: str
    repo: str
    title: str
    passed: bool
    skipped: bool = False
    summary: str = ""
    output_tail: str = ""
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    text: str
    #: True, False, or None when the rule cannot be judged from the files
    done: Optional[bool]
    evidence: str = ""


@dataclass
class Plan:
    kind: ChangeKind
    title: str
    summary: str
    steps: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    #: A class diagram of the extension, or None. See ``prompts.MODEL_SPEC_SCHEMA``.
    model_spec: Optional[dict[str, Any]] = None


@dataclass
class BuildOutcome:
    loop: LoopResult
    changes: list[RepoChange] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    checklist: list[ChecklistItem] = field(default_factory=list)
    #: "What changed", per repository, for the user
    summary: str = ""


@dataclass
class SessionOutcome:
    """Result of one user message."""

    #: "plan" or "build"
    step: str
    change: Optional[LabChange] = None
    plan: Optional[Plan] = None
    build: Optional[BuildOutcome] = None
    cost_usd: float = 0.0
    #: stop reason of the last loop that ran
    stop_reason: str = "completed"
    #: the model's last text when the plan step ended without a plan (for example a question to the user)
    final_text: str = ""
