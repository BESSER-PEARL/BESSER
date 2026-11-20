"""One user message to the lab agent about one change: plan it, or build it and report what changed.

The request names the change (a draft, or an open change); the session never picks one. A message goes
through one step:

- **Plan**: a read-only tool loop that ends with ``submit_plan``. A draft's plan reads the base clones
  (``<root>/base``) in a sandbox that cannot write them; its first submitted plan opens the change
  (``LabWorkspace.open_change``: a branch and worktree per repository). A plan that touches no repository
  keeps the draft a draft. A plan on an open change revises it and keeps its branches. Issues named in the
  request are read from GitHub before the loop and recorded on the change. The plan loop may use its own
  model (``plan_model``).
- **Build**: a tool loop with the write and shell tools in the change's own trees, confined by the
  change's sandbox; then the repositories' own checks, each repository's environment made first. Failing
  checks go back to the model, in the same conversation, for at most ``max_check_rounds`` rounds. The
  change kind's checklist is judged, and what was asked, done and reported (``submit_build_report``) is
  appended to the notes. A stop ends a running check; a stopped build reports the checks that ran, those
  of the model's last ``run_checks`` call when the checks after the loop did not start.

Every step starts with an orientation in its first message (the ``CLAUDE.md`` of the repositories in
scope and a map of each, from the base commits) and a knowledge section in its system prompt (skills,
team notes, GitHub tools). The tools work at the change's root, the folder that holds its trees (the paths
the system prompt names are relative to it).
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

from besser.spec_driven_agent.lab import checks as lab_checks
from besser.spec_driven_agent.lab.checks import checks_for, format_results, run_checks, run_checks_tool
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    BuildOutcome,
    BuildReport,
    ChangeKind,
    CheckResult,
    CheckSpec,
    ExtraTool,
    IssueRef,
    LabCallbacks,
    LabChange,
    LabLimits,
    LoopResult,
    NoteDraft,
    Plan,
    RepoChange,
    RepoSpec,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.environments import Environments, LabEnvironmentError
from besser.spec_driven_agent.lab.github_tools import (
    DEFAULT_COMMENTS,
    GitHubReader,
    GitHubReadError,
    format_issue,
    git_history_tool,
    github_tools,
    parse_issue_refs,
)
from besser.spec_driven_agent.lab.kinds import evaluate_checklist, guidance
from besser.spec_driven_agent.lab.knowledge import format_skill_index, orientation, skill_index
from besser.spec_driven_agent.lab.loop import run_lab_loop
from besser.spec_driven_agent.lab.memory import (
    LabMemory,
    LabMemoryUnavailable,
    propose_lab_note_tool,
    read_lab_note_tool,
)
from besser.spec_driven_agent.lab.prompts import (
    build_system_prompt,
    describe_outcome,
    first_user_message,
    format_build_report,
    format_model_spec,
    format_plan,
    knowledge_section,
    plan_repos_from_notes,
    submit_build_report_tool,
    submit_plan_tool,
)
from besser.spec_driven_agent.lab.sandbox_policy import SandboxPolicy, lab_sandbox_mounts
from besser.spec_driven_agent.lab.scaffolds import scaffold_generator_tool
from besser.spec_driven_agent.lab.tooling import LabToolExecutor
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError

logger = logging.getLogger(__name__)

ACTIONS = ("auto", "plan", "build")

# Issues read before a plan: references in a message beyond these are left to the github_issue tool.
_MAX_PREFETCHED_ISSUES = 3

#: Sent once when the plan step ends with text instead of a ``submit_plan`` call.
PLAN_REMINDER = (
    "You ended the plan step without calling submit_plan, so no plan reached the user. Call submit_plan now. "
    "Anything the user has to decide goes in its questions."
)

#: Opens the message that hands failed checks back to the model.
CHECKS_FAILED = (
    "The lab ran the checks of the files changed so far, and some failed. Fix what fails and run run_checks "
    "again. Then call submit_build_report again if what the change does, its deviations or what is not done "
    "changed.\n\n"
)

# Prefixed to a check's output when its repository's node_modules were installed in the change's own tree.
_NPM_CI_NOTE = ("Note: npm ci ran in this change's tree: its package-lock.json differs from the base's, so its "
                "node_modules are not linked from the shared snapshot.\n")


@dataclass
class _Run:
    """What one message's steps share."""

    started: float
    #: the GitHub read tools, built once per run (their cache and call cap are the run's)
    github: list[ExtraTool] = field(default_factory=list)


class LabSession:
    """Handles lab messages, each about the change the request names, in ``workspace``.

    ``client`` is the LLM client (``chat`` / ``chat_stream`` / ``usage``). All loops of one message share
    ``limits``: the cost cap counts the client's cumulative cost, the runtime cap counts from the start of
    :meth:`handle`. ``environments`` makes each repository's environment before its checks.
    ``sandbox_policy`` gives the mounts of a change's sandbox (default: :func:`lab_sandbox_mounts` hiding
    the workspace root and the state folder). ``github_reader`` lets the agent read GitHub; ``memory`` is
    the team's notes; ``plan_model`` the model of the plan loop (default: the client's).
    """

    def __init__(self, workspace: LabWorkspace, *, client, environments: Environments,
                 limits: LabLimits = LabLimits(), callbacks: LabCallbacks | None = None, allow_shell: bool = True,
                 max_check_rounds: int = 3, sandbox_policy: SandboxPolicy | None = None,
                 github_reader: GitHubReader | None = None, memory: LabMemory | None = None,
                 plan_model: str | None = None):
        self.workspace = workspace
        self.client = client
        self.environments = environments
        self.limits = limits
        self.callbacks = callbacks or LabCallbacks()
        self.allow_shell = allow_shell
        self.max_check_rounds = max_check_rounds
        self.sandbox_policy: SandboxPolicy = sandbox_policy or functools.partial(
            lab_sandbox_mounts, workspace_root=workspace.root, git_root=workspace.git_root,
            hidden=(workspace.state_dir,))
        self.github_reader = github_reader
        self.memory = memory
        self.plan_model = plan_model

    def handle(self, message: str, *, change_id: str, login: str, action: str = "auto",
               model_spec: dict | None = None, issues: Sequence[IssueRef] = ()) -> SessionOutcome:
        """Run the step ``action`` asks for on ``message`` for the change ``change_id`` and return what it produced.

        ``auto`` plans a draft or an open change whose plan is not approved yet (a reply to a plan
        revises it), and builds an approved one. ``plan`` plans (revising an open change's plan),
        ``build`` builds and approves the plan. ``login`` is who sent the message. ``model_spec`` is the
        class diagram of this change in the user's editor, shaped like ``prompts.MODEL_SPEC_SCHEMA``.
        ``issues`` are GitHub issues the request starts from, read with those the message names.
        Raises ``ValueError`` for an unknown action and :class:`LabWorkspaceError` for an unknown or
        closed change, a build of a draft, or base clones that are not ready. ``InvalidApiKeyError``
        propagates.
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown lab action {action!r}: expected one of {', '.join(ACTIONS)}.")
        change = self.workspace.get_change(change_id)
        if change.status == "closed":
            raise LabWorkspaceError(f"change {change.id} is closed: start a new change")
        planning = action == "plan" or (action == "auto" and (change.status == "draft" or not change.plan_approved))
        if not planning and change.status == "draft":
            raise LabWorkspaceError(f"change {change.id} is a draft: it has no plan yet. Describe the change "
                                    "first, so the lab agent can plan it.")
        cost_before = self.client.usage.estimated_cost
        run = _Run(started=time.monotonic())
        if self.github_reader is not None:
            run.github = github_tools(self.github_reader, self.workspace.base_repos)
        model_spec_text = format_model_spec(model_spec) if model_spec else None
        if planning:
            outcome = self._plan(run, message, login, change, model_spec_text, issues)
        else:
            outcome = self._build(run, message, login, change, model_spec_text)
        outcome.cost_usd = self.client.usage.estimated_cost - cost_before
        return outcome

    # -- plan --------------------------------------------------------------------------------------

    def _plan(self, run: _Run, message: str, login: str, change: LabChange, model_spec_text: Optional[str],
              issues: Sequence[IssueRef]) -> SessionOutcome:
        draft = change.status == "draft"
        if draft:
            # Fail before the model is paid for, not when the change's branches are made.
            self.workspace.ensure_ready()
        repos = self.workspace.change_repos(change)
        # Read-only: a draft's sandbox holds the base trees, which every change starts from.
        mounts = self.sandbox_policy(repos, writable=False)
        issue_texts = self._prefetch_issues(change, message, issues)
        self.callbacks.on_phase("plan", "Planning the change"
                                + (f" (model {self.plan_model})" if self.plan_model else ""))
        plans: list[Plan] = []
        kind = None if draft else change.kind
        system = build_system_prompt(mode="plan", repos=repos, kind=kind, notes=self.workspace.notes(change),
                                     knowledge=self._knowledge(run, repos, kind))
        orientation_text = self._orientation(change, inline=[REPO_BESSER])
        first = first_user_message(message=message, mode="plan", plan=None, model_spec_text=model_spec_text,
                                   orientation=orientation_text, issues=issue_texts)
        messages = [{"role": "user", "content": first}]
        tools = [submit_plan_tool(plans.append), *self._reading_tools(run, change, repos)]
        executor = LabToolExecutor(self.workspace.change_root(change), read_only=True, extra_tools=tools,
                                   time_left=self._time_left(run.started), sandbox_mounts=mounts)
        try:
            loop = self._loop(executor, system, messages, run.started, finish_tool="submit_plan",
                              model=self.plan_model)
            if loop.stop_reason == "completed" and not plans:
                _add_user_message(messages, PLAN_REMINDER)
                loop = self._loop(executor, system, messages, run.started, finish_tool="submit_plan",
                                  model=self.plan_model)
        finally:
            executor.close()
        plan = plans[-1] if plans else None
        if plan is None or plan.changes_nothing:
            # A question back, or the answer that nothing needs to change: the notes carry it to the next message.
            if plan is not None:
                reply = "\n".join([plan.summary, *(f"- {question}" for question in plan.questions)])
            else:
                reply = loop.final_text or f"(no reply: the step stopped, {loop.stop_reason})"
            self.workspace.append_note(change, f"## Request by {login} without a plan\n{login}: {message.strip()}\n"
                                               f"The lab agent: {reply}")
            return SessionOutcome(step="plan", change=change, plan=plan, stop_reason=loop.stop_reason,
                                  final_text=loop.final_text)

        heading = f"Request by {login}" if draft else f"Request by {login} (plan revised)"
        # Noted before the change opens: a failure to open it keeps the paid-for plan with the draft.
        self.workspace.append_note(change, f"## {heading}\n{message.strip()}\n\n## Plan\n{format_plan(plan)}")
        if draft:
            change = self.workspace.open_change(change, title=plan.title, kind=plan.kind)
        else:
            # The branches keep the first title's name; the pull request takes the new one.
            change.title, change.kind, change.plan_approved = plan.title, plan.kind, False
            self.workspace.save_change(change)
        return SessionOutcome(step="plan", change=change, plan=plan, stop_reason=loop.stop_reason)

    def _prefetch_issues(self, change: LabChange, message: str, issues: Sequence[IssueRef]) -> list[str]:
        """The issues of the request (``issues`` and those ``message`` names, at most three) as the plan's first
        message shows them; each issue read is recorded on the change. A read that fails is said there and logged."""
        refs: list[IssueRef] = []
        for ref in [*issues, *parse_issue_refs(message, self.workspace.base_repos, limit=_MAX_PREFETCHED_ISSUES)]:
            if ref not in refs and len(refs) < _MAX_PREFETCHED_ISSUES:
                refs.append(ref)
        texts = []
        known = {(entry.get("repo"), entry.get("number")) for entry in change.issues}
        for ref in refs:
            if self.github_reader is None:
                logger.warning("BESSER Lab: issue %s#%d of change %s not read: no GitHub reader in this run",
                               ref.repo, ref.number, change.id)
                texts.append(f"Issue {ref.repo}#{ref.number} is named, but GitHub cannot be read in this run.")
                continue
            try:
                data = self.github_reader.issue(ref.repo, ref.number, DEFAULT_COMMENTS)
            except GitHubReadError as exc:
                logger.warning("BESSER Lab: issue %s#%d of change %s not read: %s", ref.repo, ref.number, change.id,
                               exc)
                texts.append(f"Issue {ref.repo}#{ref.number} could not be read: {exc}")
                continue
            texts.append(format_issue(data))
            if data.get("kind") == "issue" and (ref.repo, ref.number) not in known:
                change.issues.append({"repo": ref.repo, "number": ref.number, "title": str(data.get("title") or ""),
                                      "closes": False})
                known.add((ref.repo, ref.number))
        if refs:
            self.workspace.save_change(change)
        return texts

    # -- build -------------------------------------------------------------------------------------

    def _build(self, run: _Run, message: str, login: str, change: LabChange,
               model_spec_text: Optional[str]) -> SessionOutcome:
        if not change.plan_approved:
            change.plan_approved = True
            self.workspace.save_change(change)
        repos = self.workspace.change_repos(change)
        mounts = self.sandbox_policy(repos, writable=True)
        self.callbacks.on_phase("build", f"Building: {change.title}")
        runner = _CheckRunner(
            check=lab_checks.confined_runner(mounts, should_continue=self.callbacks.should_continue),
            install=lab_checks.confined_command_runner(mounts, should_continue=self.callbacks.should_continue),
            environments=self.environments, repos=repos, limits=self.limits,
            time_left=self._time_left(run.started))
        tool_checks: list[list[CheckResult]] = []  # each run_checks call's results
        reports: list[BuildReport] = []
        proposals: list[NoteDraft] = []
        extra_tools = [run_checks_tool(lambda: runner.new_batch(self.workspace.repo_changes(change)), repos,
                                       runner=runner, callbacks=self.callbacks, on_results=tool_checks.append),
                       submit_build_report_tool(reports.append), propose_lab_note_tool(proposals.append)]
        if change.kind is ChangeKind.GENERATOR and REPO_BESSER in repos:
            extra_tools.append(scaffold_generator_tool(repos[REPO_BESSER].path))
        extra_tools += self._reading_tools(run, change, repos)
        system = build_system_prompt(mode="build", repos=repos, kind=change.kind,
                                     notes=self.workspace.notes(change), guidance=guidance(change.kind),
                                     knowledge=self._knowledge(run, repos, change.kind))
        plan_repos = plan_repos_from_notes(self.workspace.notes(change))
        if plan_repos is None:
            logger.warning("BESSER Lab: change %s has no plan in its notes; the build inlines BESSER's CLAUDE.md",
                           change.id)
        first = first_user_message(message=message, mode="build", plan=None, model_spec_text=model_spec_text,
                                   orientation=self._orientation(change, inline=plan_repos or [REPO_BESSER]))
        messages = [{"role": "user", "content": first}]
        executor = LabToolExecutor(self.workspace.change_root(change), allow_shell=self.allow_shell,
                                   extra_tools=extra_tools, time_left=self._time_left(run.started),
                                   sandbox_mounts=mounts)
        try:
            loop = self._loop(executor, system, messages, run.started)
            changes, checks = self._run_checks(change, repos, runner)
            rounds = 0
            while True:
                failed = [c for c in checks if not c.passed and not c.skipped]
                # A loop that stopped on a cap or a cancel cannot go on; only a finished one gets a fix round.
                if not failed or loop.stop_reason != "completed" or rounds >= self.max_check_rounds:
                    break
                rounds += 1
                self.callbacks.on_phase("build", f"Fixing failed checks (round {rounds} of {self.max_check_rounds})")
                _add_user_message(messages, CHECKS_FAILED + format_results(checks))
                then = self._loop(executor, system, messages, run.started)
                loop = _combined(loop, then)
                if not then.tool_calls:
                    break  # nothing was changed: the results stand
                changes, checks = self._run_checks(change, repos, runner)
        finally:
            executor.close()
        if not self.callbacks.should_continue():
            if loop.stop_reason == "completed":
                loop.stop_reason, loop.detail = "cancelled", "stopped during the checks"
            if not checks and tool_checks:
                # Stopped before the checks after the loop: the model's last run_checks call is what ran.
                checks = tool_checks[-1]
        report = reports[-1] if reports else None
        if report is None:
            logger.warning("BESSER Lab: the build of change %s ended without submit_build_report", change.id)
        checklist = evaluate_checklist(change.kind, changes, repos)
        summary = describe_outcome(changes, checks, checklist, loop.final_text)
        note = f"## Build request by {login}\n{message.strip()}\n\n## Build result ({loop.stop_reason})\n{summary}"
        if report is not None:
            note += f"\n\n## Build report\n{format_build_report(report)}"
        self.workspace.append_note(change, note)
        build = BuildOutcome(loop=loop, changes=changes, checks=checks, checklist=checklist, summary=summary,
                             report=report)
        return SessionOutcome(step="build", change=change, build=build, stop_reason=loop.stop_reason,
                              note_proposals=proposals)

    # -- helpers -----------------------------------------------------------------------------------

    def _loop(self, executor: LabToolExecutor, system: str, messages: list[dict], started: float,
              finish_tool: str | None = None, model: str | None = None) -> LoopResult:
        return run_lab_loop(client=self.client, executor=executor, system=system, messages=messages,
                            limits=self.limits, callbacks=self.callbacks, finish_tool=finish_tool,
                            started_at=started, model=model)

    def _time_left(self, started: float) -> Callable[[], float]:
        """Seconds left of the message's runtime budget, which caps every shell command and check."""
        return lambda: self.limits.max_runtime_seconds - (time.monotonic() - started)

    def _run_checks(self, change: LabChange, repos: Mapping[str, RepoSpec],
                    runner: _CheckRunner) -> tuple[list[RepoChange], list[CheckResult]]:
        """The change's files and the results of their checks, each check cut at the time the run has left."""
        changes = runner.new_batch(self.workspace.repo_changes(change))
        return changes, run_checks(checks_for(changes, repos), repos, runner=runner, callbacks=self.callbacks)

    def _reading_tools(self, run: _Run, change: LabChange, repos: Mapping[str, RepoSpec]) -> list[ExtraTool]:
        """The read-only tools of both steps: GitHub (when it can be read), git history, team notes."""
        run_git = self.workspace.run_git if change.status == "draft" else functools.partial(
            self.workspace.run_git, change=change)
        tools = [*run.github, git_history_tool(run_git, repos)]
        if self.memory is not None and self.memory.enabled:
            tools.append(read_lab_note_tool(self.memory))
        return tools

    def _knowledge(self, run: _Run, repos: Mapping[str, RepoSpec], kind: ChangeKind | None) -> str:
        """The system prompt's knowledge section: skills of the trees, team notes, which GitHub tools exist."""
        notes, available = "", True
        if self.memory is not None:
            available = self.memory.enabled
            if available:
                try:
                    notes = self.memory.index_text(list(repos), kind)
                except LabMemoryUnavailable as exc:
                    logger.warning("BESSER Lab: team notes left out of the prompt: %s", exc)
                    available = False
        return knowledge_section(skills=format_skill_index(skill_index(repos)), notes=notes,
                                 notes_available=available, github=bool(run.github))

    def _orientation(self, change: LabChange, *, inline: Sequence[str]) -> str:
        """The orientation of the change's first message: the ``CLAUDE.md`` of ``inline`` in full, the others' as
        an index, and every repository's map, from the commits the change started from (a draft: the base clones).

        When it cannot be built, the message says so (and the log), so the agent reads the ``CLAUDE.md`` files.
        """
        try:
            if change.status == "draft":
                shas = {key: self.workspace.run_git(key, "rev-parse", "HEAD").strip()
                        for key in self.workspace.base_repos}
            else:
                shas = {key: base.sha for key, base in change.bases.items()}
            inline = [key for key in inline if key in shas]
            return orientation(self.workspace, inline=inline, indexed=[key for key in shas if key not in inline],
                               base_shas=shas)
        except (LabWorkspaceError, ValueError, OSError) as exc:
            logger.warning("BESSER Lab: no orientation for change %s: %s", change.id, exc)
            return (f"## Orientation\nNot available in this run ({exc}): read the CLAUDE.md of each repository "
                    "you change before changing it.")


class _CheckRunner:
    """The checks' runner of one build: each check cut at the time the run has left (none starts after the cap),
    its repository's environment made first, once per batch of checks.

    An environment that cannot be made fails that repository's checks with the reason, so the model reads it.
    """

    def __init__(self, *, check: lab_checks.Runner, install: lab_checks.CommandRunner, environments: Environments,
                 repos: Mapping[str, RepoSpec], limits: LabLimits, time_left: Callable[[], float]):
        self._check = check
        self._install = install
        self._environments = environments
        self._repos = repos
        self._limits = limits
        self._time_left = time_left
        #: repository -> (error, note) of this batch's environment
        self._ensured: dict[str, tuple[str, str]] = {}

    def new_batch(self, changes: list[RepoChange]) -> list[RepoChange]:
        """Start a batch of checks (on ``changes``, returned as is): environments are checked again."""
        self._ensured = {}
        return changes

    def __call__(self, spec: CheckSpec, cwd: str) -> tuple[int, str]:
        left = self._time_left()
        if left <= 0:
            return lab_checks.TIMED_OUT, f"Not run: the run reached its {self._limits.max_runtime_seconds:.0f} s limit"
        if spec.repo not in self._ensured:
            self._ensured[spec.repo] = self._ensure(spec.repo)
        error, note = self._ensured[spec.repo]
        if error:
            return 1, error
        self._ensured[spec.repo] = ("", "")  # the note goes with the first check only
        code, output = self._check(replace(spec, timeout_seconds=min(spec.timeout_seconds, left)), cwd)
        return code, note + output

    def _ensure(self, repo: str) -> tuple[str, str]:
        started = time.time()
        try:
            stamp = self._environments.ensure(self._repos[repo], runner=self._install)
        except LabEnvironmentError as exc:
            logger.warning("BESSER Lab: the environment of %s in %s could not be made: %s", repo,
                           self._repos[repo].path, exc)
            return f"The environment of {repo} could not be made, so its checks did not run: {exc}", ""
        fresh_install = repo == REPO_FRONTEND and stamp.mode == "installed" and stamp.created_at >= started
        return "", _NPM_CI_NOTE if fresh_install else ""


def _add_user_message(messages: list[dict], text: str) -> None:
    """Continue the conversation with ``text`` from the user.

    A loop that ended on an empty reply leaves the user's tool results last; the
    text joins that message, so roles keep alternating.
    """
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        messages[-1] = {"role": "user", "content": [*blocks, {"type": "text", "text": text}]}
    else:
        messages.append({"role": "user", "content": text})


def _combined(first: LoopResult, then: LoopResult) -> LoopResult:
    """Two loops of one conversation as one: the later one's ending, both ones' work and cost."""
    return LoopResult(stop_reason=then.stop_reason, detail=then.detail, turns=first.turns + then.turns,
                      cost_usd=first.cost_usd + then.cost_usd, final_text=then.final_text or first.final_text,
                      tool_calls=first.tool_calls + then.tool_calls)
