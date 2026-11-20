"""One user message to the lab agent: plan a change, or build it and report what changed.

A message goes through one step. The plan step is a read-only tool loop that ends
with ``submit_plan``; a new change and its branches are created from that plan,
and the plan goes into the change's notes. A plan that touches no repository
(nothing needs to change) creates no change. A plan step that ends without a
change, with such a plan or without any, keeps the exchange in ``pending.json``
of the workspace's state folder, so the user's next message continues it. The
build step is a tool loop with the write and shell tools in the working copy,
followed by the repositories' own checks; failing checks go back to the model, in the same conversation, for at most
``max_check_rounds`` rounds. Then the change kind's checklist is judged and what
was asked and done is appended to the notes. A stop ends a running check; a
stopped build reports the checks that ran, those of the model's last
``run_checks`` call when the checks after the loop did not start.

The tools work at the workspace root, the folder that holds the repositories (the
paths the system prompt names are relative to it). Unless the workspace was given
a state folder outside it, that folder also holds the changes' state under
``.lab/``, so the session writes its own copy of the change's state back after
every build.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from typing import Callable, Optional

from besser.spec_driven_agent.lab import checks as lab_checks
from besser.spec_driven_agent.lab.checks import checks_for, format_results, run_checks, run_checks_tool
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    BuildOutcome,
    ChangeKind,
    CheckResult,
    CheckSpec,
    LabCallbacks,
    LabChange,
    LabLimits,
    LoopResult,
    Plan,
    RepoChange,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.kinds import evaluate_checklist, guidance
from besser.spec_driven_agent.lab.loop import run_lab_loop
from besser.spec_driven_agent.lab.prompts import (
    build_system_prompt,
    describe_outcome,
    first_user_message,
    format_model_spec,
    format_plan,
    submit_plan_tool,
)
from besser.spec_driven_agent.lab.scaffolds import scaffold_generator_tool
from besser.spec_driven_agent.lab.tooling import LabToolExecutor
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError

logger = logging.getLogger(__name__)

ACTIONS = ("auto", "plan", "build")

#: In the workspace's state folder: the exchange of plan steps that ended without a plan and without a change.
PENDING_FILENAME = "pending.json"

#: Sent once when the plan step ends with text instead of a ``submit_plan`` call.
PLAN_REMINDER = (
    "You ended the plan step without calling submit_plan, so no plan reached the user. Call submit_plan now. "
    "Anything the user has to decide goes in its questions."
)

#: Opens the message that hands failed checks back to the model.
CHECKS_FAILED = (
    "The lab ran the checks of the files changed so far, and some failed. Fix what fails and run run_checks "
    "again. Then say again, in a few sentences, what the change does as a whole and what is left open.\n\n"
)


class LabSession:
    """Handles the messages of a lab user, one at a time, in ``workspace``.

    ``client`` is the LLM client (``chat`` / ``chat_stream`` / ``usage``). All loops
    of one message share ``limits``: the cost cap counts the client's cumulative cost,
    the runtime cap counts from the start of :meth:`handle`.
    """

    def __init__(self, workspace: LabWorkspace, *, client, limits: LabLimits = LabLimits(),
                 callbacks: LabCallbacks | None = None, allow_shell: bool = True, max_check_rounds: int = 3):
        self.workspace = workspace
        self.client = client
        self.limits = limits
        self.callbacks = callbacks or LabCallbacks()
        self.allow_shell = allow_shell
        self.max_check_rounds = max_check_rounds

    def handle(self, message: str, *, user: str, action: str = "auto",
               model_spec: dict | None = None) -> SessionOutcome:
        """Run the step ``action`` asks for on ``message`` and return what it produced.

        ``auto`` plans when no change is active or its plan is not approved yet (a reply
        to a plan revises it), and builds otherwise. ``plan`` plans (revising the active
        change's plan), ``build`` builds and approves the plan.
        ``model_spec`` is the class diagram of this change in the user's editor, shaped like
        ``prompts.MODEL_SPEC_SCHEMA``. Raises ``ValueError`` for an unknown action and
        :class:`LabWorkspaceError` for a build without an active change or a working
        copy that is not ready. ``InvalidApiKeyError`` propagates.
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown lab action {action!r}: expected one of {', '.join(ACTIONS)}.")
        started = time.monotonic()
        cost_before = self.client.usage.estimated_cost
        change = self.workspace.active_change()
        model_spec_text = format_model_spec(model_spec) if model_spec else None
        if action == "plan" or (action == "auto" and (change is None or not change.plan_approved)):
            outcome = self._plan(message, user, change, model_spec_text, started)
        elif change is None:
            raise LabWorkspaceError("There is no active change to build. Describe the change first, so the lab "
                                    "agent can plan it, or resume a parked change.")
        else:
            outcome = self._build(message, change, model_spec_text, started)
        outcome.cost_usd = self.client.usage.estimated_cost - cost_before
        return outcome

    # -- plan --------------------------------------------------------------------------------------

    def _plan(self, message: str, user: str, change: Optional[LabChange], model_spec_text: Optional[str],
              started: float) -> SessionOutcome:
        pending = ""
        if change is None:
            # Fail before the model is paid for, not when the change's branches are created.
            self.workspace.ensure_ready()
            self.workspace.ensure_clean()
            pending = self._pending(user)
        self.callbacks.on_phase("plan", "Planning the change")
        plans: list[Plan] = []
        system = build_system_prompt(mode="plan", repos=self.workspace.repos,
                                     kind=change.kind if change else None,
                                     notes=self.workspace.notes(change) if change else "")
        first = first_user_message(message=message, mode="plan", plan=None, model_spec_text=model_spec_text)
        if pending:
            first = f"## The conversation so far (no plan was submitted yet)\n{pending}\n\n{first}"
        messages = [{"role": "user", "content": first}]
        executor = LabToolExecutor(self.workspace.root, read_only=True, extra_tools=[submit_plan_tool(plans.append)],
                                   time_left=self._time_left(started))
        try:
            loop = self._loop(executor, system, messages, started, finish_tool="submit_plan")
            if loop.stop_reason == "completed" and not plans:
                _add_user_message(messages, PLAN_REMINDER)
                loop = self._loop(executor, system, messages, started, finish_tool="submit_plan")
        finally:
            executor.close()
        exchange = f"The user: {message.strip()}"
        plan = plans[-1] if plans else None
        if plan is None or plan.changes_nothing:
            # A question back, or the answer that nothing needs to change: kept for the user's reply.
            if plan is not None:
                reply = "\n".join([plan.summary, *(f"- {question}" for question in plan.questions)])
            else:
                reply = loop.final_text or f"(no reply: the step stopped, {loop.stop_reason})"
            exchange += f"\nThe lab agent: {reply}"
            if change is None:
                self._save_pending(user, f"{pending}\n\n{exchange}".strip())
            else:
                self.workspace.append_note(change, f"## Request without a plan\n{exchange}")
            return SessionOutcome(step="plan", change=change, plan=plan, stop_reason=loop.stop_reason,
                                  final_text=loop.final_text)

        if change is None:
            change = self.workspace.start_change(user=user, title=plan.title, kind=plan.kind)
            heading = "Request"
            if pending:
                self._save_pending(user, "")
                message = f"{pending}\n\n{exchange}"
        else:
            # The branches keep the first title's name; the pull request takes the new one.
            change.title, change.kind, change.plan_approved = plan.title, plan.kind, False
            self.workspace.save_change(change)
            heading = "Request (plan revised)"
        self.workspace.append_note(change, f"## {heading}\n{message.strip()}\n\n## Plan\n{format_plan(plan)}")
        return SessionOutcome(step="plan", change=change, plan=plan, stop_reason=loop.stop_reason)

    # -- build -------------------------------------------------------------------------------------

    def _build(self, message: str, change: LabChange, model_spec_text: Optional[str],
               started: float) -> SessionOutcome:
        repos = self.workspace.repos
        if not change.plan_approved:
            change.plan_approved = True
            self.workspace.save_change(change)
        self.callbacks.on_phase("build", f"Building: {change.title}")
        tool_checks: list[list[CheckResult]] = []  # each run_checks call's results
        extra_tools = [run_checks_tool(lambda: self.workspace.repo_changes(change), repos,
                                       runner=self._bounded_runner(started), callbacks=self.callbacks,
                                       on_results=tool_checks.append)]
        if change.kind is ChangeKind.GENERATOR and REPO_BESSER in repos:
            extra_tools.append(scaffold_generator_tool(repos[REPO_BESSER].path))
        system = build_system_prompt(mode="build", repos=repos, kind=change.kind,
                                     notes=self.workspace.notes(change), guidance=guidance(change.kind))
        messages = [{"role": "user", "content": first_user_message(message=message, mode="build", plan=None,
                                                                    model_spec_text=model_spec_text)}]
        executor = LabToolExecutor(self.workspace.root, allow_shell=self.allow_shell, extra_tools=extra_tools,
                                   time_left=self._time_left(started))
        try:
            loop = self._loop(executor, system, messages, started)
            changes, checks = self._run_checks(change, started)
            rounds = 0
            while True:
                failed = [c for c in checks if not c.passed and not c.skipped]
                # A loop that stopped on a cap or a cancel cannot go on; only a finished one gets a fix round.
                if not failed or loop.stop_reason != "completed" or rounds >= self.max_check_rounds:
                    break
                rounds += 1
                self.callbacks.on_phase("build", f"Fixing failed checks (round {rounds} of {self.max_check_rounds})")
                _add_user_message(messages, CHECKS_FAILED + format_results(checks))
                then = self._loop(executor, system, messages, started)
                loop = _combined(loop, then)
                if not then.tool_calls:
                    break  # nothing was changed: the results stand
                changes, checks = self._run_checks(change, started)
        finally:
            executor.close()
            # The tools and the checks can write under <root>/.lab/: the session's copy of the change is kept.
            self.workspace.save_change(change)
        if not self.callbacks.should_continue():
            if loop.stop_reason == "completed":
                loop.stop_reason, loop.detail = "cancelled", "stopped during the checks"
            if not checks and tool_checks:
                # Stopped before the checks after the loop: the model's last run_checks call is what ran.
                checks = tool_checks[-1]
        checklist = evaluate_checklist(change.kind, changes, repos)
        summary = describe_outcome(changes, checks, checklist, loop.final_text)
        self.workspace.append_note(change, f"## Build request\n{message.strip()}\n\n"
                                           f"## Build result ({loop.stop_reason})\n{summary}")
        build = BuildOutcome(loop=loop, changes=changes, checks=checks, checklist=checklist, summary=summary)
        return SessionOutcome(step="build", change=change, build=build, stop_reason=loop.stop_reason)

    # -- helpers -----------------------------------------------------------------------------------

    def _loop(self, executor: LabToolExecutor, system: str, messages: list[dict], started: float,
              finish_tool: str | None = None) -> LoopResult:
        return run_lab_loop(client=self.client, executor=executor, system=system, messages=messages,
                            limits=self.limits, callbacks=self.callbacks, finish_tool=finish_tool,
                            started_at=started)

    def _time_left(self, started: float) -> Callable[[], float]:
        """Seconds left of the message's runtime budget, which caps every shell command and check."""
        return lambda: self.limits.max_runtime_seconds - (time.monotonic() - started)

    def _bounded_runner(self, started: float) -> lab_checks.Runner:
        """The checks' runner with each check cut at the time the run has left and ended by a stop;
        none starts after the cap."""
        time_left = self._time_left(started)

        def runner(spec: CheckSpec, cwd: str) -> tuple[int, str]:
            left = time_left()
            if left <= 0:
                limit = self.limits.max_runtime_seconds
                return lab_checks.TIMED_OUT, f"Not run: the run reached its {limit:.0f} s limit"
            return lab_checks.default_runner(replace(spec, timeout_seconds=min(spec.timeout_seconds, left)), cwd,
                                             should_continue=self.callbacks.should_continue)

        return runner

    def _run_checks(self, change: LabChange, started: float) -> tuple[list[RepoChange], list[CheckResult]]:
        """The change's files and the results of their checks, each check cut at the time the run has left."""
        changes = self.workspace.repo_changes(change)
        repos = self.workspace.repos
        return changes, run_checks(checks_for(changes, repos), repos, runner=self._bounded_runner(started),
                                   callbacks=self.callbacks)

    def _pending_path(self) -> str:
        return os.path.join(self.workspace.state_dir, PENDING_FILENAME)

    def _pending(self, user: str) -> str:
        """The exchange of ``user``'s plan steps that ended without a plan, or ""."""
        try:
            with open(self._pending_path(), encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return ""
        except (OSError, ValueError) as exc:
            logger.warning("BESSER Lab ignores its unreadable pending request: %s", exc)
            return ""
        return data.get("text", "") if isinstance(data, dict) and data.get("user") == user else ""

    def _save_pending(self, user: str, text: str) -> None:
        """Keep ``text`` for ``user``'s next plan step; "" forgets it."""
        path = self._pending_path()
        if not text:
            os.remove(path)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"user": user, "text": text}, handle, indent=2)


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
