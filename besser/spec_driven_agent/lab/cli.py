"""``python -m besser.spec_driven_agent.lab``: send one message about one change to the lab agent from a terminal.

``--root``, ``--state-dir`` and ``--git-root`` are the lab's folders (``BESSER_LAB_WORKSPACE_ROOT``,
``BESSER_LAB_STATE_DIR``, ``BESSER_LAB_GIT_ROOT``), laid out by ``scripts/lab/setup_workspace.sh`` from the
canonical repositories. Without ``--change`` a new draft is made; its id is the first line of stdout, so a
next message can name it. Progress (phases, tool calls, the model's text) goes to stderr, the result as
text to stdout and, with ``--json``, the whole outcome as JSON to a file. The API key is read from the
provider's usual environment variable (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...); GitHub is not
read. The exit status is 0 when the step completed (a plan step: with a plan), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Optional

from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.lab.contracts import (
    DEFAULT_UPSTREAMS,
    LAB_CACHE_DIRNAME,
    LabCallbacks,
    LabLimits,
    SessionOutcome,
)
from besser.spec_driven_agent.lab.environments import Environments, probe_link_mode
from besser.spec_driven_agent.lab.memory import LabMemory
from besser.spec_driven_agent.lab.prompts import format_build_report, format_plan
from besser.spec_driven_agent.lab.sandbox_policy import LabSandboxPolicyError
from besser.spec_driven_agent.lab.session import ACTIONS, LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError, repo_specs_from_settings
from besser.spec_driven_agent.providers.llm_client import create_llm_client

PROG = "python -m besser.spec_driven_agent.lab"


def build_parser() -> argparse.ArgumentParser:
    """The command line's argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Send one message to the lab agent of BESSER Lab, which plans or builds a change to "
                    "BESSER's own repositories in the change's own working trees.",
    )
    parser.add_argument("message", metavar="MESSAGE", help="the message to the lab agent")
    parser.add_argument("--root", required=True,
                        help="the lab workspace (BESSER_LAB_WORKSPACE_ROOT): base/ and changes/")
    parser.add_argument("--state-dir", required=True, help="the lab's state folder (BESSER_LAB_STATE_DIR)")
    parser.add_argument("--git-root", required=True, help="the folder of the git directories (BESSER_LAB_GIT_ROOT)")
    parser.add_argument("--change", default=None, metavar="ID",
                        help="the change the message is about (default: a new draft, whose id is printed)")
    parser.add_argument("--login", required=True, help="GitHub login of the person sending the message")
    parser.add_argument("--action", choices=ACTIONS, default="auto",
                        help="auto (default): plan until the change's plan is approved, build after; "
                             "build approves the plan")
    parser.add_argument("--provider", default="anthropic", help="LLM provider (default: anthropic)")
    parser.add_argument("--model", default=None, help="model id (default: the provider's default)")
    parser.add_argument("--max-cost", type=float, default=LabLimits().max_cost_usd,
                        help=f"cost cap of the message in USD (default: {LabLimits().max_cost_usd:g})")
    parser.add_argument("--json", metavar="FILE", default=None, help="write the outcome as JSON to FILE")
    return parser


def _stderr_callbacks() -> LabCallbacks:
    def progress(turn: int, tool: str, status: str, detail: Optional[str]) -> None:
        if status == "executing":
            print(f"  [{turn}] {tool}" + (f" {detail}" if detail else ""), file=sys.stderr)
        elif status == "error":
            print(f"  [{turn}] {tool} failed", file=sys.stderr)

    def text(delta: str) -> None:
        sys.stderr.write(delta)
        sys.stderr.flush()

    def phase(name: str, message: str) -> None:
        print(f"\n[{name}] {message}", file=sys.stderr)

    return LabCallbacks(on_progress=progress, on_text=text, on_phase=phase)


def _result_text(outcome: SessionOutcome) -> str:
    if outcome.build is not None:
        parts = [outcome.build.summary]
        if outcome.build.report is not None:
            parts.append("Report:\n" + format_build_report(outcome.build.report))
        parts += [f"Proposed team note: {draft.title} ({draft.kind})" for draft in outcome.note_proposals]
        return "\n\n".join(parts)
    if outcome.plan is not None:
        return format_plan(outcome.plan)
    return outcome.final_text or f"No plan was submitted (stop reason: {outcome.stop_reason})."


def main(argv: Optional[list[str]] = None) -> int:
    """Parse ``argv``, handle the message and report the outcome; returns the exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        client = create_llm_client(args.provider, api_key=None, model=args.model)
    except (ValueError, ImportError) as exc:
        print(f"{PROG}: cannot create the {args.provider} client: {exc}", file=sys.stderr)
        return 1
    try:
        workspace = LabWorkspace(args.root, repo_specs_from_settings(args.root, args.git_root, DEFAULT_UPSTREAMS),
                                 state_dir=args.state_dir, git_root=args.git_root)
        cache = os.path.join(workspace.root, LAB_CACHE_DIRNAME)
        environments = Environments(cache, mode=probe_link_mode(cache))
        change_id = args.change or workspace.create_draft(owner=args.login).id
        print(f"Change {change_id}")
        session = LabSession(workspace, client=client, environments=environments,
                             limits=LabLimits(max_cost_usd=args.max_cost), callbacks=_stderr_callbacks(),
                             memory=LabMemory(workspace.state_dir, workspace.root))
        outcome = session.handle(args.message, change_id=change_id, login=args.login, action=args.action)
    except (LabWorkspaceError, LabSandboxPolicyError, InvalidApiKeyError, OSError) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    print(_result_text(outcome))
    print(f"\nStop reason: {outcome.stop_reason}; cost ${outcome.cost_usd:.4f}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(asdict(outcome), handle, indent=2)
    completed = outcome.stop_reason == "completed" and (outcome.step == "build" or outcome.plan is not None)
    return 0 if completed else 1
