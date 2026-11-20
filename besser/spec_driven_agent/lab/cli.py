"""``python -m besser.spec_driven_agent.lab``: send one message to the lab agent from a terminal.

Progress (phases, tool calls, the model's text) goes to stderr, the result as text
to stdout and, with ``--json``, the whole outcome as JSON to a file. The API key is
read from the provider's usual environment variable (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ...). The exit status is 0 when the step completed (a plan step:
with a plan), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Optional

from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.lab.contracts import LabCallbacks, LabLimits, SessionOutcome
from besser.spec_driven_agent.lab.prompts import format_plan
from besser.spec_driven_agent.lab.session import ACTIONS, LabSession
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from besser.spec_driven_agent.providers.llm_client import create_llm_client

PROG = "python -m besser.spec_driven_agent.lab"


def build_parser() -> argparse.ArgumentParser:
    """The command line's argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Send one message to the lab agent of BESSER Lab, which plans or builds a change to "
                    "BESSER's own repositories in a working copy.",
    )
    parser.add_argument("--root", required=True,
                        help="the lab working copy: the folder holding besser/ and modeling-agent/")
    parser.add_argument("--user", required=True, help="GitHub login the change and its commits belong to")
    parser.add_argument("--message", required=True, help="the message to the lab agent")
    parser.add_argument("--action", choices=ACTIONS, default="auto",
                        help="auto (default): plan until the active change's plan is approved, build after; "
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
        return outcome.build.summary
    if outcome.plan is not None:
        return format_plan(outcome.plan)
    return f"No plan was submitted (stop reason: {outcome.stop_reason})."


def main(argv: Optional[list[str]] = None) -> int:
    """Parse ``argv``, handle the message and report the outcome; returns the exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        client = create_llm_client(args.provider, api_key=None, model=args.model)
    except (ValueError, ImportError) as exc:
        print(f"{PROG}: cannot create the {args.provider} client: {exc}", file=sys.stderr)
        return 1
    session = LabSession(LabWorkspace(args.root), client=client, limits=LabLimits(max_cost_usd=args.max_cost),
                         callbacks=_stderr_callbacks())
    try:
        outcome = session.handle(args.message, user=args.user, action=args.action)
    except (LabWorkspaceError, InvalidApiKeyError) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 1
    print(_result_text(outcome))
    print(f"\nStop reason: {outcome.stop_reason}; cost ${outcome.cost_usd:.4f}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(asdict(outcome), handle, indent=2)
    completed = outcome.stop_reason == "completed" and (outcome.step == "build" or outcome.plan is not None)
    return 0 if completed else 1
