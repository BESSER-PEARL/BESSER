"""The lab agent's tools: the Spec-Driven Agent's file and shell tools, plus the lab's own.

``LabToolExecutor`` wraps :class:`~besser.spec_driven_agent.agent.tool_executor.ToolExecutor`
in a working copy of BESSER's repositories. It offers only the tools of the
current step, refuses any other by name, and bounds every result.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Sequence

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import INVALID_ARGUMENTS_KEY, get_tools_for
from besser.spec_driven_agent.lab.contracts import LAB_STATE_DIRNAME, ExtraTool

logger = logging.getLogger(__name__)

READ_TOOLS = frozenset({"list_files", "read_file", "search_in_files"})
WRITE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines", "delete_file"})
SHELL_TOOLS = frozenset({"run_command"})
STATE_TOOLS = frozenset({"task_list", "check_syntax"})

# Refused under any .git: config written there (filter, textconv, gpg) runs when the lab's own git commands read
# it, outside the sandbox; nor does the lab agent need to read a repository's internals.
_GIT_GUARDED_TOOLS = WRITE_TOOLS | {"read_file"}

#: Above this a tool result is cut. read_file already stops at 60,000 characters
#: of file text; the margin covers its JSON escaping, so a whole read is never cut.
MAX_RESULT_CHARS = 80_000

#: What ToolExecutor writes into its workspace (command logs, run records) starts
#: with this; git must never see it as a change of the working copy.
RUN_RECORD_PATTERN = ".besser_*"


class LabToolExecutor:
    """The tools of one lab step, run in the working copy ``workspace``.

    ``read_only`` offers the read and state tools and the extra tools that do
    not write; a write or shell call is then answered with an error result.
    ``time_left`` (seconds left in the run) caps every shell command.
    """

    def __init__(self, workspace: str, *, read_only: bool = False, allow_shell: bool = True,
                 extra_tools: Sequence[ExtraTool] = (), time_left: Callable[[], float] | None = None):
        # Per-write diagnostics scan the whole workspace for generated-app contract
        # mismatches: ~11 s a write on BESSER's checkout, and none apply to its source.
        self._executor = ToolExecutor(workspace=workspace, allow_shell=allow_shell and not read_only,
                                      per_write_diagnostics=False)
        self._executor.time_left = time_left
        self.read_only = read_only
        builtin = READ_TOOLS | STATE_TOOLS
        if not read_only:
            builtin |= WRITE_TOOLS | (SHELL_TOOLS if allow_shell else frozenset())
        self._extra = {tool.schema["name"]: tool for tool in extra_tools
                       if tool.writes is False or not read_only}
        clash = builtin & set(self._extra)
        if clash:
            raise ValueError(f"extra tools shadow built-in tools: {sorted(clash)}")
        self._schemas = [tool for tool in get_tools_for(has_domain_model=False, allow_shell=allow_shell)
                         if tool["name"] in builtin]
        self._schemas += [tool.schema for tool in self._extra.values()]
        self._offered = frozenset(tool["name"] for tool in self._schemas)
        self._read_only_refused = (WRITE_TOOLS | SHELL_TOOLS
                                   | {tool.schema["name"] for tool in extra_tools if tool.writes})
        if not read_only:
            exclude_run_records(self._executor.workspace)

    @property
    def workspace(self) -> str:
        """Absolute path of the working copy the tools act on."""
        return self._executor.workspace

    @property
    def schemas(self) -> list[dict]:
        """The tool definitions to send to the model."""
        return list(self._schemas)

    def execute(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Run one tool call; returns ``(text the model reads, is_error)``.

        Never raises for a bad call: a tool that is not offered, invalid
        arguments and a failing handler all come back as error results.
        """
        if self.read_only and name in self._read_only_refused:
            return (f"{name} is not available in this step: the working copy is read-only here. "
                    "Use list_files, read_file and search_in_files."), True
        if name not in self._offered:
            return f"Unknown tool: {name}. The tools of this step are: {', '.join(sorted(self._offered))}.", True
        extra = self._extra.get(name)
        if extra is None:
            if name in WRITE_TOOLS and self._is_lab_state(arguments):
                return (f"{name} refused: {LAB_STATE_DIRNAME}/ holds BESSER Lab's record of the change "
                        "(its state and notes); the lab writes it itself. Change the repositories' files."), True
            if name in _GIT_GUARDED_TOOLS and self._is_in_git_dir(arguments):
                return (f"{name} refused: .git holds a repository's own data and configuration, which the lab's "
                        "git commands read; the lab manages branches and commits itself. Change the "
                        "repositories' files."), True
            result = self._executor.execute_typed(name, arguments)
            return bound_result(result.to_json()), result.status == "error"
        if not isinstance(arguments, dict) or INVALID_ARGUMENTS_KEY in arguments:
            return f"{name} was not run: its arguments must be one well-formed JSON object. Resend the call.", True
        try:
            return bound_result(extra.handler(arguments)), False
        except ValueError as exc:
            # An extra tool refuses its arguments by raising ValueError.
            return f"{name} refused: {exc}", True
        except Exception as exc:
            logger.warning("Lab tool %s failed", name, exc_info=True)
            return f"{name} failed: {exc}", True

    def _is_lab_state(self, arguments: dict) -> bool:
        """Whether a file tool's ``path`` is in the lab's state folder at the workspace root."""
        path = arguments.get("path") if isinstance(arguments, dict) else None
        if not isinstance(path, str):
            return False
        full = os.path.normpath(os.path.join(self.workspace, path))
        top = os.path.relpath(full, self.workspace).replace("\\", "/").split("/", 1)[0]
        return os.path.normcase(top) == os.path.normcase(LAB_STATE_DIRNAME)

    def _is_in_git_dir(self, arguments: dict) -> bool:
        """Whether a file tool's ``path``, normalised, has a ``.git`` component (a git dir or a submodule's link)."""
        path = arguments.get("path") if isinstance(arguments, dict) else None
        if not isinstance(path, str):
            return False
        full = os.path.normpath(os.path.join(self.workspace, path))
        parts = os.path.relpath(full, self.workspace).replace("\\", "/").split("/")
        return any(os.path.normcase(part) == os.path.normcase(".git") for part in parts)

    def forget_shown_results(self) -> None:
        """The conversation lost earlier tool results (compaction): show re-reads in full."""
        self._executor.forget_shown_results()

    def close(self) -> None:
        """End the shell session and every process it still holds."""
        self._executor.close()


def bound_result(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """``text`` cut to ``limit`` characters, with a note saying how to get the rest."""
    if len(text) <= limit:
        return text
    return (text[:limit] + f"\n... [result cut: {len(text) - limit:,} of {len(text):,} characters not shown. "
            "Ask for less: read_file with offset and limit, a narrower search_in_files pattern or "
            "file_glob, or a command that filters its output or writes it to a file you read in parts.]")


def exclude_run_records(workspace: str) -> None:
    """Make git ignore the ``.besser_*`` run records the executor writes at ``workspace``.

    They are written at the workspace's top level only, so only a git checkout
    rooted there can see them; a workspace holding several checkouts needs
    nothing. Uses the repository's ``info/exclude``, never a tracked file.
    """
    git_dir = _git_dir(workspace)
    if git_dir is None:
        return
    exclude = os.path.join(_common_dir(git_dir), "info", "exclude")
    try:
        with open(exclude, encoding="utf-8") as handle:
            current = handle.read()
    except FileNotFoundError:
        current = ""
    if RUN_RECORD_PATTERN in (line.strip() for line in current.splitlines()):
        return
    os.makedirs(os.path.dirname(exclude), exist_ok=True)
    with open(exclude, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(("\n" if current and not current.endswith("\n") else "") + RUN_RECORD_PATTERN + "\n")


def _git_dir(checkout: str) -> str | None:
    """The git directory of the checkout rooted at ``checkout``, or None.

    ``.git`` is a directory, or a file ``gitdir: <path>`` in a submodule or a
    linked worktree.
    """
    dot_git = os.path.join(checkout, ".git")
    if os.path.isdir(dot_git):
        return dot_git
    if not os.path.isfile(dot_git):
        return None
    with open(dot_git, encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first.startswith("gitdir:"):
        return None
    return os.path.normpath(os.path.join(checkout, first[len("gitdir:"):].strip()))


def _common_dir(git_dir: str) -> str:
    """Where ``info/exclude`` lives: a linked worktree shares its main repository's."""
    commondir = os.path.join(git_dir, "commondir")
    if not os.path.isfile(commondir):
        return git_dir
    with open(commondir, encoding="utf-8") as handle:
        return os.path.normpath(os.path.join(git_dir, handle.read().strip()))
