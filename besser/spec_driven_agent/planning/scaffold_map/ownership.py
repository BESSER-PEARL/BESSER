"""Who owns each mapped region: the generator, an earlier run's AI, a person, or this run.

Reads the run's line attribution (``.besser_attribution.json``, see
``state/agent_trace.py``) and never writes it: the map is built mid-run, and
the attribution is only advanced at packaging time. A line that no longer
matches the recorded state was changed after the last recorded point, which
during a run means it was written in this run.

The labels are advisory. The model can edit the state file through the file
tools, so nothing security-relevant may depend on them.
"""

from __future__ import annotations

from typing import Optional

from besser.spec_driven_agent.state.agent_trace import (
    KIND_AI,
    KIND_GENERATOR,
    KIND_HUMAN,
    AttributionState,
)

OWNER_GENERATOR = "generator"
OWNER_EARLIER_AI = "earlier AI run"
OWNER_HUMAN = "human edit"
OWNER_THIS_RUN = "this run"
OWNER_UNKNOWN = "unknown"

_CHANGED = {"kind": "_changed_since_recorded"}


class OwnershipIndex:
    """Per-line owners of the workspace's current files, computed lazily."""

    def __init__(self, workspace: str, lines_of) -> None:
        """``lines_of(rel)`` returns the file's current lines (SourceIndex.lines)."""
        self._state: Optional[AttributionState] = AttributionState.load(workspace)
        self._lines_of = lines_of
        self._owners: dict[str, Optional[list[str]]] = {}

    @property
    def available(self) -> bool:
        return self._state is not None

    def _file_owners(self, rel: str) -> Optional[list[str]]:
        if self._state is None:
            return None
        if rel not in self._owners:
            lines = self._lines_of(rel)
            owners: Optional[list[str]] = None
            if lines:
                if rel in self._state.files:
                    self._state.reconcile_file(rel, lines, _CHANGED)
                    hashes, indexes = self._state.files[rel]
                    owners = [self._label(self._state.contributors[i]) for i in indexes]
                else:
                    # Not in the recorded state: created after the last record.
                    owners = [OWNER_THIS_RUN] * len(lines)
            self._owners[rel] = owners
        return self._owners[rel]

    @staticmethod
    def _label(contributor: dict) -> str:
        kind = contributor.get("kind")
        if kind == KIND_GENERATOR:
            return OWNER_GENERATOR
        if kind == KIND_AI:
            return OWNER_EARLIER_AI
        if kind == KIND_HUMAN:
            return OWNER_HUMAN
        if contributor is _CHANGED or kind == _CHANGED["kind"]:
            return OWNER_THIS_RUN
        return OWNER_UNKNOWN

    def summarize(self, rel: str, start: int, end: int) -> Optional[str]:
        """``"generator"``, ``"generator + this run"``…, or None when attribution is unavailable."""
        try:
            owners = self._file_owners(rel)
        except Exception:
            # The state file is model-writable: a malformed one costs the label only.
            return None
        if not owners:
            return None
        window = owners[max(0, start - 1):max(0, end)]
        if not window:
            return None
        seen: list[str] = []
        for owner in window:
            if owner not in seen:
                seen.append(owner)
        return " + ".join(seen)
