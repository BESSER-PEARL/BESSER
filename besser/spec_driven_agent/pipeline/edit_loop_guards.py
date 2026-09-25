"""Guards that notice the model is flailing on edits, and change the ask.

A rejected edit is byte-identical on disk to no edit at all, so a model
re-quoting an ``old_text`` it reconstructed wrongly looks exactly like one
that has finished. A high edit-failure rate on weaker models manufactures
the barren rounds the Phase 3 stall guard then acts on, which is why these live
separately from the stall guard itself: one detects flailing and redirects it,
the other decides the loop is over.

Mixed into ``LLMOrchestrator``; every method here reads and writes run state
through ``self``.
"""

from __future__ import annotations

import logging

from besser.spec_driven_agent.agent.history_eviction import (
    without_rejected_edit_drafts,
)
from besser.spec_driven_agent.pipeline.constants import _EDIT_TOOLS

logger = logging.getLogger(__name__)


class EditLoopGuardsMixin:
    """_apply_edit_loop_guards and friends; see the module docstring."""

    def _chat_with_pending_force(self, system: str, messages: list[dict]) -> dict:
        """One chat call. When an escalation asked for a specific tool, make
        that request non-streaming with ``force_tool`` (only clients whose
        ``chat`` accepts it; a plain client just gets the message)."""
        messages = without_rejected_edit_drafts(messages)
        force = self._force_tool_next
        self._force_tool_next = None
        if force and self._client_supports_structured_chat():
            return self.client.chat(
                system=system, messages=messages, tools=self.tools, force_tool=force,
            )
        if self.use_streaming and self.on_text and hasattr(self.client, "chat_stream"):
            return self._call_streaming(system, messages)
        return self.client.chat(system=system, messages=messages, tools=self.tools)

    def _apply_edit_loop_guards(self, messages: list[dict], *, where: str) -> bool:
        """Per-file modify streak + repeat-rejection escalation. True = stop.

        One mechanism shared by Phase 2, the fix cycle and the Phase 3 repair
        loop, so every phase gets the same recovery rather than ``_is_stuck``
        alone.
        """
        stuck_path = self._consecutive_modify_on_same_file()
        if stuck_path is not None:
            logger.warning(
                "Per-file modify loop (%s): %d consecutive edits on %s - "
                "injecting reminder", where, self._PER_FILE_MODIFY_THRESHOLD, stuck_path,
            )
            messages.append({"role": "user", "content": [
                {"type": "text", "text": self._build_modify_loop_reminder(stuck_path)}]})
            # Don't re-fire while the model is still on the same file.
            self._last_modify_warning_path = stuck_path
        elif self._recent_modify_targets:
            last_tool, last_path = self._recent_modify_targets[-1]
            if last_tool != "modify_file" or last_path != self._last_modify_warning_path:
                # Streak broken; a fresh one on this path may warn again.
                self._last_modify_warning_path = None
        return self._escalate_repeat_rejection(messages)

    def _escalate_repeat_rejection(self, messages: list[dict]) -> bool:
        """Act on ``executor.last_repeat``. Returns True when the caller's
        loop must stop.

        The channel carries two kinds of count: repeats of the byte-identical
        rejected call, and refusals at one TARGET across redrafts. Both end
        the phase at ``_TARGET_REFUSAL_STOP_AT`` - see the constant.
        """
        hit = getattr(self.executor, "last_repeat", None)
        if not hit:
            return False
        path, seen = hit
        if self._repeat_escalations.get(path) == seen:
            return False
        self._repeat_escalations[path] = seen
        tool = next((item.get("tool") for item in reversed(self._recent_tool_failures)
                     if item.get("tool") in _EDIT_TOOLS
                     and str(item.get("path", "")).replace("\\", "/").strip() == path),
                    "modify_file")
        if seen >= self._TARGET_REFUSAL_STOP_AT:
            logger.warning(
                "Stuck edit loop: %s on %s was refused %d times with no "
                "successful edit in between; ending the phase", tool, path, seen,
            )
            self._phase2_stop_reason = "stuck_edit_loop"
            return True
        if seen >= self._REPEAT_FORCE_AT:
            # Only steer toward the range editor when the failing strategy is
            # text quotation. Sending a repeating range edit back through
            # read -> replace_file_lines is the loop it is already in.
            #
            # _force_tool_next is a single slot the executor's recovery ladder
            # also writes, earlier in the same turn. The executor saw the
            # actual refusal, so never overwrite its choice.
            if self._force_tool_next is None:
                self._force_tool_next = "read_file" if tool == "modify_file" else None
            strategy = (
                "Read the target block, then use replace_file_lines with the returned "
                "read_id and inclusive line numbers."
                if tool == "modify_file" else
                "Re-selecting the same lines will fail the same way. Correct new_text "
                "itself — complete lines, real indentation, balanced brackets — or "
                "select the whole enclosing block."
            )
            text = (
                f"<system-reminder>{tool} on `{path}` was rejected {seen} times "
                f"with the same arguments; the executor will not apply it. {strategy} "
                "The file remains editable. For an already-present change, verify "
                "behavior instead of inserting it again. A rejected edit is not "
                "completion; do not mark it done or drop the requirement."
                "</system-reminder>"
            )
        else:
            return False
        logger.warning("Repeat rejection on %s (%d): %s", path, seen, text[:80])
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        return False

    def _is_stuck(self) -> bool:
        recent = self._recent_tool_calls[-self._LOOP_THRESHOLD:]
        return (
            len(recent) >= self._LOOP_THRESHOLD
            and len({key for key, _ in recent}) == 1
            and not any(ok for _, ok in recent)
        )

    def _consecutive_modify_on_same_file(self) -> str | None:
        """Return the file path being repeatedly modified, or None.

        Fires when the tail of the recent tool history is
        ``_PER_FILE_MODIFY_THRESHOLD`` ``modify_file`` calls on the SAME
        (normalised) path that were ALL refused (the executor resets its
        miss count on a successful edit, so N good edits to one file never
        fire). A ``read_file`` on that same path does NOT break the streak:
        re-reading the file you cannot edit is the flail's own rhythm, and
        counting it as movement lets a modify/read alternation run unchecked
        for dozens of turns. Any other tool, and
        a read of a DIFFERENT file, still breaks it: those are real movement.

        Resets / suppresses repeat firing: once we've warned about a
        path, ``_last_modify_warning_path`` is set; subsequent identical
        streaks return None until the LLM either switches files or
        switches tools.
        """
        n = self._PER_FILE_MODIFY_THRESHOLD
        # The most recent edit call fixes which file the streak is about.
        path = next(
            (p for tool, p in reversed(self._recent_modify_targets)
             if tool in _EDIT_TOOLS),
            None,
        )
        if path is None:
            return None
        # Walk back over modify calls on that file, stepping over a re-read
        # of the SAME file (part of the flail — see the note at the
        # recording site). Anything else, including a read of a different
        # file, is real movement and ends the streak.
        streak = 0
        for tool, entry in reversed(self._recent_modify_targets):
            if tool in _EDIT_TOOLS and entry == path:
                streak += 1
            elif tool == "read_file" and entry == path:
                continue
            else:
                break
        if streak < n:
            return None
        if path is None:
            # modify_file without a parseable path argument — skip.
            return None
        if self.executor.consecutive_modify_misses(path) < n:
            # Successful edits are ordinary work on one file, not a flail.
            return None
        if path == self._last_modify_warning_path:
            # Already warned about this streak; wait for a real change
            # of file or tool before firing again.
            return None
        return path

    def _build_modify_loop_reminder(self, path: str) -> str:
        """Name the actual refusal reason; a syntax rejection is not a text miss."""
        n = self._PER_FILE_MODIFY_THRESHOLD
        failure = next((item for item in reversed(self._recent_tool_failures)
                        if item.get("tool") in _EDIT_TOOLS
                        and str(item.get("path", "")).replace("\\", "/").strip() == path), {})
        tool = failure.get("tool", "modify_file")
        error = str(failure.get("error", ""))[:400]
        kind = failure.get("rejection_kind")
        if kind == "syntax_error" or "syntax" in error.lower():
            advice = (
                "The proposed edit was refused by the syntax guard; this is not evidence that old_text failed to match. "
                "The proposal was NOT applied. Use the CURRENT ON-DISK excerpt, or read_file around the enclosing "
                "function/try/except block. Preserve its complete indentation and control-flow structure; "
                "do not copy the rejected would_write proposal as current source. Correct the replacement, "
                "then retry one focused edit."
            )
        else:
            advice = (
                "Use the reported reason before retrying. For a missing/ambiguous target, call read_file "
                "on the affected region, then switch to replace_file_lines with its read_id "
                "and exact inclusive line numbers instead of quoting old_text again. "
                "For another refusal, resolve that specific guard instead of repeating unchanged arguments. "
                "Verify any already-present change from current source before marking it done."
            )
        if tool == "replace_file_lines":
            # Re-reading and range-editing again is what just failed N times,
            # so do not send the model back around that same loop.
            advice = (
                "Re-reading and selecting the same range again is what just failed. "
                "The refusal reason above is about the replacement text, not the line "
                "numbers: correct new_text (complete lines, real indentation, balanced "
                "brackets), or select the whole enclosing block. If the change is "
                "already present, verify the behavior instead of editing again."
            )
        return (
            f"<system-reminder>Your last {n} {tool} calls on `{path}` "
            f"were refused. Latest reason: {error or 'inspect the tool response'}. "
            f"{advice} Do NOT rewrite `{path}` from memory.</system-reminder>"
        )

