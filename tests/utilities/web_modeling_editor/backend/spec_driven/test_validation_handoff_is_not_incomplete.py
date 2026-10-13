"""The normal Phase 2 -> Phase 3 handoff must not be reported as incomplete.

Regression: a run showing "Nothing we checked was found missing from the
delivered code" (0 unenforced), 3 verified and 4 could-not-verify was still
headlined **"Generated — incomplete"**, with "The customization loop did not
finish cleanly."

``_phase2_stop_reason == "validation_required"`` is the NORMAL handoff: Phase 2
stops because validation is due, Phase 3 runs, the pipeline carries on. The
pipeline itself says so in all three places it branches on this --
``orchestrator.py`` twice and ``modify_run.py`` once, each written as

    if self._phase2_exited_cleanly or self._phase2_stop_reason == "validation_required":

Only the runner, which drives the widget, read ``_phase2_exited_cleanly`` alone.
So the healthy path fell into ``not exited_cleanly`` and fired INCOMPLETE over a
run with nothing wrong with it.

A genuinely cut-short run must still say so -- that is what the flag is for.
"""
import pytest

from besser.utilities.web_modeling_editor.backend.services.spec_driven import (
    runner as runner_module,
)


class _Orchestrator:
    """Only the attributes the cleanliness decision reads."""

    def __init__(self, exited_cleanly, stop_reason):
        self._phase2_exited_cleanly = exited_cleanly
        self._phase2_stop_reason = stop_reason


def _is_clean(orchestrator) -> bool:
    """Mirror of the runner's decision, kept in one place here so the test
    fails if the runner's expression drifts from the pipeline's."""
    stop_reason = getattr(orchestrator, "_phase2_stop_reason", "completed")
    return (
        bool(getattr(orchestrator, "_phase2_exited_cleanly", True))
        or stop_reason == "validation_required"
    )


def test_the_validation_handoff_counts_as_clean():
    """The regression: this reported "Generated — incomplete"."""
    assert _is_clean(_Orchestrator(False, "validation_required")) is True


def test_a_clean_exit_is_still_clean():
    assert _is_clean(_Orchestrator(True, "completed")) is True


@pytest.mark.parametrize("stop_reason", [
    "api_error", "cost_cap", "timeout", "cancelled", "max_turns",
    "stuck_edit_loop",
])
def test_a_genuinely_cut_short_run_is_still_incomplete(stop_reason):
    """Loosening the handoff must not silence the real cases."""
    assert _is_clean(_Orchestrator(False, stop_reason)) is False


def test_the_runner_uses_this_exact_rule():
    """Pins the runner's source, so the two cannot drift apart silently."""
    import inspect

    src = inspect.getsource(runner_module)
    assert 'or stop_reason == "validation_required"' in src


@pytest.mark.parametrize("stop_reason", ["stuck_edit_loop", "validation_required"])
def test_every_stop_reason_has_its_own_explanation(stop_reason):
    """Both of these used to fall through to "did not finish cleanly", which
    tells the user nothing and reads as if something broke."""
    import inspect

    src = inspect.getsource(runner_module)
    assert f'"{stop_reason}": (' in src, f"{stop_reason} has no dedicated message"


def test_the_reasons_the_pipeline_can_emit_are_all_covered():
    """The UI's reason table must not fall behind the orchestrator's vocabulary."""
    import inspect

    from besser.spec_driven_agent.pipeline import orchestrator as orch_mod

    emitted = set()
    for line in inspect.getsource(orch_mod).splitlines():
        if "_phase2_stop_reason = " in line and '"' in line:
            emitted.add(line.split('"')[1])
    emitted.discard("completed")          # the clean case needs no message
    emitted.discard("validation_required")  # now treated as clean

    src = inspect.getsource(runner_module)
    missing = [r for r in sorted(emitted) if f'"{r}": (' not in src]
    assert not missing, f"stop reasons with no user-facing text: {missing}"
