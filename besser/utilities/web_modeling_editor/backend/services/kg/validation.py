"""Checking a Knowledge Graph for the web endpoints.

One place runs the OWL/SHACL consistency check for both callers:
``POST /kg/consistency`` (``routers/kg_router.py``) returns its report, and
``POST /validate-diagram`` (``routers/validation_router.py``) folds it, together
with the KG -> class diagram preflight, into a list of errors and warnings.

The check itself (:func:`~besser.BUML.notations.kg_to_buml.consistency.check_kg_consistency`)
runs pyshacl in a child process with a time budget. Here it also gets a
process-wide cap on how many run at once (``KG_CONSISTENCY_MAX_CONCURRENT``):
each one is a Python process holding pyshacl and owlrl for up to the timeout,
so a request beyond the cap is refused with :class:`ConsistencyCheckBusy`
rather than queued.
"""

import asyncio
import threading
from typing import List, Optional, Sequence, Tuple

from besser.BUML.metamodel.kg import KnowledgeGraph
from besser.BUML.notations.kg_to_buml import analyze_kg_for_class_diagram
from besser.BUML.notations.kg_to_buml.consistency import (
    ConsistencyCheckError,
    ConsistencyCheckTimeout,
    ConsistencyReport,
    check_kg_consistency,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    KG_CONSISTENCY_MAX_CONCURRENT,
    KG_CONSISTENCY_TIMEOUT_SECONDS,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    BesserError,
    ConfigurationError,
)


class ConsistencyCheckBusy(BesserError):
    """Every consistency-check slot is taken; the client should retry shortly.

    ``POST /kg/consistency`` answers it with HTTP 429. The message is written
    for the end user.
    """


_SLOTS: Optional[threading.BoundedSemaphore] = None
_SLOTS_LOCK = threading.Lock()


def _slots() -> threading.BoundedSemaphore:
    """The process-wide semaphore, built on first use from the current cap."""
    global _SLOTS
    with _SLOTS_LOCK:
        if _SLOTS is None:
            _SLOTS = threading.BoundedSemaphore(KG_CONSISTENCY_MAX_CONCURRENT)
        return _SLOTS


def _reset_slots_for_tests() -> None:
    """Drop the semaphore so the next check rebuilds it (after patching the cap)."""
    global _SLOTS
    with _SLOTS_LOCK:
        _SLOTS = None


async def run_consistency_check(kg: KnowledgeGraph) -> ConsistencyReport:
    """Run the consistency check on ``kg`` off the event loop, under the concurrency cap.

    The slot is released by the worker thread when the check (and so its child
    process) is over, not when this coroutine ends: a client that disconnects
    must not free a slot whose validator is still running.

    Raises:
        ConsistencyCheckBusy: ``KG_CONSISTENCY_MAX_CONCURRENT`` checks are
            already running.
        ConsistencyCheckTimeout: the check ran out of time.
        ConsistencyCheckError: the KG's shapes could not be loaded.
        ConfigurationError: pyshacl is not installed on this server.
    """
    slots = _slots()
    if not slots.acquire(blocking=False):
        raise ConsistencyCheckBusy(
            "Too many consistency checks are running right now. Please retry in a moment."
        )

    # Whoever claims the slot first releases it: the worker once the check is
    # over, or the coroutine if it is cancelled before the worker ever started.
    claim = threading.Lock()
    claimed = False

    def _claim() -> bool:
        nonlocal claimed
        with claim:
            if claimed:
                return False
            claimed = True
            return True

    def _check() -> Optional[ConsistencyReport]:
        if not _claim():
            return None
        try:
            return check_kg_consistency(kg, timeout_seconds=KG_CONSISTENCY_TIMEOUT_SECONDS)
        finally:
            slots.release()

    try:
        report = await asyncio.to_thread(_check)
    except asyncio.CancelledError:
        if _claim():
            slots.release()
        raise
    except ImportError as exc:
        raise ConfigurationError(str(exc)) from exc
    # ``_check`` returns None only after this coroutine was cancelled, so never here.
    return report


def format_kg_finding(code: str, description: str, node_ids: Sequence[str]) -> str:
    """Render one KG finding as a single line of a validation response.

    ``KGIssue`` and ``ConsistencyIssue`` both carry a human-readable text and the
    ids they concern; the ids are what let a caller locate the problem without a
    second request.
    """
    text = f"[{code}] {description}".rstrip()
    if node_ids:
        text = f"{text} (nodes: {', '.join(node_ids)})"
    return text


async def validate_knowledge_graph(kg: KnowledgeGraph) -> Tuple[List[str], List[str]]:
    """The ``(errors, warnings)`` of ``kg`` for ``/validate-diagram``.

    The same two checks the editor's Refine KG modal runs, so validating through
    the API and through the UI agree: every preflight issue is an error, a
    consistency violation is an error, and any other consistency finding is a
    warning. Only the findings are reported; the modal's recommended/skip
    actions are interactive choices with no meaning in a validation response.

    A check that could not run to the end does not make the KG invalid: running
    out of time, a busy server or a server without pyshacl is a warning. Shapes
    pyshacl refuses to load are a finding about the KG, so they are an error.
    """
    errors: List[str] = []
    warnings: List[str] = []

    preflight = await asyncio.to_thread(analyze_kg_for_class_diagram, kg)
    for issue in preflight.issues:
        errors.append(format_kg_finding(issue.code, issue.description, issue.affected_node_ids))

    try:
        consistency = await run_consistency_check(kg)
    except (ConsistencyCheckTimeout, ConsistencyCheckBusy, ConfigurationError) as exc:
        warnings.append(f"The consistency check did not run to completion: {exc}")
        return errors, warnings
    except ConsistencyCheckError as exc:
        errors.append(str(exc))
        return errors, warnings

    for issue in consistency.issues:
        line = format_kg_finding(
            issue.code, issue.constraint_label or issue.message, issue.affected_node_ids,
        )
        (errors if issue.severity == "violation" else warnings).append(line)
    return errors, warnings
