import asyncio
import gc
import json
import subprocess
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, semantic_consistency
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.semantic_consistency import (
    check_semantic_consistency,
    generate_object_diagram,
)
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import needs_alloy


def _book_diagram(invariant: str) -> DiagramInput:
    """A class diagram with ``Book(title: str, pages: int)`` and the OCL invariant *invariant* on ``Book``."""
    return DiagramInput(title="Books", model={
        "type": "ClassDiagram",
        "elements": {
            "cls-book": {"id": "cls-book", "type": "Class", "name": "Book", "attributes": ["attr-title", "attr-pages"],
                         "methods": []},
            "attr-title": {"id": "attr-title", "type": "ClassAttribute", "name": "title", "visibility": "public",
                           "attributeType": "str"},
            "attr-pages": {"id": "attr-pages", "type": "ClassAttribute", "name": "pages", "visibility": "public",
                           "attributeType": "int"},
            "ocl-pages": {"id": "ocl-pages", "type": "ClassOCLConstraint",
                          "constraint": f"context Book inv pages: {invariant}"},
        },
        "relationships": {
            "r-pages": {"id": "r-pages", "type": "ClassOCLLink",
                        "source": {"element": "ocl-pages"}, "target": {"element": "cls-book"}},
        },
    })


@pytest.fixture
def consistent_diagram() -> DiagramInput:
    return _book_diagram("self.pages > 0")


@pytest.fixture
def contradictory_diagram() -> DiagramInput:
    return _book_diagram("self.pages = 1 and self.pages = 2")


@pytest.fixture
def too_large_diagram() -> DiagramInput:
    """The literal 40000 needs integers in [-65536, 65535]: too many for the Alloy Analyzer, even at scope 5."""
    return _book_diagram("self.pages = 40000")


@pytest.fixture
def untranslatable_diagram() -> DiagramInput:
    """Real numbers are valid OCL without an Alloy translation."""
    return _book_diagram("self.pages > 1.5")


@pytest.fixture
def invalid_ocl_diagram() -> DiagramInput:
    return _book_diagram("self.pages >")


@pytest.fixture
def state_machine_diagram() -> DiagramInput:
    return DiagramInput(title="Machine", model={"type": "StateMachineDiagram", "elements": {}, "relationships": {}})


def _events(stream) -> list[dict]:
    """The events of the SSE *stream*, an async generator of ``data:`` lines and heartbeat comments."""
    return [json.loads(line.removeprefix("data: ")) for line in _lines(stream) if not line.startswith(":")]


def _lines(stream) -> list[str]:
    """The lines of the SSE *stream*, an async generator."""
    async def collect():
        return [line async for line in stream]
    return asyncio.run(collect())


# ---------------------------------------------------------------------------
# Diagrams that cannot be checked (no Alloy needed)
# ---------------------------------------------------------------------------

def test_only_class_diagrams_are_checked(state_machine_diagram):
    events = _events(check_semantic_consistency(state_machine_diagram))
    assert len(events) == 1
    assert events[0]["done"] is True
    assert events[0]["isValid"] is False
    assert events[0]["message"] == "Semantic consistency is only checked for class diagrams."


def test_invalid_ocl_is_not_checked(invalid_ocl_diagram):
    events = _events(check_semantic_consistency(invalid_ocl_diagram))
    assert len(events) == 1
    assert events[0]["isValid"] is False
    assert events[0]["message"] == "The OCL constraints are invalid: the semantic check was skipped."
    assert len(events[0]["errors"]) >= 1


def test_analyzer_not_available(monkeypatch, consistent_diagram):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    events = _events(check_semantic_consistency(consistent_diagram))
    assert events[-1]["isValid"] is False
    assert events[-1]["message"] == "The Alloy Analyzer is not available on this server."


# ---------------------------------------------------------------------------
# Concurrent checks (no Alloy needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/besser_api/semantic-consistency-check", "/besser_api/generate-object-diagram"])
def test_endpoint_is_busy_while_a_check_runs(monkeypatch, path, state_machine_diagram):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()  # the check running
    monkeypatch.setattr(semantic_consistency, "_check_slots", slots)
    client = TestClient(app)
    response = client.post(path, json=state_machine_diagram.model_dump(mode="json"))
    assert response.status_code == 503


def test_slot_is_released_after_a_check(monkeypatch, state_machine_diagram):
    monkeypatch.setattr(semantic_consistency, "_check_slots", threading.BoundedSemaphore(1))
    first = _events(check_semantic_consistency(state_machine_diagram))
    second = _events(check_semantic_consistency(state_machine_diagram))
    assert first[-1]["done"] is True
    assert second[-1]["done"] is True


def test_slot_is_released_when_the_stream_never_starts(monkeypatch, state_machine_diagram):
    monkeypatch.setattr(semantic_consistency, "_check_slots", threading.BoundedSemaphore(1))
    stream = check_semantic_consistency(state_machine_diagram)
    del stream  # the client went away before the stream was read
    gc.collect()
    events = _events(check_semantic_consistency(state_machine_diagram))
    assert events[-1]["done"] is True


# ---------------------------------------------------------------------------
# Searches (Alloy needed)
# ---------------------------------------------------------------------------

@needs_alloy
def test_consistent_diagram(consistent_diagram):
    events = _events(check_semantic_consistency(consistent_diagram))
    assert events[-1]["isValid"] is True
    assert events[-1]["done"] is True
    assert events[-1]["message"].startswith("Instance found within 5 objects per class")


@needs_alloy
def test_heartbeats_while_the_analyzer_runs(monkeypatch, contradictory_diagram):
    monkeypatch.setattr(semantic_consistency, "HEARTBEAT_SECONDS", 0.01)
    lines = _lines(check_semantic_consistency(contradictory_diagram, scopes=(5,)))
    assert ": semantic-check-heartbeat\n\n" in lines  # the JVM alone takes longer to start


@needs_alloy
def test_object_diagram_of_a_consistent_diagram(consistent_diagram):
    events = _events(generate_object_diagram(consistent_diagram))
    object_diagram = events[-1]["object_model"]
    names = [element["name"] for element in object_diagram["elements"].values() if element["type"] == "ObjectName"]
    assert events[-1]["isValid"] is True
    assert "Book_0" in names  # Alloy numbers the atoms of a signature from 0


@needs_alloy
def test_contradictory_diagram(contradictory_diagram):
    events = _events(check_semantic_consistency(contradictory_diagram))
    not_found = [event for event in events if event["message"].startswith("No instance within")]
    assert len(not_found) == 3  # one per scope, the last one final
    assert events[-1]["isValid"] is False
    assert events[-1]["done"] is True
    assert events[-1]["message"].startswith("No instance within 9 objects per class")


@needs_alloy
def test_timeout_ends_the_search(consistent_diagram):
    events = _events(check_semantic_consistency(consistent_diagram, timeout=0))
    searches = [event for event in events if event["message"].startswith("Searching within")]
    assert len(searches) == 1  # a timeout ends the search
    assert events[-1]["isValid"] is False
    assert events[-1]["message"].startswith("Timed out searching within 5 objects per class")


@needs_alloy
def test_bounds_too_large(too_large_diagram):
    events = _events(check_semantic_consistency(too_large_diagram))
    searches = [event for event in events if event["message"].startswith("Searching within")]
    assert len(searches) == 1  # bounds too large end the search
    assert events[-1]["isValid"] is False
    assert events[-1]["message"].startswith("Too large to search within 5 objects per class")


@needs_alloy
def test_analyzer_error(monkeypatch, consistent_diagram):
    monkeypatch.setattr(semantic_consistency, "_analyze", lambda *args: (AlloyResult.ERROR, None))
    events = _events(check_semantic_consistency(consistent_diagram))
    assert events[-1]["isValid"] is False
    assert events[-1]["message"] == "The semantic check failed unexpectedly at scope 5."


@needs_alloy
def test_object_diagram_conversion_error(monkeypatch, consistent_diagram):
    def failing_conversion(*args):
        raise ValueError("cannot convert")
    monkeypatch.setattr(semantic_consistency, "_object_model_to_json", failing_conversion)
    events = _events(generate_object_diagram(consistent_diagram))
    assert events[-1]["done"] is True
    assert events[-1]["isValid"] is False
    assert events[-1]["message"] == "The semantic check failed unexpectedly at scope 5."


@needs_alloy
def test_scopes_and_timeout_are_parameters(contradictory_diagram):
    events = _events(check_semantic_consistency(contradictory_diagram, scopes=(1, 2), timeout=10))
    assert events[-1]["message"].startswith("No instance within 2 objects per class")


@needs_alloy
def test_untranslatable_diagram(untranslatable_diagram):
    events = _events(check_semantic_consistency(untranslatable_diagram))
    assert events[-1]["isValid"] is False
    assert events[-1]["message"] == "The model cannot be translated to Alloy."
    assert "Reals" in events[-1]["errors"][0]


@needs_alloy
def test_cancelling_the_stream_stops_the_analyzer(monkeypatch, consistent_diagram):
    # The Alloy Analyzer is replaced by a process that sleeps longer than the analysis timeout
    started = []
    popen = subprocess.Popen

    def sleeping_analyzer(args, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", sleeping_analyzer)
    monkeypatch.setattr(semantic_consistency, "HEARTBEAT_SECONDS", 0.01)

    async def cancel_at_first_heartbeat():
        stream = check_semantic_consistency(consistent_diagram, scopes=(5,))
        async for line in stream:
            if line.startswith(":"):
                break
        await stream.aclose()  # the client went away
        deadline = time.monotonic() + 5
        while started[0].poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return started[0].poll()

    assert asyncio.run(cancel_at_first_heartbeat()) is not None


# ---------------------------------------------------------------------------
# Endpoints (no Alloy needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/besser_api/semantic-consistency-check", "/besser_api/generate-object-diagram"])
def test_endpoint_streams_events(path, state_machine_diagram):
    client = TestClient(app)
    response = client.post(path, json=state_machine_diagram.model_dump(mode="json"))
    event = json.loads(response.text.removeprefix("data: "))
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert event["message"] == "Semantic consistency is only checked for class diagrams."
