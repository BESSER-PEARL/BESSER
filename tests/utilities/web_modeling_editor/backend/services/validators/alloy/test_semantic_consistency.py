import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.semantic_consistency import (
    check_semantic_consistency,
    generate_object_diagram,
)

# Marks the tests that run the Alloy Analyzer, located through these environment variables
needs_alloy = pytest.mark.skipif(
    not (os.getenv("JAVA_HOME") and os.getenv("BESSER_ALLOY_JAR")),
    reason="JAVA_HOME and BESSER_ALLOY_JAR must be set to run the Alloy Analyzer",
)


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
    """The events of the SSE *stream*, an async generator of ``data:`` lines."""
    async def collect():
        return [json.loads(line.removeprefix("data: ")) async for line in stream]
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


# ---------------------------------------------------------------------------
# Searches (Alloy needed)
# ---------------------------------------------------------------------------

@needs_alloy
def test_consistent_diagram(consistent_diagram):
    events = _events(check_semantic_consistency(consistent_diagram))
    assert events[-1]["sat"] is True
    assert events[-1]["done"] is True
    assert events[-1]["message"].startswith("Instance found within 5 objects per class")


@needs_alloy
def test_the_search_announces_its_time_budget(consistent_diagram):
    events = _events(check_semantic_consistency(consistent_diagram))
    assert events[0]["time_budget_seconds"] == 140


@needs_alloy
def test_object_diagram_of_a_consistent_diagram(consistent_diagram):
    events = _events(generate_object_diagram(consistent_diagram))
    object_diagram = events[-1]["object_model"]
    names = [element["name"] for element in object_diagram["elements"].values() if element["type"] == "ObjectName"]
    assert events[-1]["sat"] is True
    assert "Book_0" in names  # Alloy numbers the atoms of a signature from 0


@needs_alloy
def test_contradictory_diagram(contradictory_diagram):
    events = _events(check_semantic_consistency(contradictory_diagram))
    not_found = [event for event in events if event["message"].startswith("No instance within")]
    assert len(not_found) == 4  # one per scope, the last one final
    assert events[-1]["sat"] is False
    assert events[-1]["done"] is True
    assert events[-1]["message"].startswith("No instance within 11 objects per class")


@needs_alloy
def test_no_instance_reason_unsat(contradictory_diagram):
    events = _events(check_semantic_consistency(contradictory_diagram))
    assert events[-1]["reason"] == "unsat"
    assert events[-1]["message"].startswith("No instance within 11 objects per class")


@needs_alloy
def test_no_instance_reason_timeout(consistent_diagram):
    events = _events(check_semantic_consistency(consistent_diagram, timeout=0))
    searches = [event for event in events if event["message"].startswith("Searching within")]
    assert len(searches) == 1  # a timeout ends the search
    assert events[-1]["reason"] == "timeout"
    assert events[-1]["sat"] is None
    assert events[-1]["message"].startswith("Timed out searching within 5 objects per class")


@needs_alloy
def test_scopes_and_timeout_are_parameters(contradictory_diagram):
    events = _events(check_semantic_consistency(contradictory_diagram, scopes=(1, 2), timeout=10))
    assert events[0]["time_budget_seconds"] == 30  # 2 scopes x (10 s + 5 s)
    assert events[-1]["message"].startswith("No instance within 2 objects per class")


@needs_alloy
def test_untranslatable_diagram(untranslatable_diagram):
    events = _events(check_semantic_consistency(untranslatable_diagram))
    assert events[-1]["isValid"] is False
    assert events[-1]["message"] == "The model cannot be translated to Alloy."
    assert "Real number" in events[-1]["errors"][0]


# ---------------------------------------------------------------------------
# Endpoints (no Alloy needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/besser_api/semantic-consistency-check", "/besser_api/generate-object-diagram"])
def test_endpoint_streams_events(path, state_machine_diagram):
    response = TestClient(app).post(path, json=state_machine_diagram.model_dump(mode="json"))
    event = json.loads(response.text.removeprefix("data: "))
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert event["message"] == "Semantic consistency is only checked for class diagrams."
