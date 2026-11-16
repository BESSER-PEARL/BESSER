"""``process_trace``: reading trace file JSON of the current schema version and of older ones."""

import pytest

from besser.utilities.web_modeling_editor.backend.constants.constants import TRACE_SCHEMA_VERSION
from besser.utilities.web_modeling_editor.backend.services.converters import process_trace, trace_buml_to_json
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml import trace_processor
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

T = "2026-10-29T09:30:00.000Z"


def _trace(version: int, links: int = 1, **file_fields) -> dict:
    """A generation: one model, ``links`` files each linked from it."""
    files = [{"id": f"f{i}", "type": "CodeArtifact", "name": f"f{i}.py", "path": f"f{i}.py", "contentHash": "h",
              "timestamp": T} for i in range(links)]
    files[0].update(file_fields)
    return {
        "schemaVersion": version, "projectId": None,
        "actors": [{"id": "ada", "type": "Person", "name": "Ada", "timestamp": T}],
        "tools": [{"id": "gen", "type": "CodeGenerator", "name": "sqlalchemy", "timestamp": T}],
        "artifacts": [{"id": "m", "type": "ModelArtifact", "name": "Library", "elementId": "diagram-1",
                       "elementType": "DomainModel", "timestamp": T}, *files],
        "operations": [{"id": "op", "type": "CodeGeneration", "status": "SUCCEEDED", "startedAt": T,
                        "performedBy": ["ada"], "reads": ["m"], "writes": [f["id"] for f in files],
                        "toolCalls": [{"id": "call", "toolId": "gen", "sequenceNumber": 0, "status": "SUCCEEDED",
                                       "startedAt": T, "logs": []}]}],
        "traceLinks": [{"id": f"l{i}", "sourceId": "m", "targetId": f"f{i}", "operationId": "op",
                        "evidence": [], "timestamp": T} for i in range(links)],
    }


def test_the_current_version_is_6():
    assert TRACE_SCHEMA_VERSION == 6


def test_a_version_5_trace_is_read_and_written_back_as_the_current_version():
    data = _trace(5)
    written = trace_buml_to_json(process_trace(data))
    assert written == {**data, "schemaVersion": TRACE_SCHEMA_VERSION}


def test_unknown_versions_are_refused():
    for version in (4, 7, "6", None):
        with pytest.raises(ConversionError, match="schemaVersion"):
            process_trace(_trace(6) | {"schemaVersion": version})


def test_a_revision_round_trips():
    data = _trace(6, links=2)
    data["artifacts"][2]["revisionOfId"] = "f0"  # f1 is an edited version of f0 ...
    data["operations"][0]["writes"] = ["f0"]
    data["tools"].append({"id": "llm", "type": "LLM", "name": "gpt-6-luna", "provider": "openai", "timestamp": T})
    data["operations"].append({  # ... written by the LLM edit, not by the generator
        "id": "edit", "type": "CodeGeneration", "status": "SUCCEEDED", "startedAt": T, "performedBy": ["ada"],
        "reads": ["f0"], "writes": ["f1"],
        "toolCalls": [{"id": "call-llm", "toolId": "llm", "sequenceNumber": 0, "status": "SUCCEEDED",
                       "startedAt": T, "logs": []}]})
    trace = process_trace(data)
    assert {a.id: a for a in trace.artifacts}["f1"].revision_of.id == "f0"
    assert trace_buml_to_json(trace) == data


@pytest.mark.parametrize("fields, message", [
    ({"revisionOfId": "missing"}, "unknown artifact"),
    ({"revisionOfId": "m"}, "must be a CodeArtifact"),
    ({"revisionOfId": "f0"}, "revision of itself"),
])
def test_an_invalid_revision_is_a_conversion_error(fields, message):
    with pytest.raises(ConversionError, match=message):
        process_trace(_trace(6, **fields))


def test_only_a_code_artifact_has_a_revision():
    data = _trace(6)
    data["artifacts"][0]["revisionOfId"] = "f0"
    with pytest.raises(ConversionError, match="only a CodeArtifact has revisionOfId"):
        process_trace(data)


LARGE_TRACE_LINKS = 2000


def test_a_large_trace_is_read_in_linear_time(assert_scales_linearly):
    # A quadratic collection once took minutes for 5000 links.
    def read(links):
        data = _trace(6, links=links)
        return lambda: process_trace(data)

    assert_scales_linearly(read, LARGE_TRACE_LINKS)
    assert len(process_trace(_trace(6, links=LARGE_TRACE_LINKS)).trace_links) == LARGE_TRACE_LINKS


def _chain(length: int, field: str) -> dict:
    """A trace whose artifacts form one ``field`` chain (each the revision, or fragment, of the one before)."""
    data = _trace(6, links=length)
    for index, artifact in enumerate(data["artifacts"][2:], start=1):
        artifact[field] = f"f{index - 1}"
    data["operations"][0]["writes"] = ["f0"]
    data["traceLinks"] = data["traceLinks"][:1]
    return data


@pytest.mark.parametrize("field", ["revisionOfId", "parentId"])
def test_a_long_chain_is_read_in_linear_time(assert_scales_linearly, field):
    # Each link once walked the chain above it: 30k versions took 15 s, inline in the export endpoint.
    def read(length):
        data = _chain(length, field)
        return lambda: process_trace(data)

    assert_scales_linearly(read, 2000)


def test_a_cycle_through_many_artifacts_is_a_conversion_error():
    data = _chain(50, "revisionOfId")
    data["artifacts"][1]["revisionOfId"] = "f49"
    with pytest.raises(ConversionError, match="revision of itself"):
        process_trace(data)


def test_a_trace_over_the_element_cap_is_refused(monkeypatch):
    monkeypatch.setattr(trace_processor, "TRACE_MAX_ELEMENTS", 3)
    with pytest.raises(ConversionError, match="holds 4 artifacts, more than the 3"):
        process_trace(_trace(6, links=3))
