"""Download filenames come from the generator registry (``GeneratorInfo``)."""

import asyncio
import re

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.config.generators import (
    SUPPORTED_GENERATORS,
    get_filename_for_generator,
)

# Names the frontend and users have seen since before the registry carried
# them; they must not change.
_LEGACY_NAMES = {
    "python": "classes.py",
    "test_case": "test_hypothesis.py",
    "pydantic": "pydantic_classes.py",
    "sqlalchemy": "sql_alchemy.py",
    "sql": "tables.sql",
    "jsonschema": "json_schema.json",
    "jsonobject": "object_model.json",
    "qiskit": "qiskit_circuit.py",
    "rdf": "vocabulary.ttl",
    "rest_api": "rest_api.zip",
    "react": "react_app.zip",
    "flutter": "flutter_app.zip",
    "terraform": "terraform.zip",
    "pytorch": "pytorch_nn.py",
    "tensorflow": "tf_nn.py",
    "bpmn": "bpmn_diagram.bpmn",
    "docker_compose": "docker_compose.zip",
    "agent": "agent_output.zip",
}


@pytest.mark.parametrize("generator, name", sorted(_LEGACY_NAMES.items()))
def test_existing_download_names_are_unchanged(generator, name):
    if generator not in SUPPORTED_GENERATORS:
        pytest.skip(f"{generator} is not registered (optional dependency missing)")
    assert get_filename_for_generator(generator) == name


@pytest.mark.parametrize("generator", sorted(SUPPORTED_GENERATORS))
def test_every_registered_generator_names_its_download(generator):
    info = SUPPORTED_GENERATORS[generator]
    if info.keep_generated_filename:
        assert info.output_type == "file"
        return
    assert info.download_filename, f"{generator} has no download_filename"
    assert info.download_filename.endswith(info.file_extension)
    assert get_filename_for_generator(generator) == info.download_filename


def test_unknown_generator_and_missing_name_fall_back():
    assert get_filename_for_generator("does-not-exist") == "output.txt"
    assert get_filename_for_generator("does-not-exist", base_name="x") == "x.txt"
    info = SUPPORTED_GENERATORS["java"]
    SUPPORTED_GENERATORS["_tmp"] = info._replace(download_filename=None)
    try:
        assert get_filename_for_generator("_tmp") == "_tmp_output.zip"
    finally:
        del SUPPORTED_GENERATORS["_tmp"]


def test_supabase_download_keeps_the_migration_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = {
        "title": "Library",
        "generator": "supabase",
        "model": {
            "version": "4.0.0",
            "type": "ClassDiagram",
            "nodes": [{
                "id": "c1", "type": "class", "position": {"x": 0, "y": 0}, "width": 160, "height": 70,
                "data": {"name": "Book", "methods": [], "attributes": [
                    {"id": "a1", "name": "title", "attributeType": "str", "visibility": "public"},
                ]},
            }],
            "edges": [],
        },
    }

    async def _request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/besser_api/generate-output", json=payload)

    response = asyncio.run(_request())
    assert response.status_code == 200, response.text
    disposition = response.headers["content-disposition"]
    assert re.fullmatch(r'attachment; filename="\d{14}_[a-z0-9_]+\.sql"', disposition), disposition
