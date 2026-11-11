"""Generators that write model names into Python refuse names that are not identifiers.

The ``NamedElement.name`` setter accepts ``;``, ``=`` and quotes so saved models keep
loading; the check runs at generation time, before anything is rendered or executed.
"""

import asyncio
import os
import subprocess

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, EnumerationLiteral, Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.django import DjangoGenerator
from besser.generators.generator_interface import InvalidIdentifierError
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.python_classes import PythonGenerator
from besser.generators.sql import SQLGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator

# Code-shaped, but inert: the gate must stop it whatever it would do.
CODE_SHAPED_NAME = "A=1;B"


def _model_with_literal(literal_name: str) -> DomainModel:
    status = Enumeration(name="Status", literals={EnumerationLiteral(name=literal_name)})
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    return DomainModel(name="Library", types={book, status})


GENERATORS = {
    "sqlalchemy": lambda model, out: SQLAlchemyGenerator(model, output_dir=out),
    "sql": lambda model, out: SQLGenerator(model, output_dir=out),
    "python": lambda model, out: PythonGenerator(model, output_dir=out),
    "pydantic": lambda model, out: PydanticGenerator(model, output_dir=out),
    "backend": lambda model, out: BackendGenerator(model, output_dir=out),
    "django": lambda model, out: DjangoGenerator(model, project_name="proj", app_name="app", output_dir=out),
}


@pytest.mark.parametrize("key", sorted(GENERATORS))
def test_code_shaped_name_is_rejected_before_anything_is_written(key, tmp_path, monkeypatch):
    def _no_subprocess(*args, **kwargs):
        raise AssertionError("a subprocess ran before the model was validated")

    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    out = tmp_path / "out"
    out.mkdir()
    generator = GENERATORS[key](_model_with_literal(CODE_SHAPED_NAME), str(out))

    with pytest.raises(InvalidIdentifierError, match="Literal 'A=1;B' of enumeration 'Status'"):
        generator.generate()
    assert os.listdir(out) == []


def test_identifier_error_is_a_value_error():
    # The backend maps ValueError to HTTP 400; the gate relies on it.
    assert issubclass(InvalidIdentifierError, ValueError)


def test_valid_names_still_generate(tmp_path):
    SQLAlchemyGenerator(_model_with_literal("AVAILABLE"), output_dir=str(tmp_path)).generate()
    assert (tmp_path / "sql_alchemy.py").is_file()


def test_generate_output_endpoint_returns_400_for_code_shaped_name(monkeypatch, tmp_path):
    from besser.utilities.web_modeling_editor.backend.backend import app

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("a subprocess ran before the model was validated")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    payload = {
        "title": "Library",
        "generator": "sql",
        "model": {
            "type": "ClassDiagram",
            "elements": {
                "enum-1": {"type": "Enumeration", "name": "Status", "attributes": ["lit-1"], "methods": []},
                "lit-1": {"type": "ClassAttribute", "name": CODE_SHAPED_NAME},
            },
            "relationships": {},
        },
    }

    async def _request():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/besser_api/generate-output", json=payload)

    response = asyncio.run(_request())
    assert response.status_code == 400, response.text
    assert "is not a valid identifier" in response.json()["detail"]
