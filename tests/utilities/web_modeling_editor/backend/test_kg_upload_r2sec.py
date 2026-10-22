"""OWL/RDF uploads that declare a DTD are refused wherever the DTD sits.

Regression tests for a bypass of the entity-expansion guard: it only looked at
the first 4096 characters, so a long leading XML comment pushed a nested-entity
DTD ("billion laughs") past the probe and into rdflib's unbounded expat parser.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.services.utils.upload_validation import (
    validate_file_content,
)

_RDF_BODY = (
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
    'xmlns:owl="http://www.w3.org/2002/07/owl#">'
    '<owl:Class rdf:about="http://ex.org/A"><rdfs:label>%s</rdfs:label></owl:Class></rdf:RDF>'
)


def _nested_entities(depth: int = 6) -> str:
    entities = ['<!ENTITY e0 "%s">' % ("A" * 10)]
    for i in range(1, depth):
        entities.append('<!ENTITY e%d "%s">' % (i, ("&e%d;" % (i - 1)) * 10))
    return "".join(entities)


def _bomb(padding: int, depth: int = 6) -> bytes:
    """A small RDF/XML file whose DTD sits behind ``padding`` chars of comment."""
    return (
        '<?xml version="1.0"?>\n<!--' + "x" * padding + "-->\n"
        "<!DOCTYPE rdf:RDF [" + _nested_entities(depth) + "]>\n"
        + _RDF_BODY % ("&e%d;" % (depth - 1))
    ).encode("utf-8")


def _rejected(content: bytes, filename: str) -> bool:
    try:
        validate_file_content(content, filename)
    except HTTPException as exc:
        assert exc.status_code == 400
        return True
    return False


@pytest.mark.parametrize("padding", [0, 10, 4096, 5000, 200_000])
def test_a_dtd_behind_a_leading_comment_of_any_length_is_rejected(padding):
    assert _rejected(_bomb(padding), "x.owl")


def test_a_dtd_behind_a_long_whitespace_lead_in_is_rejected():
    assert _rejected((" \n" * 3000 + "<!DOCTYPE x>").encode(), "x.rdf")


def test_a_lowercase_doctype_after_a_bom_is_rejected():
    assert _rejected("﻿<?xml version='1.0'?><!doctype x>".encode(), "x.rdf")


def test_an_entity_declaration_alone_is_rejected():
    assert _rejected(b'<?xml version="1.0"?>\n<!-- c -->\n<!ENTITY a "b">', "x.xml")


def test_a_dtd_hidden_as_bomless_utf16_is_rejected():
    """BOM-less UTF-16LE of ASCII markup is also valid UTF-8 (every other byte is NUL)."""
    doc = _bomb(10).decode("utf-8").replace('<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-16"?>')
    assert _rejected(doc.encode("utf-16-le"), "x.owl")


def test_an_ontology_without_a_dtd_is_accepted():
    content = ('<?xml version="1.0"?>\n<!--' + "x" * 10_000 + "-->\n" + _RDF_BODY % "Thing").encode()
    assert not _rejected(content, "x.owl")


def test_import_owl_rejects_the_padded_bomb_quickly(client):
    """End to end: the 5.6 KB nested-entity file from the review returns 400 at once."""
    started = time.monotonic()
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("bomb.owl", _bomb(5000), "application/rdf+xml")},
    )
    assert resp.status_code == 400
    assert "DTD" in resp.json()["detail"]
    assert time.monotonic() - started < 10


def test_import_owl_still_accepts_a_commented_ontology(client):
    content = ('<?xml version="1.0"?>\n<!--' + "x" * 5000 + "-->\n" + _RDF_BODY % "Thing").encode()
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("ok.owl", content, "application/rdf+xml")},
    )
    assert resp.status_code == 200, resp.text
