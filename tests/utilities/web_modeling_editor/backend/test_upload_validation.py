"""Content checks of ``validate_file_content`` for OWL/RDF uploads.

An RDF/XML upload whose prolog declares a DTD is refused before rdflib parses
it; the check reads the document with expat, so markup that only *looks* like a
DTD (in a comment, a CDATA section, a Turtle literal) is accepted.
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


# --- RDF/XML with a DTD: refused --------------------------------------------

@pytest.mark.parametrize("padding", [0, 10, 5000, 200_000])
def test_a_dtd_behind_a_leading_comment_of_any_length_is_rejected(padding):
    assert _rejected(_bomb(padding), "x.owl")


@pytest.mark.parametrize("ext", [".owl", ".rdf", ".xml"])
def test_every_rdf_xml_extension_is_checked(ext):
    assert _rejected(b'<?xml version="1.0"?>\n<!DOCTYPE x>\n<x/>', "x" + ext)


def test_a_dtd_behind_a_long_whitespace_lead_in_is_rejected():
    assert _rejected((" \n" * 3000 + "<!DOCTYPE x>\n<x/>").encode(), "x.rdf")


def test_an_entity_declaration_is_rejected():
    content = b'<?xml version="1.0"?>\n<!DOCTYPE x [<!ENTITY a "b">]>\n<x>&a;</x>'
    assert _rejected(content, "x.xml")


def test_a_dtd_in_bomless_utf16_is_rejected():
    """expat detects BOM-less UTF-16 from the leading ``<\0``; so does the check."""
    doc = _bomb(10).decode("utf-8").replace(
        '<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-16"?>'
    )
    assert _rejected(doc.encode("utf-16-le"), "x.owl")


# --- Text that merely mentions a DTD: accepted ------------------------------

def test_an_ontology_without_a_dtd_is_accepted():
    content = ('<?xml version="1.0"?>\n<!--' + "x" * 10_000 + "-->\n" + _RDF_BODY % "Thing").encode()
    assert not _rejected(content, "x.owl")


def test_doctype_text_inside_an_xml_comment_is_accepted():
    content = ('<?xml version="1.0"?>\n<!-- exported without <!DOCTYPE rdf:RDF> -->\n'
               + _RDF_BODY % "Thing").encode()
    assert not _rejected(content, "x.owl")


def test_doctype_text_inside_cdata_is_accepted():
    content = (_RDF_BODY % "<![CDATA[<!DOCTYPE html> is an HTML preamble]]>").encode()
    assert not _rejected(content, "x.rdf")


@pytest.mark.parametrize("ext", [".ttl", ".nt", ".n3"])
def test_a_text_serialisation_with_doctype_in_a_literal_is_accepted(ext):
    content = (
        '<http://ex.org/page> <http://www.w3.org/2000/01/rdf-schema#comment> '
        '"Starts with <!DOCTYPE html> and declares <!ENTITY x \\"y\\">" .\n'
    ).encode()
    assert not _rejected(content, "x" + ext)


def test_an_empty_upload_is_rejected():
    assert _rejected(b"  \n", "x.ttl")


# --- End to end through /besser_api/kg/import-owl ---------------------------

def test_import_owl_rejects_the_padded_bomb_quickly(client):
    started = time.monotonic()
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("bomb.owl", _bomb(5000), "application/rdf+xml")},
    )
    assert resp.status_code == 400
    assert "DTD" in resp.json()["detail"]
    assert time.monotonic() - started < 10


def test_import_owl_accepts_a_commented_ontology(client):
    content = ('<?xml version="1.0"?>\n<!-- <!DOCTYPE in a comment -->\n<!--' + "x" * 5000 + "-->\n"
               + _RDF_BODY % "Thing").encode()
    resp = client.post(
        "/besser_api/import-owl",
        files={"owl_file": ("ok.owl", content, "application/rdf+xml")},
    )
    assert resp.status_code == 200, resp.text
