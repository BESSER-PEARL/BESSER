"""Validation and async file I/O for uploaded files.

Shared by every router that accepts an upload (``conversion_router`` and
``kg_router``), so the size, extension and content checks are enforced the same
way everywhere. The validators raise ``HTTPException`` directly (413, 415, 400):
they run before any service code, on the raw request body.
"""

import ast
import asyncio
import os
from typing import AbstractSet
from xml.parsers import expat

from fastapi import HTTPException, UploadFile

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    ALLOWED_OWL_EXTENSIONS,
    RDF_XML_EXTENSIONS,
)


def validate_upload(file: UploadFile, *, max_size: int, allowed_extensions: AbstractSet[str], content: bytes) -> None:
    """Validate an uploaded file's size and extension.

    Raises ``HTTPException`` with status 413 (payload too large) or 415
    (unsupported media type) when the check fails.
    """
    # --- size check ---
    if len(content) > max_size:
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_mb:.0f} MB.",
        )

    # --- extension check ---
    filename = file.filename or ""
    _, ext = os.path.splitext(filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed extensions: {', '.join(sorted(allowed_extensions))}."
            ),
        )


def validate_file_content(content: bytes, filename: str) -> None:
    """Validate uploaded file content based on its extension.

    Raises ``HTTPException`` with status 400 when the content does not match
    the expected format for the given file type.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".csv":
        # Must be valid UTF-8 text with at least one line containing comma or semicolon separators
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV file is not valid UTF-8 text.")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise HTTPException(status_code=400, detail="CSV file is empty.")
        has_separator = any("," in line or ";" in line for line in lines)
        if not has_separator:
            raise HTTPException(
                status_code=400,
                detail="CSV file does not contain comma or semicolon separators.",
            )

    elif ext == ".xlsx":
        # XLSX files are ZIP archives — check the ZIP magic bytes.
        if not content.startswith(b"PK\x03\x04"):
            raise HTTPException(
                status_code=400,
                detail="XLSX file appears to be corrupted or is not a valid Excel workbook.",
            )

    elif ext == ".py":
        # Must be valid Python syntax. Decode with utf-8-sig so a leading UTF-8
        # BOM (common from Windows editors) is stripped instead of tripping the
        # parser with "invalid non-printable character U+FEFF".
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Python file is not valid UTF-8 text.")
        try:
            ast.parse(text, filename=filename)
        except SyntaxError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Python file contains invalid syntax: {e.msg} (line {e.lineno}).",
            )

    elif ext in (".png", ".jpg", ".jpeg"):
        # Check magic bytes
        if ext == ".png":
            if not content.startswith(b'\x89PNG'):
                raise HTTPException(
                    status_code=400,
                    detail="File content does not match PNG format (invalid magic bytes).",
                )
        else:  # .jpg / .jpeg
            if not content.startswith(b'\xff\xd8\xff'):
                raise HTTPException(
                    status_code=400,
                    detail="File content does not match JPEG format (invalid magic bytes).",
                )

    elif ext in ALLOWED_OWL_EXTENSIONS:
        _validate_rdf_content(content, is_xml=ext in RDF_XML_EXTENSIONS)


def _validate_rdf_content(content: bytes, *, is_xml: bool) -> None:
    """Reject an RDF/OWL upload that is empty or, for RDF/XML, declares a DTD.

    Everything else (syntax, structure, encoding) is left to rdflib, the real
    parser. Text serialisations (Turtle, N-Triples, N3) are not XML and have no
    DTD, so their content is not scanned: a ``<!DOCTYPE`` inside one of their
    literals is just text. See :func:`_reject_xml_dtd` for the RDF/XML check.
    """
    if not content.strip():
        raise HTTPException(status_code=400, detail="OWL/RDF file is empty.")
    if is_xml:
        _reject_xml_dtd(content)


class _DtdFound(Exception):
    """Raised from an expat handler: the document declares a DTD."""


class _PrologEnd(Exception):
    """Raised from an expat handler: the root element starts, so the prolog is over."""


def _reject_xml_dtd(content: bytes) -> None:
    """Refuse an RDF/XML document whose prolog declares a DTD.

    Defence in depth, not the primary guard: the expat bundled with current
    Python (2.4.1 and later) already caps entity amplification ("billion
    laughs"), and rdflib's SAX parser leaves external entities unresolved by
    default. No ontology BESSER imports needs a DTD, though, so refusing one
    keeps entity expansion out of the request path whatever expat or rdflib
    version the server runs.

    The document is read by expat, the parser rdflib itself uses, so the check
    sees exactly the markup rdflib would: ``<!DOCTYPE`` inside a comment, a
    CDATA section or a literal is not a declaration, and an encoding expat
    detects on its own (UTF-16 with or without a BOM) is detected here too.
    Parsing stops at the root element, because a DTD can only sit in the
    prolog. A document expat cannot read is left for rdflib to report.

    Raises:
        HTTPException: 400 when a DTD is declared.
    """
    parser = expat.ParserCreate()

    def _dtd(*_args: object) -> None:
        raise _DtdFound()

    def _root(*_args: object) -> None:
        raise _PrologEnd()

    parser.StartDoctypeDeclHandler = _dtd
    # Entity declarations only occur inside a DTD; refused on their own too, so
    # the check never depends on the order expat reports the two in.
    parser.EntityDeclHandler = _dtd
    parser.StartElementHandler = _root
    try:
        parser.Parse(content, True)
    except _DtdFound:
        raise HTTPException(
            status_code=400,
            detail=(
                "OWL/RDF file declares a DTD (<!DOCTYPE ...>), which is not "
                "accepted. Please re-export the ontology without a DTD."
            ),
        ) from None
    except (_PrologEnd, expat.ExpatError):
        return


async def read_file_async(path: str, mode: str = "r", **kwargs) -> str | bytes:
    """Read a file without blocking the event loop."""
    def _do_read():
        with open(path, mode, **kwargs) as fh:
            return fh.read()
    return await asyncio.to_thread(_do_read)


async def write_file_async(path: str, data: bytes | str, mode: str = "wb") -> None:
    """Write data to a file without blocking the event loop."""
    def _do_write():
        with open(path, mode) as fh:
            fh.write(data)
    await asyncio.to_thread(_do_write)
