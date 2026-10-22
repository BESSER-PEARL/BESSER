"""Validation and async file I/O for uploaded files.

Shared by every router that accepts an upload (``conversion_router`` and
``kg_router``), so the size, extension and content checks are enforced the same
way everywhere. The validators raise ``HTTPException`` directly (413, 415, 400):
they run before any service code, on the raw request body.
"""

import ast
import asyncio
import os

from fastapi import HTTPException, UploadFile

#: Extensions whose content is parsed as RDF (see ``validate_file_content``).
RDF_EXTENSIONS = frozenset({".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3"})

# Markup that declares a DTD. Matched case-insensitively against the whole
# decoded upload.
_DTD_MARKERS = ("<!DOCTYPE", "<!ENTITY")


def validate_upload(file: UploadFile, *, max_size: int, allowed_extensions: set[str], content: bytes) -> None:
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

    elif ext in RDF_EXTENSIONS:
        _validate_rdf_content(content)


def _validate_rdf_content(content: bytes) -> None:
    """Reject an RDF/OWL upload that is empty, not UTF-8, or declares a DTD.

    rdflib's RDF/XML parser builds its parser with ``xml.sax.make_parser()`` and
    never disables entity resolution, and stock expat has no expansion limit, so
    a few hundred bytes of nested internal entities ("billion laughs") expand
    into minutes of CPU inside the request. No ontology BESSER imports needs a
    DTD, so any declaration is refused before rdflib sees the file.

    The scan covers the **whole** document rather than its head. A DTD is only
    legal in the prolog, but the prolog may begin with comments and processing
    instructions of any length, so no fixed-size probe is enough. A NUL character
    is refused too: the only way to hide ``<!DOCTYPE`` from this text scan while
    keeping the bytes valid UTF-8 is to encode the markup as BOM-less UTF-16,
    which expat detects from the leading ``<\\0`` and decodes on its own.
    """
    # The real parser (rdflib) performs the structural validation.
    if not content.strip():
        raise HTTPException(status_code=400, detail="OWL/RDF file is empty.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="OWL/RDF file is not valid UTF-8 text.")

    if "\x00" in text:
        raise HTTPException(
            status_code=400,
            detail="OWL/RDF file contains NUL characters; please upload it as UTF-8 text.",
        )
    upper = text.upper()
    if any(marker in upper for marker in _DTD_MARKERS):
        raise HTTPException(
            status_code=400,
            detail=(
                "OWL/RDF file declares a DTD (<!DOCTYPE/<!ENTITY), which is "
                "not accepted: entity expansion can be used to exhaust server "
                "resources. Please re-export the ontology without a DTD."
            ),
        )


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
