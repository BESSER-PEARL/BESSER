"""Tests for the /csv-to-domain-model endpoint accepting CSV and XLSX uploads.

Regression coverage for https://github.com/BESSER-PEARL/BESSER/issues/501.
"""

import asyncio
import io

import httpx
import pytest
from httpx._transports.asgi import ASGITransport
from openpyxl import Workbook

from besser.utilities.web_modeling_editor.backend.backend import app

BASE_URL = "http://testserver"


def _run(coro):
    return asyncio.run(coro)


async def _post_files(url: str, files):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as ac:
        return await ac.post(url, files=files)


def _make_csv_bytes() -> bytes:
    return b"id,name\n1,Alice\n2,Bob\n"


def _make_xlsx_bytes() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.append(["id", "name"])
    ws.append([1, "Alice"])
    ws.append([2, "Bob"])
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def test_csv_upload_still_works():
    files = [("files", ("User.csv", _make_csv_bytes(), "text/csv"))]
    resp = _run(_post_files("/besser_api/csv-to-domain-model", files))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["diagramType"] == "ClassDiagram"


def test_xlsx_upload_is_accepted():
    files = [
        (
            "files",
            (
                "User.xlsx",
                _make_xlsx_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        )
    ]
    resp = _run(_post_files("/besser_api/csv-to-domain-model", files))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["diagramType"] == "ClassDiagram"


def test_unsupported_extension_is_rejected():
    files = [("files", ("User.pdf", b"not a spreadsheet", "application/pdf"))]
    resp = _run(_post_files("/besser_api/csv-to-domain-model", files))
    assert resp.status_code == 415
    assert ".csv" in resp.text and ".xlsx" in resp.text


def test_xlsx_with_bad_magic_bytes_is_rejected():
    files = [("files", ("User.xlsx", b"not a zip archive", "application/octet-stream"))]
    resp = _run(_post_files("/besser_api/csv-to-domain-model", files))
    assert resp.status_code == 400
    assert "XLSX" in resp.text or "Excel" in resp.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
