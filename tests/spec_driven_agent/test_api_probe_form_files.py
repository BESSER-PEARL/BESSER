"""A test workflow can send a form login and a file upload, and a failed one it retired stays failed.

``test_api`` sent every body as JSON, so a retained workflow could never exercise a
form-encoded OAuth2 login or a multipart upload: those routes were run by hand
through the shell or not at all, and never counted as confirmed. A request now
carries ``form`` (urlencoded fields) and ``files`` (multipart, text content), as the
create probe already sends form bodies. A workflow that ran, failed and was then
retired read only "the agent retired it"; the report now says it failed.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation import api_probe
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

MAIN = '''from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI()


@app.post("/token")
def token(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    if password != "correct-horse":
        raise HTTPException(status_code=401, detail="bad credentials")
    return {"access_token": "issued-for-" + username, "token_type": "bearer"}


@app.post("/document/")
async def upload(title: Annotated[str, Form()], file: Annotated[UploadFile, File()]):
    data = await file.read()
    return {"id": 1, "title": title, "name": file.filename, "lines": data.decode().count("\\n")}
'''


def _backend(tmp_path):
    backend = tmp_path / "app" / "backend"
    backend.mkdir(parents=True)
    (backend / "main_api.py").write_text(MAIN, encoding="utf-8")
    (backend / "sql_alchemy.py").write_text("# no tables\n", encoding="utf-8")
    return str(tmp_path)


def test_a_form_login_and_a_multipart_upload_run_as_a_workflow(tmp_path):
    report = api_probe.probe_api_scenario(_backend(tmp_path), [
        {"method": "POST", "path": "/token", "form": {"username": "ana", "password": "correct-horse"},
         "expected_fields": {"token_type": "bearer"}},
        {"method": "POST", "path": "/document/", "form": {"title": "Orders"},
         "files": {"file": {"filename": "orders.csv", "content": "id,total\n1,10\n2,20\n", "content_type": "text/csv"}},
         "expected_fields": {"name": "orders.csv", "lines": 3, "title": "Orders"}},
        {"method": "POST", "path": "/token", "form": {"username": "ana", "password": "wrong"}, "expected_status": 401},
    ])
    assert report["status"] == "passed", report


@pytest.mark.parametrize("request_, message", [
    ({"method": "POST", "path": "/token", "json": {}, "form": {"a": "b"}}, "either json or form/files"),
    ({"method": "POST", "path": "/token", "form": {"a": {"nested": 1}}}, "form must map"),
    ({"method": "POST", "path": "/document/", "files": {"file": {"filename": "a.txt"}}}, "files must map"),
])
def test_a_body_in_two_shapes_or_a_malformed_upload_is_refused(request_, message):
    with pytest.raises(ValueError, match=message):
        api_probe._validate_requests([request_])


def test_a_failed_workflow_the_agent_retired_is_reported_as_failed_and_retired():
    report = build_verification_report({"api_scenarios_retired": [
        {"scenario_id": "upload-flow", "reason": "test_api cannot send the file", "last_status": "failed"},
        {"scenario_id": "old-flow", "reason": "replaced by upload-flow-v2", "last_status": "passed"},
    ]})
    whats = {item.id: (item.what, item.why) for item in report.notVerified}
    assert whats["upload-flow"] == ("Test workflow 'upload-flow' (failed, retired)",
                                    "it ran against the app and failed, then the agent retired it: "
                                    "test_api cannot send the file")
    assert whats["old-flow"] == ("Test workflow 'old-flow'", "the agent retired it: replaced by upload-flow-v2")
