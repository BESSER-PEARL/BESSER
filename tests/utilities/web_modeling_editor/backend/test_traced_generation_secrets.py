"""A trace is exported and shared: the file contents it keeps have their credentials redacted, while the
delivered files keep them (the app needs its keys). ``contentHash`` stays the hash of the delivered file."""

import base64
import hashlib
import io
import json
import pathlib
import re
import zipfile

import pytest

from besser.utilities.secret_redaction import REDACTED_SECRET
from besser.utilities.web_modeling_editor.backend.services.converters import class_buml_to_json

TRACED = "/besser_api/traceability/generate-output"
FIXTURES = pathlib.Path(__file__).parents[3] / "generators" / "fixtures"
API_KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
PLAIN_KEY = "q8Zt3vLm9Rk2Wx7Np4Hs"  # no provider prefix: only its YAML key says it is one


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _delivered(body) -> dict:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(body["contentBase64"]))) as archive:
        return {name: archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist()}


def _files(trace) -> dict:
    return {a["path"]: a for a in trace["artifacts"] if a["type"] == "CodeArtifact" and "startLine" not in a}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_django_trace_keeps_settings_without_the_secret_key(client, library_book_author_model):
    response = client.post(TRACED, json={
        "id": "class-diagram", "title": "Library", "model": class_buml_to_json(library_book_author_model),
        "generator": "django", "config": {"project_name": "library_site", "app_name": "books"},
        "actorName": "Ada"})
    assert response.status_code == 200, response.text
    body = response.json()
    settings = _delivered(body)["library_site/settings.py"]
    secret = re.search(r"SECRET_KEY = ['\"]([^'\"]+)['\"]", settings).group(1)

    stored = _files(body["trace"])["library_site/settings.py"]
    assert secret not in json.dumps(body["trace"])
    assert f'SECRET_KEY = "{REDACTED_SECRET}"' in stored["content"] or f"SECRET_KEY = '{REDACTED_SECRET}'" in \
        stored["content"]
    assert stored["contentHash"] == _sha256(settings)  # the delivered file, as it is
    assert stored["content"].count("\n") == settings.count("\n")  # fragments' lines still point at it


def test_an_agent_trace_keeps_the_users_config_without_its_keys(client):
    diagram = json.loads((FIXTURES / "library_template_agent_diagram.json").read_text("utf-8"))
    config = f"nlp:\n  openai:\n    api_key: {API_KEY}\n  other:\n    api_key: {PLAIN_KEY}\n"
    response = client.post(TRACED, json={**diagram, "id": "agent-diagram", "generator": "agent", "config": None,
                                         "configYaml": config, "actorName": "Ada"})
    assert response.status_code == 200, response.text
    body = response.json()
    delivered = _delivered(body)["config.yaml"]
    assert API_KEY in delivered and PLAIN_KEY in delivered

    trace = json.dumps(body["trace"])
    assert API_KEY not in trace and PLAIN_KEY not in trace
    stored = _files(body["trace"])["config.yaml"]
    assert stored["content"].count(REDACTED_SECRET) == 2
    assert stored["contentHash"] == _sha256(delivered)
