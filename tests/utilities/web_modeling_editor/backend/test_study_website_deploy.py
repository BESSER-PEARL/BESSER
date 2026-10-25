"""The study deploy keeps the normal website generator and relays its archive."""

import asyncio
import importlib
import io
import json

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import besser.utilities.web_modeling_editor.backend.services.converters  # noqa: F401
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput


study = importlib.import_module(
    "besser.utilities.web_modeling_editor.backend.routers.study_router"
)
generation = importlib.import_module(
    "besser.utilities.web_modeling_editor.backend.routers.generation_router"
)


def request_with_versions(versions):
    project = ProjectInput.model_validate({
        "id": "study", "type": "Project", "name": "Museum",
        "createdAt": "2026-01-01T00:00:00Z",
        "diagrams": {
            "ClassDiagram": [{"title": "Class", "model": {}}],
            "GUINoCodeDiagram": [{"title": "Website", "model": {}}],
        },
    })
    return study.StudyDeployRequest(
        agent_model={}, web_app_project=project, web_app_versions=versions,
    )


def test_study_website_uses_generator_and_returns_base_path(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_URL", raising=False)
    monkeypatch.setenv("STUDY_FRONTEND_INTERNAL_URL", "http://frontend:8080")
    monkeypatch.setenv("STUDY_FRONTEND_PUBLIC_URL", "https://museum.example.org")
    monkeypatch.setenv("STUDY_DEPLOY_TOKEN", "private-token-of-at-least-32-characters")
    captured = {}

    async def fake_generate(project, generator, config):
        captured["config"] = config
        return StreamingResponse(iter([b"generated archive"]))

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_open(post, timeout):
        captured["url"] = post.full_url
        captured["body"] = post.data
        captured["token"] = post.get_header("X-study-deploy-token")
        captured["timeout"] = timeout
        return Reply(json.dumps({"success": True, "versions": ["base", "child"]}).encode())

    monkeypatch.setattr(generation, "_handle_web_app_project_generation", fake_generate)
    monkeypatch.setattr(study.urllib.request, "urlopen", fake_open)
    versions = [
        {"slug": "base", "guiModel": {}},
        {"slug": "child", "guiModel": {}},
    ]
    url = asyncio.run(study._deploy_study_website(request_with_versions(versions)))

    assert url == "https://museum.example.org/study-site/"
    assert captured["config"] == {"webAppVersions": versions}
    assert captured["url"] == "http://frontend:8080/api/study-site"
    assert captured["body"] == b"generated archive"
    assert captured["token"] == "private-token-of-at-least-32-characters"


def test_study_website_rejects_duplicate_slugs(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_URL", raising=False)
    monkeypatch.setenv("STUDY_FRONTEND_INTERNAL_URL", "http://frontend:8080")
    monkeypatch.setenv("STUDY_FRONTEND_PUBLIC_URL", "https://museum.example.org")
    monkeypatch.setenv("STUDY_DEPLOY_TOKEN", "private-token-of-at-least-32-characters")
    request = request_with_versions([
        {"slug": "base", "guiModel": {}},
        {"slug": "base", "guiModel": {}},
    ])
    with pytest.raises(HTTPException, match="unique names"):
        asyncio.run(study._deploy_study_website(request))


def test_study_website_requires_separate_origin(monkeypatch):
    monkeypatch.setenv("STUDY_FRONTEND_INTERNAL_URL", "http://frontend:8080")
    monkeypatch.setenv("STUDY_FRONTEND_PUBLIC_URL", "https://editor.example.org")
    monkeypatch.setenv("DEPLOYMENT_URL", "https://editor.example.org")
    monkeypatch.setenv("STUDY_DEPLOY_TOKEN", "private-token-of-at-least-32-characters")
    request = request_with_versions([{"slug": "base", "guiModel": {}}])
    with pytest.raises(HTTPException, match="distinct from the editor"):
        asyncio.run(study._deploy_study_website(request))
