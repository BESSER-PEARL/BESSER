"""Dockerfile COPY checks resolve sources against the build context.

The Phase 3 check resolved ``package.json`` / ``requirements.txt`` beside the
Dockerfile. A root Dockerfile with ``COPY frontend/package*.json ./`` - a correct
build - was reported as a blocker, and a Dockerfile naming a requirements.txt
it could not see got a default FastAPI requirements.txt written beside it by
what was supposed to be a check.
"""

from __future__ import annotations

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.docker_context import (
    compose_build_contexts,
    dockerfile_copy_issues,
)


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")


def _orch(tmp_path) -> LLMOrchestrator:
    cls = Class(name="Book")
    cls.attributes = {Property(name="title", type=PrimitiveDataType("str"), is_id=True)}
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="App", types={cls}),
        output_dir=str(tmp_path), enable_checkpointing=False, enable_tracing=False,
        enable_toolchain_validation=False,
    )


def _docker_messages(orch) -> list[str]:
    return [i.message for i in orch._collect_validation_issues()
            if "Dockerfile" in i.message and "exist" in i.message]


def test_root_dockerfile_copying_from_a_subdirectory_is_not_a_blocker(tmp_path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Dockerfile.frontend").write_text(
        "FROM node:20\nWORKDIR /app\nCOPY frontend/package*.json ./\n"
        "RUN npm install\nCOPY frontend/ ./\n", encoding="utf-8")

    assert _docker_messages(_orch(tmp_path)) == []


def test_a_missing_requirements_txt_is_reported_not_written(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "Dockerfile").write_text(
        "FROM python:3.11-slim\nCOPY requirements.txt .\n"
        "RUN pip install -r requirements.txt\n", encoding="utf-8")

    messages = _docker_messages(_orch(tmp_path))

    assert not (tmp_path / "backend" / "requirements.txt").exists()
    assert messages == [
        "backend/Dockerfile references backend/requirements.txt but it doesn't exist"]


def test_compose_build_context_is_the_resolution_root(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "docker").mkdir()
    dockerfile = tmp_path / "docker" / "backend.Dockerfile"
    content = "FROM python:3.11\nCOPY backend/requirements.txt .\n"
    dockerfile.write_text(content, encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  api:\n    build:\n      context: .\n"
        "      dockerfile: docker/backend.Dockerfile\n", encoding="utf-8")

    contexts = compose_build_contexts(str(tmp_path))
    assert dockerfile_copy_issues(str(tmp_path), str(dockerfile), content, contexts) == []
    # Without the compose file the Dockerfile's own directory is the context.
    assert dockerfile_copy_issues(str(tmp_path), str(dockerfile), content, {}) == [
        "docker/backend.Dockerfile references docker/backend/requirements.txt "
        "but it doesn't exist"]


def test_copy_from_a_build_stage_does_not_read_the_context(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    content = (
        "FROM node:20 AS build\nCOPY --from=deps /app/package.json ./\n"
        "COPY --chown=node:node [\"web/package.json\", \"./\"]\n"
    )
    dockerfile.write_text(content, encoding="utf-8")

    assert dockerfile_copy_issues(str(tmp_path), str(dockerfile), content, {}) == [
        "Dockerfile references web/package.json but it doesn't exist"]


def test_a_fastapi_scaffold_gets_its_deleted_requirements_back_before_validation(tmp_path):
    """Phase 1 always writes requirements.txt; a Phase 2 edit that deletes it
    is repaired before validation, at the path the COPY resolves to."""
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "main.py").write_text("from jose import jwt\n", encoding="utf-8")
    (tmp_path / "Dockerfile.backend").write_text(
        "FROM python:3.11-slim\nCOPY backend/requirements.txt .\n", encoding="utf-8")
    orch = _orch(tmp_path)
    orch._generator_used = "generate_fastapi_backend"

    assert _docker_messages(orch) == []
    restored = (tmp_path / "backend" / "requirements.txt").read_text(encoding="utf-8")
    assert "fastapi" in restored and "python-jose" in restored
    assert not (tmp_path / "requirements.txt").exists()


def test_other_scaffolds_do_not_get_a_fastapi_requirements_file(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\nCOPY requirements.txt .\n", encoding="utf-8")
    orch = _orch(tmp_path)
    orch._generator_used = "generate_django"

    assert _docker_messages(orch) == [
        "Dockerfile references requirements.txt but it doesn't exist"]
    assert not (tmp_path / "requirements.txt").exists()


def test_the_lockfile_strip_keeps_the_npm_ci_fix(tmp_path):
    """The strip started from the pre-fix text and wrote ``npm ci`` back."""
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM node:20\nCOPY package.json package-lock.json ./\nRUN npm ci\n", encoding="utf-8")

    _orch(tmp_path)._collect_validation_issues()

    content = dockerfile.read_text(encoding="utf-8")
    assert "npm ci" not in content and "RUN npm install" in content
    assert "package-lock.json" not in content
