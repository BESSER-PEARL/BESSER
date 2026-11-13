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


# --- What the workspace walks read (and skip) -------------------------------

def _messages(orch) -> list[str]:
    return [i.message for i in orch._collect_validation_issues()]


def test_a_utf8_bom_or_a_coding_line_is_valid_python(tmp_path):
    """Python reads both; a strict UTF-8 read called the BOM U+FEFF a syntax error
    and crashed Phase 3 with UnicodeDecodeError on the latin-1 file."""
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "bom.py").write_bytes(b"\xef\xbb\xbfx = 1\n")
    (tmp_path / "backend" / "legacy.py").write_bytes(b"# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n")

    orch = _orch(tmp_path)

    assert [m for m in _messages(orch) if m.startswith("Syntax error")] == []
    assert [m for m in orch._validate_phase1_output() if "syntax" in m] == []


def test_an_undecodable_file_is_a_finding_not_a_crash(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"x = '\xff\xfe'\n")

    syntax = [m for m in _messages(_orch(tmp_path)) if m.startswith("Syntax error")]

    assert len(syntax) == 1 and syntax[0].startswith("Syntax error in bad.py line 1:")


def test_installed_dependencies_are_not_the_app(tmp_path):
    """npm and pip install other people's code: a Python 2 script or a
    Dockerfile in a package is not a defect the model can fix, and the check
    must not rewrite files there."""
    vendored_dockerfile = "FROM node:20\nCOPY requirements.txt .\nRUN npm ci\n"
    for vendored in ("frontend/node_modules/somepkg", ".venv/lib/site-packages/pkg"):
        (tmp_path / vendored).mkdir(parents=True)
        (tmp_path / vendored / "gen.py").write_text("print 'python 2'\n", encoding="utf-8")
        (tmp_path / vendored / "Dockerfile").write_text(vendored_dockerfile, encoding="utf-8")
    (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    orch = _orch(tmp_path)

    assert [m for m in _messages(orch) if "node_modules" in m or ".venv" in m] == []
    assert [m for m in orch._validate_phase1_output() if "node_modules" in m or ".venv" in m] == []
    assert (tmp_path / "frontend/node_modules/somepkg/Dockerfile").read_text(
        encoding="utf-8") == vendored_dockerfile


def test_a_rival_framework_inside_a_virtualenv_is_not_a_switch(tmp_path):
    (tmp_path / "backend" / ".venv" / "flask_ext").mkdir(parents=True)
    (tmp_path / "backend" / ".venv" / "flask_ext" / "x.py").write_text(
        "import flask\n", encoding="utf-8")
    orch = _orch(tmp_path)
    orch._generator_used = "generate_fastapi_backend"

    assert orch._collect_framework_switch_issues() == []


def test_an_unparseable_null_byte_does_not_abort_the_import_check(tmp_path, monkeypatch):
    """Python 3.11's ast.parse raises ValueError, not SyntaxError, on a null byte."""
    from besser.spec_driven_agent.validation import python_imports

    (tmp_path / "nul.py").write_text("x = 1\n", encoding="utf-8")

    def parse_like_3_11(source, *args, **kwargs):
        raise ValueError("source code string cannot contain null bytes")

    monkeypatch.setattr(python_imports, "parse_source", parse_like_3_11)
    assert python_imports._unresolvable_local_imports(str(tmp_path)) == []


def test_the_data_contract_sweep_skips_a_virtualenv(tmp_path):
    from besser.spec_driven_agent.validation.contract_checks import _walk_python

    (tmp_path / "backend" / ".venv").mkdir(parents=True)
    (tmp_path / "backend" / ".venv" / "lib.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "backend" / "app.py").write_text("x = 1\n", encoding="utf-8")

    parsed = _walk_python(str(tmp_path))[0]

    assert [rel for rel, *_ in parsed] == ["backend/app.py"]
