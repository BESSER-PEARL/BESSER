"""Deterministic scaffold repair: restore a requirements.txt the LLM dropped.

The deterministic backend generator always writes a correct requirements.txt,
but a weak Phase-2 model sometimes deletes it while customizing, leaving a
Dockerfile that ``COPY requirements.txt``s a file that no longer exists (a hard
image-build blocker the weak model then can't self-repair). The orchestrator
restores it deterministically instead of relying on the LLM fix loop.
"""
import os

from besser.generators.llm.orchestrator import _ensure_requirements_txt


def test_writes_base_stack_when_missing(tmp_path):
    d = str(tmp_path)
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\nCOPY requirements.txt .\n"
        "RUN pip install -r requirements.txt\n"
    )
    assert _ensure_requirements_txt(d) is True
    content = (tmp_path / "requirements.txt").read_text()
    for base in ("fastapi", "uvicorn", "pydantic", "sqlalchemy"):
        assert base in content, f"missing base dep {base}"


def test_infers_extras_from_imports(tmp_path):
    d = str(tmp_path)
    (tmp_path / "auth.py").write_text("from jose import jwt\nimport passlib\n")
    assert _ensure_requirements_txt(d) is True
    content = (tmp_path / "requirements.txt").read_text()
    assert "python-jose" in content
    assert "passlib" in content


def test_no_op_when_already_present(tmp_path):
    d = str(tmp_path)
    (tmp_path / "requirements.txt").write_text("flask==3.0.0\n")
    # Must NOT overwrite an existing file.
    assert _ensure_requirements_txt(d) is False
    assert (tmp_path / "requirements.txt").read_text() == "flask==3.0.0\n"
