"""The generated backend's error handlers log the traceback they hide.

Production run 438889bc: a bcrypt/passlib mismatch raised "ValueError: password
cannot be longer than 72 bytes" inside the register route. The global
ValueError handler turned it into a 400 and logged nothing, so neither the
server log nor the probe's log showed the cause, and the model rewrote
password hashing trying to satisfy the check.
"""
import ast
import asyncio
import logging

import pytest

from besser.generators.backend import BackendGenerator

fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def main_api(tmp_path, library_book_author_model):
    BackendGenerator(library_book_author_model, output_dir=str(tmp_path)).generate()
    path = next(tmp_path.rglob("main_api.py"))
    return path.read_text(encoding="utf-8")


def _handler(src, name, logger):
    """Compile one handler from the generated module, bound to ``logger``."""
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    node.decorator_list = []
    from fastapi import Request, status
    from fastapi.responses import JSONResponse
    namespace = {"logger": logger, "JSONResponse": JSONResponse, "status": status,
                 "Request": Request, "ValueError": ValueError}
    from sqlalchemy.exc import SQLAlchemyError
    namespace["SQLAlchemyError"] = SQLAlchemyError
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main_api.py", "exec"), namespace)
    return namespace[name]


class _Request:
    method = "POST"

    class url:
        path = "/auth/register"


def _raise_and_handle(handler, exc):
    try:
        raise exc
    except type(exc) as caught:
        return asyncio.run(handler(_Request(), caught))


def test_value_error_handler_logs_the_traceback(main_api, caplog):
    logger = logging.getLogger("generated.main_api")
    handler = _handler(main_api, "value_error_handler", logger)
    with caplog.at_level(logging.ERROR, logger="generated.main_api"):
        response = _raise_and_handle(handler, ValueError("password cannot be longer than 72 bytes"))

    records = [r for r in caplog.records if r.exc_info]
    assert records, "the ValueError handler logged no traceback"
    assert "password cannot be longer than 72 bytes" in caplog.text
    assert "Traceback" in caplog.text
    # The client contract is unchanged.
    assert response.status_code == 400
    assert b'"detail":"Invalid input data provided"' in response.body


def test_sqlalchemy_error_handler_logs_the_traceback(main_api, caplog):
    from sqlalchemy.exc import SQLAlchemyError
    logger = logging.getLogger("generated.main_api")
    handler = _handler(main_api, "sqlalchemy_error_handler", logger)
    with caplog.at_level(logging.ERROR, logger="generated.main_api"):
        response = _raise_and_handle(handler, SQLAlchemyError("no such column: user.hash"))

    assert any(r.exc_info for r in caplog.records), "the SQLAlchemyError handler logged no traceback"
    assert response.status_code == 500
