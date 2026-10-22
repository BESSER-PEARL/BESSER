"""``DjangoGenerator.generate`` must not crash on a non-UTF-8 console.

On Windows a redirected stdout is cp1252. The status messages used to start
with emoji, so a successful generation raised ``UnicodeEncodeError`` on
'\u2705', and a failed one replaced the real error with a second
``UnicodeEncodeError`` on '\u274c' raised from inside the except handler.
The status messages now go through ``logging`` rather than ``print``.
"""
import io
import logging
import shutil
import subprocess
import sys

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.generators.django import DjangoGenerator


def _model():
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="title", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="M", types={book})


def _cp1252_stdout(monkeypatch):
    # Set inside the test body: pytest's capture re-assigns sys.stdout after fixture setup.
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    return stream


def test_generation_succeeds_on_a_cp1252_console(tmp_path, monkeypatch, caplog):
    pytest.importorskip("django")
    if shutil.which("django-admin") is None:
        pytest.skip("django-admin is not on PATH (activate the virtual environment)")
    _cp1252_stdout(monkeypatch)
    with caplog.at_level(logging.INFO, logger="besser.generators.django.django_generator"):
        DjangoGenerator(model=_model(), project_name="myproject", app_name="app",
                        output_dir=str(tmp_path)).generate()

    assert (tmp_path / "myproject" / "app" / "models.py").is_file()
    assert "completed successfully" in caplog.text


def test_generation_failure_keeps_the_original_error(tmp_path, monkeypatch):
    _cp1252_stdout(monkeypatch)

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", _fail)
    with pytest.raises(subprocess.CalledProcessError):
        DjangoGenerator(model=_model(), project_name="myproject", app_name="app",
                        output_dir=str(tmp_path)).generate()
