"""Generator child processes run under BESSER_GENERATOR_SUBPROCESS_TIMEOUT."""

import subprocess
from unittest import mock

import pytest

from besser.generators.django import DjangoGenerator
from besser.generators.settings import DEFAULT_SUBPROCESS_TIMEOUT_SECONDS, subprocess_timeout
from besser.generators.sql import SQLGenerator


def _timeout(cmd, *args, timeout=None, **kwargs):
    raise subprocess.TimeoutExpired(cmd, timeout)


@pytest.mark.parametrize("raw, expected", [
    (None, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
    ("30", 30.0),
    ("2.5", 2.5),
    ("", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
    ("abc", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
    ("0", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
    ("-5", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS),
])
def test_timeout_setting(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("BESSER_GENERATOR_SUBPROCESS_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("BESSER_GENERATOR_SUBPROCESS_TIMEOUT", raw)
    assert subprocess_timeout() == expected


def test_sql_generator_times_out(monkeypatch, library_book_author_model, tmp_path):
    monkeypatch.setenv("BESSER_GENERATOR_SUBPROCESS_TIMEOUT", "7")
    generator = SQLGenerator(model=library_book_author_model, output_dir=str(tmp_path), sql_dialect="sqlite")
    with mock.patch("besser.generators.sql.sql_generator.subprocess.run", side_effect=_timeout) as run:
        with pytest.raises(subprocess.TimeoutExpired) as exc_info:
            generator.generate()
    assert run.call_args.kwargs["timeout"] == 7.0
    assert exc_info.value.timeout == 7.0
    assert not (tmp_path / "tables_sqlite.sql").exists()


def test_django_generator_times_out(monkeypatch, library_book_author_model, tmp_path):
    monkeypatch.setenv("BESSER_GENERATOR_SUBPROCESS_TIMEOUT", "9")
    generator = DjangoGenerator(
        model=library_book_author_model, project_name="proj", app_name="app",
        containerization=False, output_dir=str(tmp_path),
    )
    with mock.patch("besser.generators.django.django_generator.subprocess.run", side_effect=_timeout) as run:
        with pytest.raises(subprocess.TimeoutExpired):
            generator.generate()
    assert run.call_args.kwargs["timeout"] == 9.0
