"""``besser model``: change an element in place (an attribute, an association end, a class's parents),
and the checks that keep a change from being accepted wrong (defaults, enumeration literals in OCL).

Each in-place change is checked on the file's text (only the changed element's lines differ) and where
it shows: ``besser validate --ocl`` passes and the generated backend imports and applies it.
"""

import difflib
import json
import os
import subprocess
import sys
import textwrap

import pytest

STARTER_BOOK = (
    'book = Class(name="Book", attributes={\n'
    '    Property(name="title", type=StringType),\n'
    '    Property(name="pages", type=IntegerType),\n'
    '    Property(name="release", type=DateType),\n'
    '})\n'
)


def _changed_lines(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines removed from ``before`` and those added in ``after``."""
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    return [line[2:] for line in diff if line.startswith("- ")], [line[2:] for line in diff if line.startswith("+ ")]


@pytest.fixture
def project(besser, tmp_path):
    """A project of 'besser init' (the starter's Library, Book and Author); returns a reader of model.py."""
    assert besser("init").returncode == 0
    return lambda: (tmp_path / "model.py").read_text(encoding="utf-8")


@pytest.fixture
def change(besser):
    """Run ``besser model ARGS``; check it succeeded with one line and return that line."""
    def run(*args, stdin=""):
        result = besser("model", *args, stdin=stdin, output=None)
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(result.stdout.splitlines()) == 1, result.stdout
        return result.stdout.strip()
    return run


@pytest.fixture
def refused(besser, tmp_path):
    """Run ``besser model ARGS``; check it was refused (exit 2) and wrote nothing; return its message."""
    def run(*args, stdin=""):
        before = (tmp_path / "model.py").read_bytes()
        result = besser("model", *args, stdin=stdin)
        assert result.returncode == 2, result.stdout + result.stderr
        assert (tmp_path / "model.py").read_bytes() == before
        return result.stderr
    return run


# ── defaults are literals of their attribute's type ─────────────────────────


@pytest.mark.parametrize("attribute, message", [
    # Accepted before, and the backend then wrote default='now' on a Date column.
    ("created:date=now", "default 'now': a date attribute takes no default. The generated backend writes a "
                         "default as text, which its date column refuses on insert, and has no 'now'. Leave it "
                         "without one, make it optional (date?) or name it created_at or updated_at (no default)"),
    ("created:date=2024-01-31", "a date attribute takes no default"),
    ("opened:datetime=2024-01-31T10:00", "a datetime attribute takes no default"),
    ("at:time=10:00", "a time attribute takes no default. "),
    ("copies:int=1.5", "default '1.5' is not an int (a whole number, e.g. 0)"),  # accepted before
    ("copies:int=true", "default 'true' is not an int"),
    ("price:float=cheap", "default 'cheap' is not a number (e.g. 0.5)"),
    ("lent:bool=1", "default '1' is not true or false"),
])
def test_a_default_that_is_no_literal_of_its_type_is_refused(project, refused, attribute, message):
    assert message in refused("add-attribute", "Book", attribute)


def test_a_default_of_an_enumeration_names_its_literals(project, change, refused):
    change("set-enum", "Genre", "NOVEL", "POEM")

    assert "default 'ESSAY' is not a literal of Genre (NOVEL, POEM)" in refused("add-attribute", "Book",
                                                                                 "genre:Genre=ESSAY")


def test_the_defaults_the_backend_applies_are_accepted(project, change):
    line = change("add-attribute", "Book", "copies:int=1", "price:float=2", "lent:bool=false", "note:str=now",
                  "created_at:datetime")

    assert line == ("model.py: added Book.copies:int=1, price:float=2, lent:bool=false, note:str=now, "
                    "created_at:datetime")
