"""The docs gate: an unreachable external inventory (docs.python.org returned
503 during the v8.0.1 release) must not fail the build; any other warning must."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "docs" / "check_docs.py"
pytest.importorskip("sphinx")
_spec = importlib.util.spec_from_file_location("check_docs", _PATH)
check_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_docs)

OUTAGE = (
    "WARNING: failed to reach any of the inventories with the following issues:\n"
    "intersphinx inventory 'https://docs.python.org/3/objects.inv' not fetchable due to "
    "<class 'requests.exceptions.HTTPError'>: 503 Server Error\n"
)


def test_an_inventory_outage_is_tolerated():
    assert check_docs.unexpected_warnings(OUTAGE) == []


def test_other_warnings_still_fail_the_build():
    text = OUTAGE + "docs/source/x.rst:3: WARNING: undefined label: 'missing'\n"
    assert check_docs.unexpected_warnings(text) == ["docs/source/x.rst:3: WARNING: undefined label: 'missing'"]


def test_errors_still_fail_the_build():
    assert check_docs.unexpected_warnings("x.rst:1: ERROR: Unknown directive type\n")
