"""``clean_attribute_name`` / ``clean_method_name``: the GUI converter's name cleaners.

Both read names from a request body (a GUI diagram's class attributes and
methods). CodeQL alerts 26, 42 and 43 (py/polynomial-redos): their regexes
``\\s*:\\s*.*$`` and ``\\s*\\(.*$`` took quadratic time on a run of spaces with no
':' or '(' after it - 100,000 spaces took several seconds per name.
"""
import time

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.utils import (
    clean_attribute_name,
    clean_method_name,
)

# Far below what the old regexes needed on this input, far above what a linear scan needs.
_LINEAR_BUDGET_SECONDS = 1.0
_SPACES = " " * 100_000


@pytest.mark.parametrize("raw, expected", [
    ("+ name: str", "name"),
    ("- price : float", "price"),
    ("# id:int", "id"),
    ("title", "title"),
    ("  padded  ", "padded"),
    ("", ""),
])
def test_attribute_names_lose_visibility_and_type(raw, expected):
    assert clean_attribute_name(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("+ decrease_stock(qty: int)", "decrease_stock"),
    ("- total(): float", "total"),
    ("# reset ()", "reset"),
    ("summary: str", "summary"),
    ("run", "run"),
    ("", ""),
    (None, None),
])
def test_method_names_lose_visibility_parameters_and_type(raw, expected):
    assert clean_method_name(raw) == expected


@pytest.mark.parametrize("clean", [clean_attribute_name, clean_method_name])
@pytest.mark.parametrize("raw", [f"a{_SPACES}x", f":{_SPACES}x", f"a{_SPACES}(x"])
def test_a_long_run_of_spaces_is_cleaned_in_linear_time(clean, raw):
    started = time.perf_counter()
    clean(raw)
    assert time.perf_counter() - started < _LINEAR_BUDGET_SECONDS
