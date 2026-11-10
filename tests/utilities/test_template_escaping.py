"""Tests for the string-literal escaping filters shared by generator templates."""

import ast

import pytest
from jinja2 import Environment

from besser.utilities.template_escaping import (
    ESCAPING_FILTERS,
    dart_str,
    hcl_str,
    py_str,
    register_escaping_filters,
    ttl_str,
    yaml_str,
)

PLAIN = "A plain description, with commas: and {braces} 100%"
TRICKY = "it's a \"quote\" \\ back\nnew\rret\ttab ${x} %{y} $name"


@pytest.mark.parametrize("escape", list(ESCAPING_FILTERS.values()))
def test_plain_text_is_unchanged(escape):
    assert escape(PLAIN) == PLAIN


@pytest.mark.parametrize("escape", list(ESCAPING_FILTERS.values()))
def test_none_renders_empty(escape):
    assert escape(None) == ""


def test_py_str_round_trips_through_a_python_literal():
    rendered = f"'{py_str(TRICKY)}'"
    assert "\n" not in rendered
    assert ast.literal_eval(rendered) == TRICKY


def test_py_str_matches_the_replace_chain_it_replaces():
    # The BAF template used replace('\\', '\\\\') | replace("'", "\\'"); for text
    # without line breaks the filter must render the same bytes.
    text = "it's \"quoted\" \\ here"
    assert py_str(text) == text.replace("\\", "\\\\").replace("'", "\\'")


def test_py_str_leaves_braces_for_plain_literals():
    assert py_str("{name}") == "{name}"


def test_ttl_str_escapes_turtle_echars():
    assert ttl_str('a "b" \\ c\nd\re\tf') == 'a \\"b\\" \\\\ c\\nd\\re\\tf'
    assert ttl_str("it's") == "it's"


def test_hcl_str_escapes_quotes_and_template_sequences():
    assert hcl_str('a "b" \\ c\nd') == 'a \\"b\\" \\\\ c\\nd'
    assert hcl_str("${var.x} %{if y}") == "$${var.x} %%{if y}"
    assert hcl_str("$5 and 100%") == "$5 and 100%"


def test_dart_str_escapes_quotes_and_interpolation():
    assert dart_str("it's $name ${x}") == "it\\'s \\$name \\${x}"
    assert dart_str("a\\b\nc") == "a\\\\b\\nc"
    assert dart_str('"double"') == '"double"'


def test_yaml_str_escapes_double_quoted_scalar():
    assert yaml_str('a "b" \\ c\nd') == 'a \\"b\\" \\\\ c\\nd'


def test_register_escaping_filters_adds_every_filter():
    env = register_escaping_filters(Environment())
    assert set(ESCAPING_FILTERS) <= set(env.filters)
    assert env.from_string("'{{ v | py_str }}'").render(v="it's") == "'it\\'s'"
