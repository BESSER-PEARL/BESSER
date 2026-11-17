"""The checks the GUI converter runs on text of the user's design take linear time.

They were regular expressions. The placeholder search was quadratic on a long
run of braces (1.2 s at 20,000 characters); a design is user input, so all three
are plain string checks now.
"""
import time

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors import (
    processor,
)


@pytest.mark.parametrize("href, expected", [
    ("https://example.org/orders", True),
    ("mailto:someone@example.org", True),
    ("  tel:+352123", True),
    ("//cdn.example.org/a.css", True),
    ("#top", True),
    ("orders", False),
    ("/orders/new", False),
    ("order list", False),
    ("1st:page", False),            # a scheme starts with a letter
    ("café:menu", False),      # and is ASCII
    ("a b:c", False),
    ("", False),
])
def test_an_href_with_a_scheme_or_fragment_is_not_a_page(href, expected):
    assert processor._is_not_a_page_href(href) is expected


@pytest.mark.parametrize("text, expected", [
    ("{{ total }}", True),
    ("Total: {{total}} items", True),
    ("{{}}", True),
    ("{{{}}", True),
    ("{{ open\n}}", False),         # a placeholder stays on one line
    ("}}{{", False),
    ("{ total }", False),
    ("128", False),
])
def test_a_template_placeholder_is_found_on_one_line(text, expected):
    assert processor._has_template_placeholder(text) is expected


@pytest.mark.parametrize("route, expected", [
    ("/orders", True),
    ("/orders/new-order", True),
    ("/résumé_1", True),
    ("orders", False),
    ("/", False),
    ("/orders/", False),
    ("//orders", False),
    ("/orders?x=1", False),
    ("/orders list", False),
    ("", False),
])
def test_a_stored_route_is_kept_only_when_it_is_url_safe(route, expected):
    assert processor._is_stored_route(route) is expected


def test_long_runs_in_design_text_are_checked_in_linear_time():
    started = time.perf_counter()
    assert processor._is_not_a_page_href(" " * 400_000 + "x") is False
    assert processor._has_template_placeholder("{{" * 200_000) is False
    assert processor._is_stored_route("/" + "-" * 400_000 + "!") is False
    # The old placeholder pattern needed minutes on its input.
    assert time.perf_counter() - started < 5
