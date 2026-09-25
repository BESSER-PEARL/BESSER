r"""A keyword boundary belongs only where the keyword's own edge is a word char.

``\b`` before the ``.`` of ``.net`` requires the PRECEDING character to be a
word character, so ``\b\.net\b`` matched only inside ``asp.net`` and never
after a space. "Build a .NET 8 API" therefore got no idiom guidance at all --
for the one spelling someone added ``.net`` to the keyword list to catch.

The old test decided using the needle with punctuation stripped, then built
the pattern from the unstripped needle: ``.net`` -> "net".isalnum() -> took a
boundary branch it could never satisfy. ``next.js`` took the same branch and
worked only by luck, its leading ``n`` being a word character.

The guidance block this gates exists because a benchmark run found generated
code LESS idiomatic than a naive-LLM baseline in ~80% of scenarios, so losing
it silently for a whole stack is not cosmetic.
"""
import pytest

from besser.spec_driven_agent.planning.stack_metadata import (
    _contains_word,
    detect_idiom_stack,
)


@pytest.mark.parametrize("text, needle", [
    ("Build a .NET 8 minimal API for orders", ".net"),   # the regression
    ("Build an ASP.NET Core API", ".net"),               # kept working
    ("use next.js for the frontend", "next.js"),
    ("write the service in c#", "c#"),
    ("a Rust axum service", "rust"),
    (".net at the very start of the text", ".net"),
])
def test_a_named_stack_is_recognised(text, needle):
    assert _contains_word(text, needle) is True


@pytest.mark.parametrize("text, needle", [
    ("a trustworthy inventory design", "rust"),      # why boundaries exist
    ("the subnetwork routing layer", ".net"),
    ("dotnetting around the problem", "dotnet"),
    ("javascripts everywhere", "javascript"),
    ("gonextjsing is not a word", "next.js"),
])
def test_a_substring_inside_another_word_is_not_a_match(text, needle):
    """Loosening the leading boundary must not turn these into matches."""
    assert _contains_word(text, needle) is False


def test_an_empty_needle_matches_nothing():
    assert _contains_word("anything at all", "") is False


@pytest.mark.parametrize("instructions", [
    "Build a .NET 8 minimal API for orders",
    "Build an ASP.NET Core API",
    "Build a Next.js app",
])
def test_idiom_guidance_is_selected_end_to_end(instructions):
    assert detect_idiom_stack(instructions) is not None


def test_prose_naming_no_stack_still_selects_nothing():
    """The detector must stay quiet rather than guess a stack from a near-miss."""
    assert detect_idiom_stack("Build a trustworthy inventory system") is None
