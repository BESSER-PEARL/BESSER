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
    detect_stack,
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


# -- the keyword fallback (no routing answer) ---------------------------------
# Without a routing answer the request's words pick the Phase 0.5 template and
# the idiom block. "cargo" and "rocket" are more often the domain than the
# build tool or the framework, so they no longer count on their own.

@pytest.mark.parametrize("instructions", [
    "Build a REST API in Go for a cargo tracking company.",
    "Build a NestJS backend for a model rocket club: members, launches, rockets.",
    "A web app for a rocket launch schedule.",
    "Cargo ship manifest web app.",
    "A web app to plan the spring semester courses.",
    "Let's go: a web app for a bakery.",
    "A web app for Swift Logistics to track parcels.",
    "A web app to score dart tournaments.",
    "A web app for a courier company: standard or express delivery.",
    "Go-kart track booking web app.",
])
def test_a_domain_word_names_no_template(instructions):
    assert detect_stack(instructions) is None


@pytest.mark.parametrize("instructions", [
    "Build a REST API in Go for a cargo tracking company.",
    "A web app for a rocket launch schedule.",
])
def test_a_domain_word_selects_no_rust_idioms(instructions):
    assert detect_idiom_stack(instructions) != "rust_axum"


@pytest.mark.parametrize("instructions", [
    "Build a Rust web server using Axum",
    "Use actix for the API",
    "rocket + serde",
    "Rocket.rs with tokio",
    "Standard Cargo.toml layout please",
])
def test_rust_and_its_crates_still_name_rust(instructions):
    assert detect_stack(instructions) == "rust"
    assert detect_idiom_stack(instructions) == "rust_axum"
