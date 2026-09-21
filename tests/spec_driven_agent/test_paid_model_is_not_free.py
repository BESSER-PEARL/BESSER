"""A paid model priced at $0 silently disables the cost cap.

``_is_free_local_model`` decides both the pricing tier and, at
``orchestrator.py:587``, whether ``protect_scaffold`` is on. Its own
docstring warns that "a false positive prices a paid model at $0 and
SILENTLY DISABLES the cost cap" - and it was doing exactly that for four
Mistral API models, including ``mistral-small-latest``, which is the
Mistral planning model and therefore ran uncapped on every Mistral run.

The cause is that Mistral's paid ids are tagless and un-namespaced, so
they fell past the Ollama rule and the gateway rule to the family marker
list, where "mistral-small", "codestral" and "devstral" matched them.

The discriminator is the separator, not the family: Ollama writes
``mistral-small:latest`` with a colon, Mistral's API writes
``mistral-small-latest`` with a hyphen.
"""

import pytest

from besser.spec_driven_agent.llm_client import _get_pricing, _is_free_local_model


PAID_MISTRAL_IDS = [
    "mistral-small-latest",
    "codestral-latest",
    "devstral-small-latest",
    "devstral-medium-latest",
]


@pytest.mark.parametrize("model_id", PAID_MISTRAL_IDS)
def test_a_paid_mistral_api_id_is_not_free(model_id):
    assert _is_free_local_model(model_id) is False


@pytest.mark.parametrize("model_id", PAID_MISTRAL_IDS)
def test_a_paid_mistral_api_id_is_not_priced_at_zero(model_id):
    """$0 is what disables the cap, so assert the price, not just the flag."""
    pricing = _get_pricing(model_id)

    assert pricing["input"] > 0, f"{model_id} priced at $0 - the cost cap is off"
    assert pricing["output"] > 0


@pytest.mark.parametrize("model_id", [
    "mistral-small:latest",   # Ollama writes the tag with a colon
    "codestral:22b",
    "devstral:latest",
])
def test_a_self_hosted_tag_is_still_free(model_id):
    """Rule 2 catches these before the new rule can see them."""
    assert _is_free_local_model(model_id) is True


@pytest.mark.parametrize("model_id", ["mistral-small", "codestral", "devstral"])
def test_a_tagless_family_name_is_still_free(model_id):
    """The family list exists for tagless self-hosted ids; keep it working."""
    assert _is_free_local_model(model_id) is True


def test_a_free_tier_alias_ending_in_latest_is_still_free():
    """``-free`` is checked first, so the new rule cannot override it."""
    assert _is_free_local_model("something-latest-free") is True


@pytest.mark.parametrize("model_id", [
    "qwen3-coder:30b",
    "llama3:8b",
    "meituan/longcat-2.0:free",
])
def test_the_other_free_paths_are_untouched(model_id):
    assert _is_free_local_model(model_id) is True
