"""Unit tests for ``services.utils.agent_config_manual_mapping_utils``.

Covers representative profile->config mappings (age threshold, language detection,
abbreviation/text-keyword rule), the merge_dicts re-export semantics, determinism,
and the structure of the returned recommendation.
"""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_manual_mapping_utils import (
    LANGUAGE_CODE_TO_NAME,
    build_manual_mapping_recommendation,
    get_manual_agent_config_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
    load_default_agent_recommendation_config,
    merge_dicts,
)


@pytest.fixture(autouse=True)
def _clear_default_config_cache():
    load_default_agent_recommendation_config.cache_clear()
    yield
    load_default_agent_recommendation_config.cache_clear()


# ---------------------------------------------------------------------------
# Rule: older adults (age threshold + readability adjustments)
# ---------------------------------------------------------------------------


def test_age_threshold_triggers_older_adults_rule():
    profile = {"User": {"age": 72}}
    result = build_manual_mapping_recommendation(profile, user_profile_name="grandma")

    rule_ids = {entry["id"] for entry in result["matchedRules"]}
    assert "older_adults_readability" in rule_ids

    cfg = result["config"]
    # The older_adults rule sets formal style, simple complexity, large font, high contrast.
    p = cfg["presentation"]
    assert p["agentStyle"] == "formal"
    assert p["languageComplexity"] == "simple"
    assert p["sentenceLength"] == "concise"
    assert p["interfaceStyle"]["size"] == 20
    assert p["interfaceStyle"]["contrast"] == "high"
    assert p["useAbbreviations"] is False


def test_age_threshold_not_triggered_for_young_adult():
    profile = {"User": {"age": 30}}
    result = build_manual_mapping_recommendation(profile)
    rule_ids = {entry["id"] for entry in result["matchedRules"]}
    assert "older_adults_readability" not in rule_ids


# ---------------------------------------------------------------------------
# Rule: language detection (single-language localization via LANGUAGE_CODE_TO_NAME)
# ---------------------------------------------------------------------------


def test_iso_language_code_maps_to_agent_language():
    """A single ISO 639-1 code in a language field should be mapped via
    ``LANGUAGE_CODE_TO_NAME`` and trigger the ``single_language_localization``
    rule, which sets ``agentLanguage`` to the detected primary language."""
    # Sanity check the mapping table is what we rely on.
    assert LANGUAGE_CODE_TO_NAME["fr"] == "french"

    profile = {"User": {"language": "fr"}}
    result = build_manual_mapping_recommendation(profile)
    rule_ids = {entry["id"] for entry in result["matchedRules"]}
    assert "single_language_localization" in rule_ids
    assert result["signals"]["detectedLanguages"] == ["french"]
    assert result["config"]["presentation"]["agentLanguage"] == "french"


def test_multilingual_user_triggers_code_switching_rule():
    profile = {"User": {"languages": ["en", "fr"]}}
    result = build_manual_mapping_recommendation(profile)
    rule_ids = {entry["id"] for entry in result["matchedRules"]}
    assert "multilingual_code_switching" in rule_ids
    # The multilingual rule keeps agentLanguage as "original"
    assert result["config"]["presentation"]["agentLanguage"] == "original"
    assert result["signals"]["isMultilingual"] is True


# ---------------------------------------------------------------------------
# Rule: profile-text keyword rule (abbreviation / low-literacy via text match)
# ---------------------------------------------------------------------------


def test_low_literacy_keyword_disables_abbreviations_and_enlarges_font():
    profile = {"User": {"description": "User has low literacy and limited reading skills"}}
    result = build_manual_mapping_recommendation(profile)
    rule_ids = {entry["id"] for entry in result["matchedRules"]}
    assert "low_literacy_accessibility" in rule_ids

    p = result["config"]["presentation"]
    assert p["useAbbreviations"] is False
    assert p["interfaceStyle"]["size"] == 22
    assert p["interfaceStyle"]["contrast"] == "high"
    # Low-literacy rule also enables speech modality.
    assert "speech" in result["config"]["modality"]["inputModalities"]
    assert "speech" in result["config"]["modality"]["outputModalities"]


# ---------------------------------------------------------------------------
# merge_dicts (re-exported via the same package)
# ---------------------------------------------------------------------------


def test_merge_dicts_base_values_preserved_when_not_overridden():
    base = {"a": 1, "nested": {"b": 2, "c": 3}}
    override = {"nested": {"c": 99}}
    out = merge_dicts(base, override)
    assert out == {"a": 1, "nested": {"b": 2, "c": 99}}


def test_merge_dicts_nested_merge_does_not_mutate_inputs():
    base = {"x": {"y": 1}}
    override = {"x": {"z": 2}}
    out = merge_dicts(base, override)
    assert out == {"x": {"y": 1, "z": 2}}
    assert base == {"x": {"y": 1}}
    assert override == {"x": {"z": 2}}


# ---------------------------------------------------------------------------
# Determinism + structural shape
# ---------------------------------------------------------------------------


def test_recommendation_is_deterministic_for_identical_input():
    profile = {"User": {"age": 70, "language": "de"}}
    a = build_manual_mapping_recommendation(profile, user_profile_name="user")
    b = build_manual_mapping_recommendation(profile, user_profile_name="user")
    assert a == b


def test_recommendation_contains_all_expected_top_level_sections():
    profile = {"User": {"age": 35}}
    result = build_manual_mapping_recommendation(profile)
    assert set(result.keys()) >= {"config", "matchedRules", "signals"}

    cfg = result["config"]
    for section in ("presentation", "modality", "behavior", "content", "system"):
        assert section in cfg, f"missing config section {section!r}"


def test_user_profile_name_propagates_into_content_section():
    profile = {"User": {}}
    result = build_manual_mapping_recommendation(profile, user_profile_name="alice")
    assert result["config"]["content"]["userProfileName"] == "alice"
    assert result["config"]["content"]["adaptContentToUserProfile"] is True


# ---------------------------------------------------------------------------
# get_manual_agent_config_mapping — metadata / shape
# ---------------------------------------------------------------------------


def test_get_manual_agent_config_mapping_exposes_rules_and_metadata():
    mapping = get_manual_agent_config_mapping()
    assert "version" in mapping
    assert "sourceDocument" in mapping
    assert "allowedValues" in mapping
    assert isinstance(mapping["rules"], list) and mapping["rules"]
    # Mutating the returned mapping must not affect the next call (deepcopy).
    mapping["rules"].clear()
    again = get_manual_agent_config_mapping()
    assert again["rules"], "expected fresh rules list on subsequent call"
