"""Unit tests for ``services.utils.agent_config_recommendation_utils``.

Covers:
 - extract_json_object: plain JSON, fenced JSON, prose-wrapped JSON, malformed input.
 - normalize_recommended_agent_config: numeric clamps, enum whitelist coercion,
   missing keys filled from defaults, modality normalization.
 - load_default_agent_recommendation_config: returns a dict with the documented
   top-level sections; ``cache_clear()`` is invoked between tests so cache state
   does not leak.
 - merge_dicts: recursive merge without mutating inputs.
"""

from __future__ import annotations

import pytest

from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    extract_json_object,
    load_default_agent_recommendation_config,
    merge_dicts,
    normalize_recommended_agent_config,
)


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_plain_json():
    out = extract_json_object('{"a": 1, "b": [2, 3]}')
    assert out == {"a": 1, "b": [2, 3]}


def test_extract_json_object_strips_markdown_fence():
    raw = '```json\n{"k": "v"}\n```'
    assert extract_json_object(raw) == {"k": "v"}


def test_extract_json_object_strips_unlabeled_fence():
    raw = '```\n{"k": 1}\n```'
    assert extract_json_object(raw) == {"k": 1}


def test_extract_json_object_with_leading_and_trailing_prose():
    raw = 'Sure, here you go:\n{"hello": "world"}\nLet me know!'
    assert extract_json_object(raw) == {"hello": "world"}


def test_extract_json_object_empty_input_raises():
    with pytest.raises(ValueError):
        extract_json_object("")
    with pytest.raises(ValueError):
        extract_json_object("   \n  ")


def test_extract_json_object_no_json_raises():
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")


def test_extract_json_object_malformed_json_raises():
    # Has braces but invalid JSON inside — json.loads should raise.
    with pytest.raises(Exception):
        extract_json_object("{not: valid}")


def test_extract_json_object_rejects_root_array():
    # The function requires a JSON object root.
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


# ---------------------------------------------------------------------------
# merge_dicts
# ---------------------------------------------------------------------------


def test_merge_dicts_override_wins_for_scalars():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = merge_dicts(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_merge_dicts_recurses_into_nested_dicts():
    base = {"outer": {"keep": 1, "replace": 2}}
    override = {"outer": {"replace": 99, "added": 3}}
    result = merge_dicts(base, override)
    assert result == {"outer": {"keep": 1, "replace": 99, "added": 3}}


def test_merge_dicts_does_not_mutate_inputs():
    base = {"x": {"y": 1}}
    override = {"x": {"y": 2}}
    merge_dicts(base, override)
    assert base == {"x": {"y": 1}}
    assert override == {"x": {"y": 2}}


def test_merge_dicts_handles_none_override():
    base = {"a": 1}
    assert merge_dicts(base, None) == {"a": 1}  # type: ignore[arg-type]


def test_merge_dicts_replaces_dict_with_non_dict():
    base = {"a": {"nested": 1}}
    override = {"a": "scalar"}
    assert merge_dicts(base, override) == {"a": "scalar"}


# ---------------------------------------------------------------------------
# load_default_agent_recommendation_config
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_default_config_cache():
    """The default config loader is ``lru_cache``-wrapped — clear between tests
    so a test that mutates the loaded dict (which could happen via merge_dicts
    sharing references) cannot leak into the next."""
    load_default_agent_recommendation_config.cache_clear()
    yield
    load_default_agent_recommendation_config.cache_clear()


def test_load_default_agent_recommendation_config_returns_expected_sections():
    cfg = load_default_agent_recommendation_config()
    assert isinstance(cfg, dict)
    for section in ("presentation", "modality", "behavior", "content", "system"):
        assert section in cfg, f"missing section {section!r}"

    presentation = cfg["presentation"]
    assert "interfaceStyle" in presentation
    assert "voiceStyle" in presentation
    assert "agentLanguage" in presentation


def test_load_default_agent_recommendation_config_returns_consistent_shape():
    """The function used to be lru_cache'd — that was dropped to allow
    hot-edits to default_config.json. Two calls must still produce
    equal dicts (the file hasn't changed between calls)."""
    a = load_default_agent_recommendation_config()
    b = load_default_agent_recommendation_config()
    assert a == b


# ---------------------------------------------------------------------------
# normalize_recommended_agent_config
# ---------------------------------------------------------------------------


def test_normalize_clamps_font_size_above_max():
    raw = {"presentation": {"interfaceStyle": {"size": 99}}}
    out = normalize_recommended_agent_config(raw, None)
    assert out["presentation"]["interfaceStyle"]["size"] == 32  # max bound


def test_normalize_clamps_font_size_below_min():
    raw = {"presentation": {"interfaceStyle": {"size": 0}}}
    out = normalize_recommended_agent_config(raw, None)
    assert out["presentation"]["interfaceStyle"]["size"] == 10  # min bound


def test_normalize_clamps_line_spacing():
    raw = {"presentation": {"interfaceStyle": {"lineSpacing": 50}}}
    out = normalize_recommended_agent_config(raw, None)
    assert out["presentation"]["interfaceStyle"]["lineSpacing"] == 3


def test_normalize_clamps_voice_speed():
    raw = {"presentation": {"voiceStyle": {"speed": 9.5}}}
    out = normalize_recommended_agent_config(raw, None)
    assert out["presentation"]["voiceStyle"]["speed"] == 2.0  # max
    raw_low = {"presentation": {"voiceStyle": {"speed": -1}}}
    out_low = normalize_recommended_agent_config(raw_low, None)
    assert out_low["presentation"]["voiceStyle"]["speed"] == 0.5  # min


def test_normalize_invalid_enum_falls_back_to_default():
    raw = {
        "presentation": {
            "agentLanguage": "klingon",      # not in allow-list
            "agentStyle": "INVALID",
            "interfaceStyle": {
                "font": "comic-sans",
                "alignment": "diagonal",
                "contrast": "neon",
            },
            "voiceStyle": {"gender": "robot"},
        },
        "system": {"agentPlatform": "carrier-pigeon"},
    }
    out = normalize_recommended_agent_config(raw, None)
    p = out["presentation"]
    assert p["agentLanguage"] in RECOMMENDATION_ALLOWED_VALUES["agentLanguage"]
    assert p["agentStyle"] in RECOMMENDATION_ALLOWED_VALUES["agentStyle"]
    assert p["interfaceStyle"]["font"] in RECOMMENDATION_ALLOWED_VALUES["font"]
    assert p["interfaceStyle"]["alignment"] in RECOMMENDATION_ALLOWED_VALUES["alignment"]
    assert p["interfaceStyle"]["contrast"] in RECOMMENDATION_ALLOWED_VALUES["contrast"]
    assert p["voiceStyle"]["gender"] in RECOMMENDATION_ALLOWED_VALUES["voiceGender"]
    assert out["system"]["agentPlatform"] in RECOMMENDATION_ALLOWED_VALUES["agentPlatform"]


def test_normalize_fills_missing_keys_from_defaults():
    out = normalize_recommended_agent_config({}, None)
    defaults = load_default_agent_recommendation_config()
    assert out["presentation"]["agentLanguage"] == defaults["presentation"]["agentLanguage"]
    assert out["presentation"]["interfaceStyle"]["size"] == defaults["presentation"]["interfaceStyle"]["size"]
    assert out["modality"]["inputModalities"] == defaults["modality"]["inputModalities"]
    assert out["behavior"]["responseTiming"] == defaults["behavior"]["responseTiming"]


def test_normalize_modality_collapses_to_text_unless_speech_present():
    raw = {"modality": {"inputModalities": ["video", "image"]}}
    out = normalize_recommended_agent_config(raw, None)
    # Implementation maps anything-not-speech to ["text"].
    assert out["modality"]["inputModalities"] == ["text"]

    raw2 = {"modality": {"inputModalities": ["text", "speech"]}}
    out2 = normalize_recommended_agent_config(raw2, None)
    assert out2["modality"]["inputModalities"] == ["text", "speech"]


def test_normalize_uses_selected_profile_name_when_content_missing():
    raw = {"content": {"adaptContentToUserProfile": True}}
    out = normalize_recommended_agent_config(raw, "alice_profile")
    assert out["content"]["userProfileName"] == "alice_profile"


def test_normalize_llm_block_drops_unknown_provider():
    raw = {"system": {"llm": {"provider": "atlantis-ai", "model": "x"}}}
    out = normalize_recommended_agent_config(raw, None)
    # Unknown provider — block should be empty.
    assert out["system"]["llm"] == {}


def test_normalize_llm_block_keeps_known_provider():
    raw = {"system": {"llm": {"provider": "openai", "model": "gpt-5-mini"}}}
    out = normalize_recommended_agent_config(raw, None)
    assert out["system"]["llm"] == {"provider": "openai", "model": "gpt-5-mini"}
