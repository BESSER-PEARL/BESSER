"""Rule-based manual mapping for agent configuration recommendations.

The rule source is the literature synthesis in:
"Papers_where_they_say_changing_X_in_chatbot_is_good_for_user_with_characteristc_Y.pdf".
"""

import functools
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    load_default_agent_recommendation_config,
    merge_dicts,
    normalize_recommended_agent_config,
)

PRIMARY_LANGUAGE_TOKEN = "__PRIMARY_LANGUAGE__"

_MANUAL_MAPPING_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "constants" / "manual_mapping_rules.json"
)

_REQUIRED_RULE_KEYS = {"id", "label", "priority", "conditions", "recommendation"}
_REQUIRED_CONDITION_KEYS = {"type"}


def _validate_manual_mapping_rules(rules: Any) -> None:
    """Fail-fast schema check for the externalized rule data."""
    if not isinstance(rules, list):
        raise RuntimeError(
            f"manual_mapping_rules.json must contain a JSON array at the top level, "
            f"got {type(rules).__name__}."
        )
    seen_ids: Set[str] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise RuntimeError(
                f"manual_mapping_rules.json entry #{index} must be an object, "
                f"got {type(rule).__name__}."
            )
        missing = _REQUIRED_RULE_KEYS - rule.keys()
        if missing:
            raise RuntimeError(
                f"manual_mapping_rules.json entry #{index} (id={rule.get('id')!r}) "
                f"is missing required keys: {sorted(missing)}."
            )
        rule_id = rule["id"]
        if not isinstance(rule_id, str) or not rule_id:
            raise RuntimeError(
                f"manual_mapping_rules.json entry #{index} has an invalid 'id': {rule_id!r}."
            )
        if rule_id in seen_ids:
            raise RuntimeError(
                f"manual_mapping_rules.json contains duplicate rule id: {rule_id!r}."
            )
        seen_ids.add(rule_id)
        if not isinstance(rule["conditions"], list) or not rule["conditions"]:
            raise RuntimeError(
                f"manual_mapping_rules.json rule {rule_id!r} must have a non-empty "
                f"'conditions' list."
            )
        for cond_index, condition in enumerate(rule["conditions"]):
            if not isinstance(condition, dict):
                raise RuntimeError(
                    f"manual_mapping_rules.json rule {rule_id!r} condition #{cond_index} "
                    f"must be an object."
                )
            missing_cond = _REQUIRED_CONDITION_KEYS - condition.keys()
            if missing_cond:
                raise RuntimeError(
                    f"manual_mapping_rules.json rule {rule_id!r} condition #{cond_index} "
                    f"is missing keys: {sorted(missing_cond)}."
                )
        if not isinstance(rule["recommendation"], dict):
            raise RuntimeError(
                f"manual_mapping_rules.json rule {rule_id!r} 'recommendation' must be an object."
            )


@functools.lru_cache(maxsize=1)
def _load_manual_mapping_rules() -> List[Dict[str, Any]]:
    """Load and validate the externalized manual mapping rule table."""
    try:
        with _MANUAL_MAPPING_RULES_PATH.open(encoding="utf-8") as fh:
            rules = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"manual_mapping_rules.json not found at {_MANUAL_MAPPING_RULES_PATH}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"manual_mapping_rules.json is not valid JSON: {exc}."
        ) from exc
    _validate_manual_mapping_rules(rules)
    return rules

MANUAL_MAPPING_VERSION = "2026-04-16"
MANUAL_MAPPING_SOURCE_DOCUMENT = (
    "Papers_where_they_say_changing_X_in_chatbot_is_good_for_user_with_characteristc_Y.pdf"
)

SUPPORTED_AGENT_LANGUAGES = set(RECOMMENDATION_ALLOWED_VALUES["agentLanguage"]) - {"original"}

LANGUAGE_CODE_TO_NAME = {
    "en": "english",
    "eng": "english",
    "fr": "french",
    "fra": "french",
    "de": "german",
    "deu": "german",
    "es": "spanish",
    "spa": "spanish",
    "pt": "portuguese",
    "por": "portuguese",
    "lb": "luxembourgish",
    "ltz": "luxembourgish",
    "ar": "arabic",
    "ara": "arabic",
    "fa": "persian",
    "fas": "persian",
    "he": "hebrew",
    "heb": "hebrew",
    "ur": "urdu",
    "urd": "urdu",
}

_MANUAL_MAPPING_NOTES: List[str] = [
    "Rules are derived from the paper list and notes in the source PDF.",
    "Rules are deterministic and only output currently supported agent configuration values.",
    "These rules complement, but do not replace, explicit user choices.",
]


def __getattr__(name: str) -> Any:
    """Lazy module-level accessors for backward-compatible public names.

    The rule data was moved out of this module into
    ``backend/constants/manual_mapping_rules.json``. To keep the previous
    import surface working we expose ``MANUAL_AGENT_CONFIG_RULES`` and
    ``MANUAL_AGENT_CONFIG_MAPPING`` as lazily-loaded attributes.
    """
    if name == "MANUAL_AGENT_CONFIG_RULES":
        return deepcopy(_load_manual_mapping_rules())
    if name == "MANUAL_AGENT_CONFIG_MAPPING":
        return {
            "version": MANUAL_MAPPING_VERSION,
            "sourceDocument": MANUAL_MAPPING_SOURCE_DOCUMENT,
            "notes": list(_MANUAL_MAPPING_NOTES),
            "rules": deepcopy(_load_manual_mapping_rules()),
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _extract_age(profile_document: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_norm = _normalize_key(key)
                if key_norm.endswith("age") or key_norm == "age":
                    parsed = _as_float(nested)
                    if parsed is not None and 0 <= parsed <= 120:
                        candidates.append(parsed)
                _walk(nested)
        elif isinstance(value, list):
            for entry in value:
                _walk(entry)

    _walk(profile_document)
    if not candidates:
        return None
    return candidates[0]


def _collect_text_fragments(value: Any, sink: List[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            sink.append(str(key))
            _collect_text_fragments(nested, sink)
    elif isinstance(value, list):
        for entry in value:
            _collect_text_fragments(entry, sink)
    elif value is not None:
        sink.append(str(value))


def _extract_detected_languages(profile_document: Dict[str, Any], profile_text: str) -> Set[str]:
    detected: Set[str] = set()

    def _register_language_token(raw_token: str) -> None:
        token = raw_token.strip().lower()
        if not token:
            return
        if token in SUPPORTED_AGENT_LANGUAGES:
            detected.add(token)
            return
        mapped = LANGUAGE_CODE_TO_NAME.get(token)
        if mapped:
            detected.add(mapped)
            return

        # Handle comma-separated language lists in a single field.
        parts = re.split(r"[,;/|]", token)
        for part in parts:
            part_token = part.strip().lower()
            if part_token in SUPPORTED_AGENT_LANGUAGES:
                detected.add(part_token)
            elif part_token in LANGUAGE_CODE_TO_NAME:
                detected.add(LANGUAGE_CODE_TO_NAME[part_token])

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_norm = _normalize_key(key)
                if "language" in key_norm or key_norm in {"iso6391", "iso6933", "iso3166"}:
                    if isinstance(nested, str):
                        _register_language_token(nested)
                    elif isinstance(nested, list):
                        for entry in nested:
                            if isinstance(entry, str):
                                _register_language_token(entry)
                _walk(nested)
        elif isinstance(value, list):
            for entry in value:
                _walk(entry)

    _walk(profile_document)

    # Fallback only for full language names to avoid false positives from short codes.
    for language_name in SUPPORTED_AGENT_LANGUAGES.union({"arabic", "persian", "hebrew", "urdu"}):
        if re.search(rf"\b{re.escape(language_name)}\b", profile_text):
            detected.add(language_name)

    return detected


def _extract_profile_signals(profile_document: Dict[str, Any]) -> Dict[str, Any]:
    text_fragments: List[str] = []
    _collect_text_fragments(profile_document, text_fragments)
    profile_text = " ".join(text_fragments).lower()

    age = _extract_age(profile_document)
    detected_languages = _extract_detected_languages(profile_document, profile_text)

    return {
        "age": age,
        "profileText": profile_text,
        "detectedLanguages": sorted(detected_languages),
        "isMultilingual": len(detected_languages) >= 2,
    }


def _condition_matches(condition: Dict[str, Any], signals: Dict[str, Any]) -> bool:
    condition_type = condition.get("type")
    age = signals.get("age")
    profile_text = signals.get("profileText", "")
    detected_languages = set(signals.get("detectedLanguages", []))

    if condition_type == "age_gte":
        threshold = _as_float(condition.get("value"))
        return age is not None and threshold is not None and age >= threshold

    if condition_type == "age_lte":
        threshold = _as_float(condition.get("value"))
        return age is not None and threshold is not None and age <= threshold

    if condition_type == "age_between":
        minimum = _as_float(condition.get("min"))
        maximum = _as_float(condition.get("max"))
        return age is not None and minimum is not None and maximum is not None and minimum <= age <= maximum

    if condition_type == "profile_text_contains_any":
        values = [str(value).lower() for value in condition.get("values", [])]
        return any(value in profile_text for value in values)

    if condition_type == "language_count_at_least":
        threshold = int(condition.get("value", 0))
        return len(detected_languages) >= threshold

    if condition_type == "language_count_equals":
        expected = int(condition.get("value", -1))
        return len(detected_languages) == expected

    if condition_type == "detected_language_in":
        values = {str(value).lower() for value in condition.get("values", [])}
        return bool(detected_languages.intersection(values))

    return False


def _rule_matches(rule: Dict[str, Any], signals: Dict[str, Any]) -> bool:
    conditions = rule.get("conditions", [])
    if not conditions:
        return False

    mode = str(rule.get("matchMode", "all")).lower()
    results = [_condition_matches(condition, signals) for condition in conditions]

    if mode == "any":
        return any(results)
    return all(results)


def _primary_detected_language(signals: Dict[str, Any]) -> str:
    detected_languages = [
        language for language in signals.get("detectedLanguages", []) if language in SUPPORTED_AGENT_LANGUAGES
    ]
    if not detected_languages:
        return "original"
    return detected_languages[0]


def _resolve_dynamic_values(value: Any, signals: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_dynamic_values(nested, signals) for key, nested in value.items()}
    if isinstance(value, list):
        return [_resolve_dynamic_values(entry, signals) for entry in value]
    if value == PRIMARY_LANGUAGE_TOKEN:
        return _primary_detected_language(signals)
    return deepcopy(value)


def _to_structured_config(current_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(current_config, dict):
        return {}

    section_keys = ("presentation", "modality", "behavior", "content", "system")
    if any(key in current_config for key in section_keys):
        structured = {}
        for key in section_keys:
            section = current_config.get(key)
            if isinstance(section, dict):
                structured[key] = deepcopy(section)
        return structured

    # Flat to structured conversion for compatibility.
    return {
        "presentation": {
            "agentLanguage": current_config.get("agentLanguage"),
            "agentStyle": current_config.get("agentStyle"),
            "languageComplexity": current_config.get("languageComplexity"),
            "sentenceLength": current_config.get("sentenceLength"),
            "interfaceStyle": current_config.get("interfaceStyle"),
            "voiceStyle": current_config.get("voiceStyle"),
            "avatar": current_config.get("avatar"),
            "useAbbreviations": current_config.get("useAbbreviations"),
        },
        "modality": {
            "inputModalities": current_config.get("inputModalities"),
            "outputModalities": current_config.get("outputModalities"),
        },
        "behavior": {
            "responseTiming": current_config.get("responseTiming"),
        },
        "content": {
            "adaptContentToUserProfile": current_config.get("adaptContentToUserProfile"),
            "userProfileName": current_config.get("userProfileName"),
        },
    }


def get_manual_agent_config_mapping() -> Dict[str, Any]:
    """Return the complete manual mapping definition and metadata."""
    return {
        "version": MANUAL_MAPPING_VERSION,
        "sourceDocument": MANUAL_MAPPING_SOURCE_DOCUMENT,
        "notes": list(_MANUAL_MAPPING_NOTES),
        "allowedValues": deepcopy(RECOMMENDATION_ALLOWED_VALUES),
        "rules": deepcopy(_load_manual_mapping_rules()),
    }


def build_manual_mapping_recommendation(
    user_profile_document: Dict[str, Any],
    user_profile_name: Optional[str] = None,
    current_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a rule-based agent configuration recommendation from profile signals."""
    defaults = load_default_agent_recommendation_config()
    merged_config = merge_dicts(defaults, _to_structured_config(current_config))

    signals = _extract_profile_signals(user_profile_document if isinstance(user_profile_document, dict) else {})

    matched_rules: List[Dict[str, Any]] = []
    rule_overrides: Dict[str, Any] = {}

    for rule in sorted(_load_manual_mapping_rules(), key=lambda entry: entry.get("priority", 1000)):
        if not _rule_matches(rule, signals):
            continue

        matched_rules.append(
            {
                "id": rule.get("id"),
                "label": rule.get("label"),
                "summary": rule.get("summary"),
                "priority": rule.get("priority"),
                "evidence": list(rule.get("evidence", [])),
            }
        )

        resolved_recommendation = _resolve_dynamic_values(rule.get("recommendation", {}), signals)
        if isinstance(resolved_recommendation, dict):
            rule_overrides = merge_dicts(rule_overrides, resolved_recommendation)

    if rule_overrides:
        merged_config = merge_dicts(merged_config, rule_overrides)

    merged_config = merge_dicts(
        merged_config,
        {
            "content": {
                "adaptContentToUserProfile": True,
                "userProfileName": user_profile_name,
            }
        },
    )

    normalized_config = normalize_recommended_agent_config(merged_config, user_profile_name)

    return {
        "config": normalized_config,
        "matchedRules": matched_rules,
        "signals": {
            "age": signals.get("age"),
            "detectedLanguages": list(signals.get("detectedLanguages", [])),
            "isMultilingual": bool(signals.get("isMultilingual", False)),
        },
    }
