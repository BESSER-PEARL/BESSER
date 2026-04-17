"""Rule-based manual mapping for agent configuration recommendations.

The rule source is the literature synthesis in:
"Papers_where_they_say_changing_X_in_chatbot_is_good_for_user_with_characteristc_Y.pdf".
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from .agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    load_default_agent_recommendation_config,
    merge_dicts,
    normalize_recommended_agent_config,
)

PRIMARY_LANGUAGE_TOKEN = "__PRIMARY_LANGUAGE__"

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

MANUAL_AGENT_CONFIG_RULES: List[Dict[str, Any]] = [
    {
        "id": "older_adults_readability",
        "label": "Older adults readability support",
        "summary": "Increase readability and reduce linguistic complexity for older adults.",
        "priority": 100,
        "evidence": [
            "Design Guidelines of Mobile Apps for Older Adults: Systematic Review and Thematic Analysis (2023)",
            "A study to guide developers in building accessible conversational systems to the older people",
            "Designing Conversational AI for Aging: A Systematic Review of Older Adults' Perceptions and Needs",
        ],
        "matchMode": "any",
        "conditions": [
            {"type": "age_gte", "value": 60},
            {
                "type": "profile_text_contains_any",
                "values": ["older adult", "older adults", "elderly", "senior", "aging"],
            },
        ],
        "recommendation": {
            "presentation": {
                "agentStyle": "formal",
                "languageComplexity": "simple",
                "sentenceLength": "concise",
                "interfaceStyle": {
                    "size": 20,
                    "font": "sans",
                    "lineSpacing": 1.8,
                    "alignment": "left",
                    "contrast": "high",
                },
                "useAbbreviations": False,
            },
            "behavior": {
                "responseTiming": "instant",
            },
        },
    },
    {
        "id": "adolescents_relatable_style",
        "label": "Adolescents conversational style",
        "summary": "Use a relatable, concise style for teenagers and adolescents.",
        "priority": 200,
        "evidence": [
            "Developing A Conversational Interface for an ACT-based Online Program: Understanding Adolescents' Expectations of Conversational Style",
            "On Being Cool - Exploring Interaction Design for Teenagers",
            "UX Design for Teenagers (Ages 13-17)",
        ],
        "matchMode": "any",
        "conditions": [
            {"type": "age_between", "min": 13, "max": 17},
            {
                "type": "profile_text_contains_any",
                "values": ["adolescent", "adolescents", "teen", "teenager", "teenagers", "youth"],
            },
        ],
        "recommendation": {
            "presentation": {
                "agentStyle": "informal",
                "languageComplexity": "medium",
                "sentenceLength": "concise",
            },
            "behavior": {
                "responseTiming": "instant",
            },
        },
    },
    {
        "id": "children_short_attention",
        "label": "Children language simplification",
        "summary": "Prefer short and simple language for children.",
        "priority": 210,
        "evidence": [
            "The Last Decade of HCI Research on Children and Voice-based Conversational Agents",
            "Younger children tend to have shorter attention spans and more limited language skills.",
        ],
        "matchMode": "any",
        "conditions": [
            {"type": "age_lte", "value": 12},
            {
                "type": "profile_text_contains_any",
                "values": ["child", "children", "kid", "kids", "toddler"],
            },
        ],
        "recommendation": {
            "presentation": {
                "languageComplexity": "simple",
                "sentenceLength": "concise",
                "useAbbreviations": False,
                "interfaceStyle": {
                    "size": 18,
                    "font": "sans",
                    "lineSpacing": 1.7,
                },
            },
            "behavior": {
                "responseTiming": "instant",
            },
        },
    },
    {
        "id": "low_literacy_accessibility",
        "label": "Low literacy accessibility",
        "summary": "Increase readability and multimodal support for low-literate users.",
        "priority": 220,
        "evidence": [
            "Designing User Interfaces for Illiterate and Semi-Literate Users: A Systematic Review and Future Research Agenda",
            "Actionable UI Design Guidelines for Smartphone Applications Inclusive of Low-Literate Users",
        ],
        "matchMode": "any",
        "conditions": [
            {
                "type": "profile_text_contains_any",
                "values": [
                    "illiterate",
                    "semi-literate",
                    "semiliterate",
                    "low-literate",
                    "low literate",
                    "low literacy",
                    "basic literacy",
                ],
            },
        ],
        "recommendation": {
            "presentation": {
                "languageComplexity": "simple",
                "sentenceLength": "concise",
                "useAbbreviations": False,
                "interfaceStyle": {
                    "size": 22,
                    "font": "sans",
                    "lineSpacing": 2.0,
                    "alignment": "left",
                    "contrast": "high",
                },
            },
            "modality": {
                "inputModalities": ["text", "speech"],
                "outputModalities": ["text", "speech"],
            },
        },
    },
    {
        "id": "neurodivergent_support",
        "label": "Neurodivergent user support",
        "summary": "Favor clear and less overwhelming communication patterns.",
        "priority": 230,
        "evidence": [
            "Designing Emerging Technologies for and with Neurodiverse Users",
            "A scoping review of inclusive and adaptive human-AI interaction design for neurodivergent users",
            "Chatbot Accessibility Guidance: A review and way forward",
        ],
        "matchMode": "any",
        "conditions": [
            {
                "type": "profile_text_contains_any",
                "values": ["neurodivergent", "neurodiverse", "autism", "adhd", "dyslexia", "cognitive disability"],
            },
        ],
        "recommendation": {
            "presentation": {
                "languageComplexity": "simple",
                "sentenceLength": "concise",
                "interfaceStyle": {
                    "font": "sans",
                    "lineSpacing": 1.8,
                    "alignment": "left",
                    "contrast": "high",
                },
            },
            "behavior": {
                "responseTiming": "delayed",
            },
        },
    },
    {
        "id": "multilingual_code_switching",
        "label": "Multilingual code-switching support",
        "summary": "For multilingual users, preserve language flexibility and strengthen language understanding.",
        "priority": 240,
        "evidence": [
            "Multilingual adult users tend to strongly prefer chatbots that can code-mix or code-switch.",
            "User Interface (UI) Design Issues for Multilingual Users: A Case Study",
        ],
        "matchMode": "any",
        "conditions": [
            {"type": "language_count_at_least", "value": 2},
            {
                "type": "profile_text_contains_any",
                "values": ["multilingual", "bilingual", "code-switch", "code switch", "code-mix", "code mix"],
            },
        ],
        "recommendation": {
            "presentation": {
                "agentLanguage": "original",
                "languageComplexity": "medium",
            },
        },
    },
    {
        "id": "single_language_localization",
        "label": "Single language localization",
        "summary": "When one supported language is clearly detected, localize the agent language.",
        "priority": 250,
        "evidence": [
            "Cross-Cultural Web Design Guidelines",
            "Culturally responsive AI chatbots: From framework to field evidence",
        ],
        "matchMode": "all",
        "conditions": [
            {"type": "language_count_equals", "value": 1},
            {
                "type": "detected_language_in",
                "values": ["english", "french", "german", "spanish", "luxembourgish", "portuguese"],
            },
        ],
        "recommendation": {
            "presentation": {
                "agentLanguage": PRIMARY_LANGUAGE_TOKEN,
            },
        },
    },
    {
        "id": "rtl_layout_fallback",
        "label": "Right-to-left language layout fallback",
        "summary": "Bias alignment for right-to-left language contexts using available alignment options.",
        "priority": 260,
        "evidence": [
            "Towards the Right Direction in BiDirectional User Interfaces",
            "Left-right vs right-left reading in language contexts.",
        ],
        "matchMode": "any",
        "conditions": [
            {"type": "detected_language_in", "values": ["arabic", "persian", "hebrew", "urdu"]},
            {"type": "profile_text_contains_any", "values": ["right-to-left", "rtl"]},
        ],
        "recommendation": {
            "presentation": {
                "sentenceLength": "concise",
                "interfaceStyle": {
                    "alignment": "justify",
                },
            },
        },
    },
]


MANUAL_AGENT_CONFIG_MAPPING: Dict[str, Any] = {
    "version": MANUAL_MAPPING_VERSION,
    "sourceDocument": MANUAL_MAPPING_SOURCE_DOCUMENT,
    "notes": [
        "Rules are derived from the paper list and notes in the source PDF.",
        "Rules are deterministic and only output currently supported agent configuration values.",
        "These rules complement, but do not replace, explicit user choices.",
    ],
    "rules": MANUAL_AGENT_CONFIG_RULES,
}


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
        "notes": list(MANUAL_AGENT_CONFIG_MAPPING.get("notes", [])),
        "allowedValues": deepcopy(RECOMMENDATION_ALLOWED_VALUES),
        "rules": deepcopy(MANUAL_AGENT_CONFIG_RULES),
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

    for rule in sorted(MANUAL_AGENT_CONFIG_RULES, key=lambda entry: entry.get("priority", 1000)):
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
