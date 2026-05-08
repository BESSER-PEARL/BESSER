"""Utilities for LLM-based agent configuration recommendation."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RECOMMENDATION_ALLOWED_VALUES = {
    "agentLanguage": ["original", "english", "french", "german", "spanish", "luxembourgish", "portuguese"],
    "agentStyle": ["original", "formal", "informal"],
    "languageComplexity": ["original", "simple", "medium", "complex"],
    "sentenceLength": ["original", "concise", "verbose"],
    "font": ["sans", "serif", "monospace", "neutral", "grotesque", "condensed"],
    "alignment": ["left", "center", "justify"],
    "color": [
        "var(--besser-primary-contrast)",
        "#000000",
        "#ffffff",
        "#1a73e8",
        "#34a853",
        "#fbbc05",
        "#db4437",
        "#6a1b9a",
    ],
    "contrast": ["low", "medium", "high"],
    "voiceGender": ["male", "female", "ambiguous"],
    "responseTiming": ["instant", "delayed"],
    "agentPlatform": ["websocket", "streamlit", "telegram"],
    "intentRecognitionTechnology": ["classical", "llm-based"],
    "llmProvider": ["openai", "huggingface", "huggingfaceapi", "replicate"],
    "openaiModels": ["gpt-5.5", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts without mutating the inputs."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_default_agent_recommendation_config() -> Dict[str, Any]:
    """Load default structured agent configuration and complete missing sections.

    The file is re-read on every call so hot edits to ``default_config.json``
    are picked up without needing to restart the process or clear a cache.
    """
    fallback = {
        "presentation": {
            "agentLanguage": "original",
            "agentStyle": "original",
            "languageComplexity": "original",
            "sentenceLength": "concise",
            "interfaceStyle": {
                "size": 16,
                "font": "sans",
                "lineSpacing": 1.5,
                "alignment": "left",
                "color": "var(--besser-primary-contrast)",
                "contrast": "medium",
            },
            "voiceStyle": {
                "gender": "male",
                "speed": 1,
            },
            "avatar": None,
            "useAbbreviations": False,
        },
        "modality": {
            "inputModalities": ["text"],
            "outputModalities": ["text"],
        },
        "behavior": {
            "responseTiming": "instant",
        },
        "content": {
            "adaptContentToUserProfile": False,
            "userProfileName": None,
        },
        "system": {
            "agentPlatform": "streamlit",
            "intentRecognitionTechnology": "classical",
            "llm": {},
        },
    }

    config_path = Path(__file__).resolve().parents[2] / "constants" / "default_config.json"
    if not config_path.is_file():
        return fallback

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            return merge_dicts(fallback, loaded)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load default_config.json for recommendations: %s", exc)

    return fallback


# Backwards-compat shim: callers (and tests) used to clear an ``lru_cache``
# wrapping the loader. The cache is gone, but exposing a no-op ``cache_clear``
# keeps existing fixtures working without forcing a coordinated test rewrite.
load_default_agent_recommendation_config.cache_clear = lambda: None  # type: ignore[attr-defined]


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from plain or markdown-fenced model output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Empty LLM response")

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a valid JSON object")

    candidate = text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON root must be an object")
    return parsed


def _pick_allowed(value: Any, allowed: List[Any], default: Any) -> Any:
    return value if value in allowed else default


def _clamp_number(value: Any, minimum: float, maximum: float, default: float, digits: Optional[int] = None) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)

    num = min(max(num, minimum), maximum)
    if digits is not None:
        num = round(num, digits)
    return num


def _normalize_modality(value: Any, default_value: List[str]) -> List[str]:
    if not isinstance(value, list):
        return list(default_value)
    return ["text", "speech"] if "speech" in value else ["text"]


def normalize_recommended_agent_config(
    raw_config: Dict[str, Any],
    selected_profile_name: Optional[str],
) -> Dict[str, Any]:
    """Normalize recommendation to a safe structured config with allowed frontend values."""
    defaults = load_default_agent_recommendation_config()
    merged = merge_dicts(defaults, raw_config if isinstance(raw_config, dict) else {})

    presentation = merged.get("presentation") if isinstance(merged.get("presentation"), dict) else {}
    modality = merged.get("modality") if isinstance(merged.get("modality"), dict) else {}
    behavior = merged.get("behavior") if isinstance(merged.get("behavior"), dict) else {}
    content = merged.get("content") if isinstance(merged.get("content"), dict) else {}
    system = merged.get("system") if isinstance(merged.get("system"), dict) else {}

    default_presentation = defaults["presentation"]
    default_modality = defaults["modality"]
    default_behavior = defaults["behavior"]
    default_system = defaults["system"]

    interface_style = presentation.get("interfaceStyle") if isinstance(presentation.get("interfaceStyle"), dict) else {}
    voice_style = presentation.get("voiceStyle") if isinstance(presentation.get("voiceStyle"), dict) else {}

    normalized_llm = {}
    llm_obj = system.get("llm") if isinstance(system.get("llm"), dict) else {}
    provider = llm_obj.get("provider")
    if provider in RECOMMENDATION_ALLOWED_VALUES["llmProvider"]:
        model_value = llm_obj.get("model")
        model_str = model_value.strip() if isinstance(model_value, str) else ""
        normalized_llm = {"provider": provider, "model": model_str} if model_str else {"provider": provider}

    resolved_profile_name = selected_profile_name.strip() if isinstance(selected_profile_name, str) else None
    normalized_content_user_name = content.get("userProfileName")
    if not isinstance(normalized_content_user_name, str) or not normalized_content_user_name.strip():
        normalized_content_user_name = resolved_profile_name

    normalized = {
        "presentation": {
            "agentLanguage": _pick_allowed(
                presentation.get("agentLanguage"),
                RECOMMENDATION_ALLOWED_VALUES["agentLanguage"],
                default_presentation["agentLanguage"],
            ),
            "agentStyle": _pick_allowed(
                presentation.get("agentStyle"),
                RECOMMENDATION_ALLOWED_VALUES["agentStyle"],
                default_presentation["agentStyle"],
            ),
            "languageComplexity": _pick_allowed(
                presentation.get("languageComplexity"),
                RECOMMENDATION_ALLOWED_VALUES["languageComplexity"],
                default_presentation["languageComplexity"],
            ),
            "sentenceLength": _pick_allowed(
                presentation.get("sentenceLength"),
                RECOMMENDATION_ALLOWED_VALUES["sentenceLength"],
                default_presentation["sentenceLength"],
            ),
            "interfaceStyle": {
                "size": int(
                    _clamp_number(
                        interface_style.get("size"),
                        10,
                        32,
                        default_presentation["interfaceStyle"]["size"],
                    )
                ),
                "font": _pick_allowed(
                    interface_style.get("font"),
                    RECOMMENDATION_ALLOWED_VALUES["font"],
                    default_presentation["interfaceStyle"]["font"],
                ),
                "lineSpacing": _clamp_number(
                    interface_style.get("lineSpacing"),
                    1,
                    3,
                    default_presentation["interfaceStyle"]["lineSpacing"],
                    digits=1,
                ),
                "alignment": _pick_allowed(
                    interface_style.get("alignment"),
                    RECOMMENDATION_ALLOWED_VALUES["alignment"],
                    default_presentation["interfaceStyle"]["alignment"],
                ),
                "color": _pick_allowed(
                    interface_style.get("color"),
                    RECOMMENDATION_ALLOWED_VALUES["color"],
                    default_presentation["interfaceStyle"]["color"],
                ),
                "contrast": _pick_allowed(
                    interface_style.get("contrast"),
                    RECOMMENDATION_ALLOWED_VALUES["contrast"],
                    default_presentation["interfaceStyle"]["contrast"],
                ),
            },
            "voiceStyle": {
                "gender": _pick_allowed(
                    voice_style.get("gender"),
                    RECOMMENDATION_ALLOWED_VALUES["voiceGender"],
                    default_presentation["voiceStyle"]["gender"],
                ),
                "speed": _clamp_number(
                    voice_style.get("speed"),
                    0.5,
                    2.0,
                    default_presentation["voiceStyle"]["speed"],
                    digits=2,
                ),
            },
            "avatar": None,
            "useAbbreviations": bool(
                presentation.get(
                    "useAbbreviations",
                    default_presentation["useAbbreviations"],
                )
            ),
        },
        "modality": {
            "inputModalities": _normalize_modality(
                modality.get("inputModalities"),
                default_modality["inputModalities"],
            ),
            "outputModalities": _normalize_modality(
                modality.get("outputModalities"),
                default_modality["outputModalities"],
            ),
        },
        "behavior": {
            "responseTiming": _pick_allowed(
                behavior.get("responseTiming"),
                RECOMMENDATION_ALLOWED_VALUES["responseTiming"],
                default_behavior["responseTiming"],
            ),
        },
        "content": {
            "adaptContentToUserProfile": bool(content.get("adaptContentToUserProfile", True)),
            "userProfileName": normalized_content_user_name,
        },
        "system": {
            "agentPlatform": _pick_allowed(
                system.get("agentPlatform"),
                RECOMMENDATION_ALLOWED_VALUES["agentPlatform"],
                default_system["agentPlatform"],
            ),
            "intentRecognitionTechnology": _pick_allowed(
                system.get("intentRecognitionTechnology"),
                RECOMMENDATION_ALLOWED_VALUES["intentRecognitionTechnology"],
                default_system["intentRecognitionTechnology"],
            ),
            "llm": normalized_llm,
        },
    }

    return normalized
