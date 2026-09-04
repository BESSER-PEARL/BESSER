"""Shared helpers for LLM integrations across BESSER utilities.

Centralised here so both the deterministic ``kg_to_buml`` LLM call and
the new KG-cleanup LLM flow can reuse the same response-cleaning logic.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def clean_json_response(response: str) -> str:
    """Strip markdown fences from a JSON response produced by an LLM."""
    json_text = response.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    elif json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    return json_text.strip()


def parse_json_safely(json_text: str) -> Optional[Any]:
    """Parse JSON, returning ``None`` on decode errors instead of raising."""
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None
