"""Integration coverage for the agent personalization mapping with v4 payloads.

The migration frontend bundles personalization state (saved configurations,
user profiles, per-diagram base models) and ships **v4** ``agent_model`` /
``user_profile`` payloads inside ``config.personalizationMapping`` when
generating a personalized agent. These tests pin the backend side of that
contract:

- ``normalize_personalization_mapping`` converts a v4 AgentDiagram payload
  into a BUML agent model and routes the raw user-profile payload through the
  profile generator callback;
- ``generate_user_profile_document`` accepts a v4 UserDiagram model (the
  exact shape the frontend stores in ``besser_userProfiles``) end-to-end
  through the real object processor + JSONObject generator;
- malformed mapping entries surface as HTTP 400s instead of being swallowed.

There was previously zero ``personalizationMapping`` coverage in ``tests/``.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

import pytest
from fastapi import HTTPException

from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    normalize_personalization_mapping,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "fixtures", "v4"
)


def _load(name: str) -> dict:
    path = os.path.normpath(os.path.join(FIXTURES_DIR, name))
    with open(path) as f:
        return json.load(f)


def _v4_user_profile_model() -> dict:
    """Minimal v4 UserDiagram model, as stored in ``besser_userProfiles``.

    ``className`` references the backend's user reference domain model
    (``constants/user_buml_model.py``) the same way the frontend inspector
    does.
    """
    return {
        "version": "4.0.0",
        "id": "user-profile-1",
        "title": "Teenager",
        "type": "UserDiagram",
        "nodes": [
            {
                "id": "user-node-1",
                "type": "UserModelName",
                "position": {"x": 0, "y": 0},
                "width": 160,
                "height": 70,
                "measured": {"width": 160, "height": 70},
                "data": {"name": "Teen", "className": "User", "attributes": []},
            },
            {
                "id": "user-node-2",
                "type": "UserModelName",
                "position": {"x": 0, "y": 160},
                "width": 160,
                "height": 70,
                "measured": {"width": 160, "height": 70},
                "data": {
                    "name": "TeenInfo",
                    "className": "Personal_Information",
                    "attributes": [
                        {"id": "attr-1", "name": "age", "value": "16", "attributeOperator": "=="},
                    ],
                },
            },
        ],
        "edges": [],
        "assessments": {},
    }


# ---------------------------------------------------------------------------
# normalize_personalization_mapping — v4 agent_model conversion
# ---------------------------------------------------------------------------


def test_normalize_personalization_mapping_converts_v4_agent_model():
    fixture = _load("agent_diagram_basic.json")

    forwarded_profiles = []

    def fake_profile_generator(payload):
        forwarded_profiles.append(payload)
        return {"model": {"name": "Teenager"}}

    config = {
        "personalizationMapping": [
            {
                "name": "Teenager",
                "configuration": {"agentStyle": "friendly"},
                "user_profile": _v4_user_profile_model(),
                "agent_model": deepcopy(fixture["model"]),
            }
        ]
    }
    json_data = {
        "title": fixture.get("title", "Agent"),
        "model": deepcopy(fixture["model"]),
        "config": config,
    }

    normalize_personalization_mapping(config, json_data, fake_profile_generator)

    entry = config["personalizationMapping"][0]

    # The v4 agent_model payload was converted to a BUML agent model in place.
    agent = entry["agent_model"]
    assert not isinstance(agent, dict)
    assert any(s.name == "Greeting" for s in agent.states)
    assert any(i.name == "greet" for i in agent.intents)

    # The user_profile was replaced by the simplified profile document and the
    # raw v4 payload was forwarded untouched to the generator callback.
    assert entry["user_profile"] == {"model": {"name": "Teenager"}}
    assert len(forwarded_profiles) == 1
    assert forwarded_profiles[0]["type"] == "UserDiagram"
    assert isinstance(forwarded_profiles[0]["nodes"], list)

    # The updated config is written back onto the payload.
    assert json_data["config"] is config


def test_normalize_personalization_mapping_missing_user_profile_is_400():
    config = {"personalizationMapping": [{"name": "x", "agent_model": {"nodes": [], "edges": []}}]}
    with pytest.raises(HTTPException) as exc_info:
        normalize_personalization_mapping(config, {"config": config}, lambda payload: payload)
    assert exc_info.value.status_code == 400
    assert "user_profile" in exc_info.value.detail


def test_normalize_personalization_mapping_non_dict_entry_is_400():
    config = {"personalizationMapping": ["not-an-object"]}
    with pytest.raises(HTTPException) as exc_info:
        normalize_personalization_mapping(config, {"config": config}, lambda payload: payload)
    assert exc_info.value.status_code == 400


def test_normalize_personalization_mapping_empty_mapping_is_noop():
    config = {"personalizationMapping": []}
    json_data = {"config": config}
    normalize_personalization_mapping(config, json_data, lambda payload: payload)
    assert config["personalizationMapping"] == []
    assert json_data["config"] is config


# ---------------------------------------------------------------------------
# generate_user_profile_document — real v4 UserDiagram end-to-end
# ---------------------------------------------------------------------------


def test_generate_user_profile_document_accepts_v4_user_diagram():
    """The exact v4 model shape the frontend stores in ``besser_userProfiles``
    (and ships in ``personalizationMapping[].user_profile``) must flow through
    the real object processor + JSONObject generator."""
    result = generate_user_profile_document(_v4_user_profile_model())

    assert isinstance(result, dict) and result, "expected a non-empty profile document"
    # Hierarchy normalization fired: "model" root, never raw "objects".
    assert "model" in result and "objects" not in result


def test_normalize_personalization_mapping_end_to_end_with_real_profile_generator():
    """Full pipeline: v4 agent_model -> BUML agent AND v4 user_profile ->
    simplified profile document, exactly as the generation router wires it."""
    fixture = _load("agent_diagram_basic.json")
    config = {
        "personalizationMapping": [
            {
                "name": "Teenager",
                "configuration": {"agentStyle": "friendly"},
                "user_profile": _v4_user_profile_model(),
                "agent_model": deepcopy(fixture["model"]),
            }
        ]
    }
    json_data = {
        "title": fixture.get("title", "Agent"),
        "model": deepcopy(fixture["model"]),
        "config": config,
    }

    normalize_personalization_mapping(config, json_data, generate_user_profile_document)

    entry = config["personalizationMapping"][0]
    assert not isinstance(entry["agent_model"], dict)
    assert isinstance(entry["user_profile"], dict) and entry["user_profile"]
    assert "model" in entry["user_profile"]
