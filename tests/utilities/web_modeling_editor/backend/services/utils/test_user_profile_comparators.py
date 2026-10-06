"""A UserDiagram criterion keeps its comparator on the way into the generated agent.

``age > 65`` reached the profile document (and every agent built from it) as
``age: 65``: the object model stores only the value. Jinja's ``tojson`` then
rendered any ``>`` that did survive as ``\u003e`` in the agent source.
"""

import json
import re

import pytest

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document,
)


def _senior_diagram(criterion: str, operator: str) -> dict:
    """A User linked to a Personal_Information box carrying one criterion, as the editor saves it."""
    return {"id": "d", "title": "Senior", "type": "UserDiagram", "model": {"type": "UserDiagram", "elements": {
        "u": {"id": "u", "name": "user", "type": "UserModelName", "className": "User", "attributes": [],
              "bounds": {"x": 0, "y": 0, "width": 200, "height": 60}},
        "p": {"id": "p", "name": "info", "type": "UserModelName", "className": "Personal_Information",
              "attributes": ["a"], "bounds": {"x": 300, "y": 0, "width": 200, "height": 60}},
        "a": {"id": "a", "name": criterion, "type": "UserModelAttribute", "owner": "p",
              "attributeOperator": operator, "bounds": {"x": 300, "y": 30, "width": 200, "height": 30}},
    }, "relationships": {"l": {"id": "l", "type": "ObjectLink", "name": "",
                               "source": {"element": "u"}, "target": {"element": "p"}}}}}


@pytest.mark.parametrize("criterion, operator, expected", [
    ("age > 65", ">", "> 65"),
    ("age >= 65", ">=", ">= 65"),
    ("age < 18", "<", "< 18"),
    ("age <= 18", "<=", "<= 18"),
    ("age = 30", "==", 30),
    ("age == 30", "==", 30),
])
def test_profile_document_keeps_the_comparator(criterion, operator, expected):
    document = generate_user_profile_document(_senior_diagram(criterion, operator))
    assert document["model"]["Personal_Information"]["age"] == expected


def test_generated_agent_shows_the_comparator(tmp_path):
    profile = generate_user_profile_document(_senior_diagram("age > 65", ">"))
    config = {"personalizationMapping": [{"name": "Senior", "configuration": {}, "user_profile": profile}]}
    agent = Agent("Helper")
    agent.new_state("initial", initial=True)
    BAFGenerator(agent, output_dir=str(tmp_path), config=config, generation_mode=GenerationMode.CODE_ONLY).generate()

    code = (tmp_path / "Helper.py").read_text(encoding="utf-8")
    assert '"age": "> 65"' in code
    literal = re.search(r"user_profiles\['Senior'\] = json\.loads\(r'''(.*?)'''\)", code, re.S).group(1)
    assert json.loads(literal)["model"]["Personal_Information"]["age"] == "> 65"
