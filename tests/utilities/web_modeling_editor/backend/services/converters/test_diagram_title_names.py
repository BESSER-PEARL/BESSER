"""A diagram title becomes the model's name, and titles are free text.

A repo opened from GitHub gets the class diagram "<repo> model", and most repo
names have hyphens ("bookshop-api"), which a B-UML name rejects.
"""
import pytest

from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml import (
    process_agent_diagram,
    process_class_diagram,
    process_object_diagram,
)

EMPTY = {"version": "4.0.0", "nodes": [], "edges": []}


@pytest.mark.parametrize("process", [
    process_class_diagram,
    lambda data: process_object_diagram(data, DomainModel("Shop")),
    process_agent_diagram,
], ids=["class", "object", "agent"])
def test_a_title_with_hyphens_and_spaces_becomes_a_valid_model_name(process):
    model = process({"title": "bookshop-api model", "model": EMPTY})
    assert model.name == "bookshop_api_model"
