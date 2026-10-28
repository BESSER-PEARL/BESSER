import os

import pytest

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy.alloy_specification import AlloyBounds
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import AlloyResult, AlloySolver

# Marks the tests that run the Alloy Analyzer, located through these environment variables
needs_alloy = pytest.mark.skipif(
    not (os.getenv("JAVA_HOME") and os.getenv("BESSER_ALLOY_JAR")),
    reason="JAVA_HOME and BESSER_ALLOY_JAR must be set to run the Alloy Analyzer",
)


def check_consistency(model: DomainModel, scope: int) -> tuple[AlloyResult, AlloyBounds | None]:
    """Checks the consistency of *model* with AlloySolver; returns the result and the bounds it searched."""
    solver = AlloySolver(model)
    result = solver.check_consistency(scope=scope)
    return result, solver.get_scopes()


def generate_object_models(
    model: DomainModel, scope: int, num_models: int = 1
) -> tuple[AlloyResult, list[ObjectModel]]:
    """Generates up to *num_models* object models of *model* with AlloySolver; returns the result and the models."""
    solver = AlloySolver(model)
    return solver.generate_object_models(scope=scope, num_models=num_models)


def describe(object_model: ObjectModel) -> set[str]:
    """The objects, attribute values and links of *object_model*, e.g., as ``Team_0: Team``,
    ``Team_0.name = 'Barcelona'`` and ``Team_0.player = Player_0`` (a link, from each end)."""
    description = set()
    for obj in object_model.objects:
        description.add(f"{obj.name_}: {obj.classifier.name}")
        for slot in obj.slots:
            description.add(f"{obj.name_}.{slot.attribute.name} = {slot.value.value!r}")
    for link in object_model.links:
        a, b = link.connections
        description.add(f"{a.object.name_}.{b.name} = {b.object.name_}")
        description.add(f"{b.object.name_}.{a.name} = {a.object.name_}")
    return description
