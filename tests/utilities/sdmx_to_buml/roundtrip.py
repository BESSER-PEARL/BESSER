"""Compare a domain model with its web-editor round trip (B-UML -> editor JSON -> B-UML)."""

from besser.BUML.metamodel.structural import Class, DomainModel, Enumeration
from besser.utilities.web_modeling_editor.backend.services.converters import (
    class_buml_to_json,
    process_class_diagram,
)


def signature(model: DomainModel) -> dict:
    """What the round trip must keep: classes (with their metadata URI), attributes, enumerations, associations.

    The class descriptions are compared by ``test_class_descriptions_survive_the_editor``: the editor
    converters currently append a copy of each description on every round trip.
    """
    classes = {
        t.name: (sorted((a.name, a.type.name, bool(a.is_id), bool(a.is_optional)) for a in t.attributes),
                 t.metadata.uri if t.metadata else None)
        for t in model.types if isinstance(t, Class)
    }
    enums = {t.name: sorted(lit.name for lit in t.literals) for t in model.types if isinstance(t, Enumeration)}
    associations = sorted(
        tuple(sorted((e.name, e.type.name, e.multiplicity.min, str(e.multiplicity.max)) for e in a.ends))
        for a in model.associations
    )
    return {"classes": classes, "enums": enums, "associations": associations}


def round_trip(model: DomainModel) -> DomainModel:
    """The model as the editor would read it back after showing it."""
    return process_class_diagram({"title": model.name, "model": class_buml_to_json(model)})


def round_trip_differences(model: DomainModel) -> list[str]:
    """Human-readable differences between a model and its round trip (empty: identical)."""
    before, after = signature(model), signature(round_trip(model))
    diffs = []
    for key in before:
        if before[key] != after[key]:
            if isinstance(before[key], dict):
                for name in sorted(set(before[key]) | set(after[key])):
                    if before[key].get(name) != after[key].get(name):
                        diffs.append(f"{key}.{name}: {before[key].get(name)} != {after[key].get(name)}")
            else:
                lost = [x for x in before[key] if x not in after[key]]
                gained = [x for x in after[key] if x not in before[key]]
                diffs.append(f"{key}: lost {lost[:3]}, gained {gained[:3]}")
    return diffs
