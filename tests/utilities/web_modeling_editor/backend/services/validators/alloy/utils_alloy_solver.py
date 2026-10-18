from besser.BUML.metamodel.object import ObjectModel


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
