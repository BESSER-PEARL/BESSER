"""Processing a UserDiagram must not mutate the shared user reference domain model.

The user reference domain model is a module-level global used by every
UserDiagram request. Comments linked to objects used to be written into the
metadata of that model's classes, so one request's comments leaked into all
later (and concurrent) requests.
"""

from besser.BUML.metamodel.structural import Class, DomainModel, Metadata
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (
    process_object_diagram,
)


def _box():
    return {"x": 0, "y": 0, "width": 200, "height": 100}


def _user_diagram(title, comment_text):
    return {
        "title": title,
        "model": {
            "type": "UserDiagram",
            "elements": {
                "u-user": {
                    "id": "u-user", "name": "user_1", "type": "UserModelName", "owner": None,
                    "bounds": _box(), "attributes": [], "methods": [], "className": "User",
                },
                "c-1": {
                    "id": "c-1", "name": comment_text, "type": "Comments", "owner": None,
                    "bounds": _box(),
                },
            },
            "relationships": {
                "l-1": {
                    "id": "l-1", "name": "", "type": "Link", "owner": None, "bounds": _box(),
                    "source": {"element": "c-1", "direction": "Right"},
                    "target": {"element": "u-user", "direction": "Left"},
                },
            },
        },
    }


def _metadata_snapshot(model):
    snapshot = {}
    for cls in model.get_classes():
        meta = cls.metadata
        snapshot[cls.name] = None if meta is None else (meta.description, meta.uri, meta.icon)
    return snapshot


def _descriptions(object_model):
    texts = []
    if object_model.metadata and object_model.metadata.description:
        texts.append(object_model.metadata.description)
    for obj in object_model.objects:
        meta = obj.classifier.metadata
        if meta and meta.description:
            texts.append(meta.description)
    return "\n".join(texts)


def test_user_diagram_comments_do_not_leak_between_requests():
    before = _metadata_snapshot(user_reference_domain_model)

    first = process_object_diagram(
        _user_diagram("First", "first request comment"), user_reference_domain_model
    )
    second = process_object_diagram(
        _user_diagram("Second", "second request comment"), user_reference_domain_model
    )

    assert len(first.objects) == 1
    assert len(second.objects) == 1

    # The second result carries nothing from the first request.
    assert "first request comment" not in _descriptions(second)
    # Each comment is kept on its own request's object model, not dropped.
    assert first.metadata.description == "first request comment"
    assert second.metadata.description == "second request comment"

    # The shared reference model is untouched.
    assert _metadata_snapshot(user_reference_domain_model) == before


def test_object_diagram_comments_still_annotate_per_request_classes():
    """An ordinary ObjectDiagram keeps annotating its own (per-request) classes."""
    book = Class(name="Book", metadata=Metadata(description="A book"))
    model = DomainModel(name="Library", types={book})
    payload = _user_diagram("Objects", "linked comment")
    element = payload["model"]["elements"]["u-user"]
    element.update({"type": "ObjectName", "name": "book_1", "className": "Book"})

    object_model = process_object_diagram(payload, model)

    assert len(object_model.objects) == 1
    assert book.metadata.description == "A book\nlinked comment"
