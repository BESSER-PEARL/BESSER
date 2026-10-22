"""Editor element ids survive the class-diagram converters (issue #659).

``process_class_diagram`` stores each editor id on the B-UML element and
``class_buml_to_json`` writes it back instead of minting a fresh uuid, so a
renamed element keeps its id and every diagram that points at it (a GUI
form's ``data-source``, an object's ``classId``) still resolves. Elements
without an id still get a fresh one, and two elements never share one.
"""
import os
import re
import uuid

from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    class_buml_to_json,
    parse_buml_content,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from tests.utilities.web_modeling_editor.backend.services.converters.test_timestamp_roundtrip import _diagram

# Every class-diagram element of ``_diagram()`` that maps to a B-UML element.
MODEL_IDS = {
    "status": "Enumeration", "open": "ClassAttribute", "closed": "ClassAttribute",
    "library": "Class", "lib-name": "ClassAttribute", "renew": "ClassMethod",
    "book": "AbstractClass", "book-status": "ClassAttribute", "novel": "Class", "loan": "Class",
}
RELATIONSHIP_IDS = {"holds": "ClassBidirectional", "lends": "ClassBidirectional", "inherits": "ClassInheritance"}


def _ids(model_json):
    return ({key: e["type"] for key, e in model_json["elements"].items()},
            {key: r["type"] for key, r in model_json["relationships"].items()})


def test_json_to_buml_keeps_every_editor_id():
    model = process_class_diagram(_diagram())
    status = next(e for e in model.get_enumerations() if e.name == "Status")
    library = model.get_class_by_name("Library")
    novel = model.get_class_by_name("Novel")
    assert status.id == "status"
    assert {lit.name: lit.id for lit in status.literals} == {"OPEN": "open", "CLOSED": "closed"}
    assert library.id == "library"
    assert [a.id for a in library.attributes] == ["lib-name"]
    assert [m.id for m in library.methods] == ["renew"]
    assert {a.name: a.id for a in model.associations} == {"Library_Book": "holds", "Library_Novel": "lends"}
    assert [g.id for g in model.generalizations] == ["inherits"]
    assert next(iter(novel.generalizations)).id == "inherits"
    # The class promoted to an association class keeps its element id.
    assert model.get_class_by_name("Loan").id == "loan"


def test_json_buml_json_reuses_every_id():
    exported = class_buml_to_json(process_class_diagram(_diagram()))
    elements, relationships = _ids(exported)
    for key, element_type in MODEL_IDS.items():
        assert elements.get(key) == element_type, key
    for key, rel_type in RELATIONSHIP_IDS.items():
        assert relationships.get(key) == rel_type, key
    # References use the same ids.
    assert exported["elements"]["lib-name"]["owner"] == "library"
    assert exported["elements"]["library"]["attributes"] == ["lib-name"]
    assert exported["relationships"]["inherits"]["source"]["element"] == "novel"
    assert exported["relationships"]["inherits"]["target"]["element"] == "book"


def test_a_second_round_trip_gives_the_same_ids():
    once = class_buml_to_json(process_class_diagram(_diagram()))
    twice = class_buml_to_json(process_class_diagram({"title": "Lending", "model": once}))
    # OCL boxes and visual links are not model elements and are re-minted.
    visual = ("ClassOCLConstraint", "ClassOCLLink", "ClassLinkRel", "Comments", "Link")
    for side in (0, 1):
        assert ({k: t for k, t in _ids(once)[side].items() if t not in visual}
                == {k: t for k, t in _ids(twice)[side].items() if t not in visual})


def test_a_renamed_class_keeps_its_id():
    model = process_class_diagram(_diagram())
    model.get_class_by_name("Library").name = "Branch"
    exported = class_buml_to_json(model)
    assert exported["elements"]["library"]["name"] == "Branch"
    holds = exported["relationships"]["holds"]
    assert {holds["source"]["element"], holds["target"]["element"]} == {"library", "book"}


def test_buml_code_has_no_ids_unless_enabled(tmp_path):
    """Released BESSER versions reject ``id=``, so writing ids is opt-in."""
    model = process_class_diagram(_diagram())
    assert model.get_class_by_name("Library").id == "library"
    path = os.path.join(tmp_path, "model.py")
    domain_model_to_code(model, path)
    assert 'id="' not in open(path, encoding="utf-8").read()


def test_ids_survive_the_buml_python_round_trip(tmp_path):
    path = os.path.join(tmp_path, "model.py")
    domain_model_to_code(process_class_diagram(_diagram()), path, emit_ids=True)
    code = open(path, encoding="utf-8").read()
    assert re.search(r'^Library = Class\(name="Library".*, id="library"\)$', code, re.M)
    assert re.search(r'^Library_name: Property = Property\(name="name".*, id="lib-name"\)$', code, re.M)
    elements, relationships = _ids(class_buml_to_json(parse_buml_content(code)))
    for key, element_type in MODEL_IDS.items():
        assert elements.get(key) == element_type, key
    for key, rel_type in RELATIONSHIP_IDS.items():
        assert relationships.get(key) == rel_type, key


def test_a_model_without_ids_gets_fresh_ones_and_unchanged_code(tmp_path):
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    model = DomainModel(name="Lib", types={book})
    path = os.path.join(tmp_path, "model.py")
    domain_model_to_code(model, path)
    assert "id=" not in open(path, encoding="utf-8").read()
    first, second = class_buml_to_json(model), class_buml_to_json(model)
    for key in first["elements"]:
        uuid.UUID(key)
    assert set(first["elements"]).isdisjoint(second["elements"])
    assert book.id is None  # serialising does not write ids back


def test_elements_sharing_an_id_are_both_kept():
    a = Class(name="A", id="dup", attributes={Property(name="x", type=StringType, id="dup")})
    b = Class(name="B", id="dup")
    exported = class_buml_to_json(DomainModel(name="M", types={a, b}))
    names = sorted((e["type"], e["name"]) for e in exported["elements"].values())
    assert names == [("Class", "A"), ("Class", "B"), ("ClassAttribute", "x")]
    assert sum(1 for key in exported["elements"] if key == "dup") == 1


def test_unsafe_ids_are_replaced_with_a_warning(caplog):
    """The editor keys plain objects by id and the GUI parser splits field
    references at '.', so only a safe charset is kept."""
    bad = ["__proto__", "has space", 'quo"te', "attr.price", "x" * 129, "", "é"]
    good = ["class_p3gi94e5x_mudwdjvk_ufx", "0f3c-1:a_b", "x" * 128]
    classes = [Class(name=f"C{i}", id=value) for i, value in enumerate(bad + good)]
    with caplog.at_level("WARNING"):
        exported = class_buml_to_json(DomainModel(name="M", types=set(classes)))
    keys = set(exported["elements"])
    assert keys.isdisjoint(bad)
    assert set(good) <= keys
    for key in keys - set(good):
        uuid.UUID(key)
    assert sum("invalid element id" in r.getMessage() for r in caplog.records) == len(bad)


def test_a_duplicate_id_is_replaced_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        class_buml_to_json(DomainModel(name="M", types={Class(name="A", id="dup"), Class(name="B", id="dup")}))
    assert sum("duplicate element id" in r.getMessage() for r in caplog.records) == 1


def _export_buml(diagram):
    import asyncio

    import httpx

    from besser.utilities.web_modeling_editor.backend.backend import app

    async def post():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/besser_api/export-buml", json=diagram)

    response = asyncio.run(post())
    assert response.status_code == 200, response.text
    return response.text


def test_the_editor_export_writes_ids_only_when_the_deployment_turns_it_on(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.routers import conversion_router

    assert 'id="' not in _export_buml(_diagram())
    monkeypatch.setattr(conversion_router, "BUML_EMIT_IDS", True)
    assert 'id="library"' in _export_buml(_diagram())


def test_the_spec_driven_push_writes_ids_only_when_the_deployment_turns_it_on(tmp_path, monkeypatch):
    from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router

    export = {"id": "p", "type": "Project", "name": "Lending", "createdAt": "2026-01-01T00:00:00Z",
              "diagrams": {"ClassDiagram": [{"id": "d", "title": "Lending", "model": _diagram()["model"]}]}}

    def pushed(workdir):
        spec_driven_router._write_smart_model_to_buml(str(workdir), export)
        return open(os.path.join(workdir, "buml", "domain_model.py"), encoding="utf-8").read()

    assert 'id="' not in pushed(tmp_path / "off")
    monkeypatch.setattr(spec_driven_router, "BUML_EMIT_IDS", True)
    assert 'id="library"' in pushed(tmp_path / "on")
