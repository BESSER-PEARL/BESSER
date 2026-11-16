"""JSONSchemaGenerator records which lines of its files come from which model element."""

import json
import os

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, Generalization, IntegerType,
    Multiplicity, Property, StringType,
)
from besser.generators.json import JSONSchemaGenerator
from besser.generators.json.json_trace import json_line_ranges


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN", id=ident("lit-open")),
                                    EnumerationLiteral("CLOSED", id=ident("lit-closed"))}, id=ident("enum-1"))
    person = Class("Person", {Property("fullname", StringType, id=ident("attr-name"))}, id=ident("class-person"))
    employee = Class("Employee", {Property("salary", IntegerType, id=ident("attr-salary"))}, id=ident("class-emp"))
    library = Class("Library", {Property("title", StringType, id=ident("attr-title"))}, id=ident("class-lib"))
    book = Class("Book", {Property("state", status, id=ident("attr-state"))}, id=ident("class-book"))
    has = BinaryAssociation("has", {
        Property("library", library, multiplicity=Multiplicity(1, 1), id=ident("end-lib")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-has"))
    return DomainModel("Lib", types={status, person, employee, library, book}, associations={has},
                       generalizations={Generalization(general=person, specific=employee, id=ident("gen-1"))})


def _generate(model, tmp_path, name, mode="regular"):
    out = tmp_path / name
    out.mkdir()
    generator = JSONSchemaGenerator(model, output_dir=str(out), mode=mode)
    generator.generate()
    files = {}
    for root, _, names in os.walk(out):
        for file_name in names:
            full = os.path.join(root, file_name)
            files[os.path.relpath(full, out).replace(os.sep, "/")] = open(full, "rb").read()
    return generator, files


def _lines(files, path):
    return files[path].decode("utf-8").split("\n")


def test_line_ranges_follow_json_dumps():
    document = {"a": {"b": [1, {"c": 2}, []], "d": {}}, "e": "x"}
    ranges = json_line_ranges(document)
    lines = json.dumps(document, indent=4).split("\n")
    assert ranges[()] == (1, len(lines))
    assert lines[ranges[("a", "b", 1, "c")][0] - 1].strip() == '"c": 2'
    start, end = ranges[("a", "b", 1)]
    assert (lines[start - 1].strip(), lines[end - 1].strip().rstrip(",")) == ("{", "}")
    assert lines[ranges[("a", "d")][0] - 1].strip() == '"d": {}'
    assert lines[ranges[("e",)][0] - 1].strip() == '"e": "x"'


def test_regular_schema_regions_point_at_each_element(tmp_path):
    generator, files = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = _lines(files, "json_schema.json")
    regions = generator.traced_files["json_schema.json"].regions

    kinds = {(r.element_id, r.rule.split("#")[1]) for r in regions}
    assert kinds == {
        ("class-person", "class"), ("class-emp", "class"), ("class-lib", "class"), ("class-book", "class"),
        ("attr-name", "attribute"), ("attr-salary", "attribute"), ("attr-title", "attribute"),
        ("attr-state", "attribute"), ("gen-1", "generalization"), ("assoc-has", "association"),
        ("end-lib", "association_end"), ("end-books", "association_end"),
        ("enum-1", "enumeration"), ("lit-open", "literal"), ("lit-closed", "literal"),
    }
    for region in regions:
        text = "\n".join(lines[region.start_line - 1:region.end_line])
        if region.element_type in ("Generalization", "BinaryAssociation"):
            continue  # unnamed, or its lines hold its ends' names
        assert f'"{region.element_name}"' in text, (region, text)
    # Classes are dumped under both "properties" and "definitions": two regions each.
    book = [r for r in regions if r.element_id == "class-book"]
    assert len(book) == 2 and all(lines[r.start_line - 1].strip() == '"Book": {' for r in book)
    state = [r for r in regions if r.element_id == "attr-state"]
    assert len(state) == 2
    for cls, attr in zip(book, state):
        assert cls.start_line < attr.start_line <= attr.end_line <= cls.end_line
    gen = next(r for r in regions if r.element_id == "gen-1")
    assert '"$ref": "#/definitions/Person"' in lines[gen.start_line]
    literal = next(r for r in regions if r.element_id == "lit-open")
    assert lines[literal.start_line - 1].strip() == '"OPEN",'


def test_smart_data_files_are_traced_per_class(tmp_path):
    generator, files = _generate(_model(with_ids=True), tmp_path, "traced", mode="smart_data")
    traces = generator.traced_files
    book_files = {p for p in traces if p.startswith("Book/")}
    assert book_files == {"Book/schema.json"} | {f"Book/examples/{n}" for n in (
        "example.json", "example-normalized.json", "example.jsonld", "example-normalized.jsonld")}
    assert "Book/ADOPTERS.yaml" in files and "Book/ADOPTERS.yaml" not in traces
    for path in book_files:
        assert [s.element.element_id for s in traces[path].sources] == ["class-book"]
        lines = _lines(files, path)
        by_key = {(r.element_id, r.rule.split("#")[1]): r for r in traces[path].regions}
        assert set(by_key) == {("attr-state", "attribute"), ("assoc-has", "association"),
                               ("end-lib", "association_end")}, path
        assert lines[by_key[("attr-state", "attribute")].start_line - 1].strip().startswith('"state":')
        assert lines[by_key[("end-lib", "association_end")].start_line - 1].strip().startswith('"library":')


def test_tracing_does_not_change_the_generated_files(tmp_path):
    for mode in ("regular", "smart_data"):
        _, traced = _generate(_model(with_ids=True), tmp_path, f"traced-{mode}", mode)
        untraced_generator, untraced = _generate(_model(with_ids=False), tmp_path, f"untraced-{mode}", mode)
        assert traced == untraced
        assert all(b"\x02" not in content for content in traced.values())
        assert all(t.regions == () for t in untraced_generator.traced_files.values())
