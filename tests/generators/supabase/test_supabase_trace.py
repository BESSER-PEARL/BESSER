"""SupabaseGenerator records which lines of its migration file come from which model element.

The file name and its header carry a timestamp, so each test resolves the file by glob and compares
outputs with the timestamp masked.
"""

import glob
import os
import re

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral, IntegerType, Multiplicity, Property,
    StringType,
)
from besser.BUML.metamodel.traceability.traceability import model_elements
from besser.generators.supabase import SupabaseGenerator


def _model(with_ids: bool) -> DomainModel:
    def ident(value):
        return value if with_ids else None

    status = Enumeration("Status", {EnumerationLiteral("OPEN", id=ident("lit-open")),
                                    EnumerationLiteral("CLOSED", id=ident("lit-closed"))}, id=ident("enum-1"))
    user = Class("User", {Property("nickname", StringType, id=ident("attr-nick"))}, id=ident("class-user"))
    book = Class("Book", {Property("pages", IntegerType, id=ident("attr-pages")),
                          Property("state", status, id=ident("attr-state"))}, id=ident("class-book"))
    tag = Class("Tag", {Property("label", StringType, id=ident("attr-label"))}, id=ident("class-tag"))
    owns = BinaryAssociation("owns", {
        Property("owner", user, multiplicity=Multiplicity(1, 1), id=ident("end-owner")),
        Property("books", book, multiplicity=Multiplicity(0, "*"), id=ident("end-books")),
    }, id=ident("assoc-owns"))
    tagged = BinaryAssociation("tagged", {
        Property("taggedBooks", book, multiplicity=Multiplicity(0, "*"), id=ident("end-tbooks")),
        Property("tags", tag, multiplicity=Multiplicity(0, "*"), id=ident("end-tags")),
    }, id=ident("assoc-tagged"))
    category = BinaryAssociation("category", {
        Property("category", tag, multiplicity=Multiplicity(1, 1), id=ident("end-category")),
        Property("categorised", book, multiplicity=Multiplicity(0, "*"), id=ident("end-categorised")),
    }, id=ident("assoc-category"))
    return DomainModel("Shelf", types={status, user, book, tag}, associations={owns, tagged, category})


def _strip_ids(model: DomainModel) -> None:
    """Remove every id in place: the same objects keep the same set order, so the output stays comparable."""
    for element in model_elements(model):
        element.id = None


def _generate(model, tmp_path, name):
    out = tmp_path / name
    out.mkdir()
    generator = SupabaseGenerator(model, output_dir=str(out))
    generator.generate()
    (path,) = glob.glob(str(out / "*.sql"))
    return generator, os.path.basename(path), open(path, "rb").read()


def _masked(content: bytes, file_name: str) -> bytes:
    timestamp = file_name.split("_")[0]
    return re.sub(rb"\d{14}", b"TS", content.replace(timestamp.encode(), b"TS"))


def test_regions_cover_every_traced_kind_and_contain_the_element_code(tmp_path):
    generator, file_name, content = _generate(_model(with_ids=True), tmp_path, "traced")
    lines = content.decode("utf-8").split("\n")
    regions = generator.traced_files[file_name].regions

    by_key = {(r.element_id, r.rule.split("#")[1]): r for r in regions}
    assert set(by_key) >= {
        ("enum-1", "enumeration"), ("lit-open", "literal"), ("lit-closed", "literal"),
        ("class-user", "table"), ("class-book", "table"), ("class-tag", "table"),
        ("attr-nick", "attribute"), ("attr-pages", "attribute"), ("attr-state", "attribute"),
        ("assoc-tagged", "table"), ("end-tbooks", "association_end"), ("end-tags", "association_end"),
        ("class-user", "auth_trigger"), ("class-user", "row_level_security"), ("class-book", "grant"),
    }
    for region in regions:
        text = "\n".join(lines[region.start_line - 1:region.end_line]).lower()
        if region.rule.endswith(("#foreign_key", "#auth_trigger")) or region.element_type == "BinaryAssociation":
            continue  # lines hold the ends' or the table's names, not the association's own
        assert region.element_name.lower() in text, (region, text)
    assert lines[by_key[("class-book", "table")].start_line - 1] == 'CREATE TABLE IF NOT EXISTS public."book" ('
    assert lines[by_key[("attr-pages", "attribute")].start_line - 1].strip().startswith('"pages" INTEGER')
    assert lines[by_key[("assoc-tagged", "table")].start_line - 1] == 'CREATE TABLE IF NOT EXISTS public."tagged" ('
    assert "ENUM (" in "\n".join(lines[by_key[("enum-1", "enumeration")].start_line - 1:
                                       by_key[("enum-1", "enumeration")].end_line])
    table, column = by_key[("class-book", "table")], by_key[("attr-pages", "attribute")]
    assert table.start_line < column.start_line <= column.end_line <= table.end_line
    # The 1:N association becomes a foreign key added by ALTER TABLE.
    fk = next(r for r in regions if r.element_id == "assoc-category" and r.rule.endswith("#foreign_key"))
    assert lines[fk.start_line - 1] == 'ALTER TABLE public."book"'


def test_tracing_does_not_change_the_generated_sql(tmp_path):
    model = _model(with_ids=True)
    _, traced_name, traced = _generate(model, tmp_path, "traced")
    _strip_ids(model)
    untraced_generator, untraced_name, untraced = _generate(model, tmp_path, "untraced")
    assert _masked(traced, traced_name) == _masked(untraced, untraced_name)
    assert b"\x02" not in traced
    assert untraced_generator.traced_files[untraced_name].regions == ()
