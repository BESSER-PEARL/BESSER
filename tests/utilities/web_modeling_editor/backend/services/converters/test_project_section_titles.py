"""Section headers ``project_to_code`` writes above a builder's own banner.

For several models of one kind, ``project_to_code`` writes a numbered, titled
header (``# STRUCTURAL MODEL 1: "Library" #``) right above the banner the
builder writes (``# STRUCTURAL MODEL #``). ``_extract_all_sections`` read the
border line between the two as the first section's code, so each model came
back as an empty, titled diagram plus an untitled one with the content.
"""
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural import Class, DomainModel, Metadata, Property, StringType
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    _extract_all_sections,
    project_to_json,
)


def _domain_model(name, class_name):
    cls = Class(name=class_name, attributes={Property(name="title", type=StringType)})
    return DomainModel(name=name, types={cls})


def test_each_structural_model_comes_back_once_with_its_title_and_classes(tmp_path):
    project = Project(
        name="p", models=[_domain_model("Library", "Book"), _domain_model("Shop", "Item")],
        owner="o", metadata=Metadata(description="d"),
    )
    path = tmp_path / "project.py"
    project_to_code(project, str(path))

    diagrams = project_to_json(path.read_text(encoding="utf-8"))["diagrams"]["ClassDiagram"]

    assert [
        (d["title"], sorted(e["name"] for e in d["model"]["elements"].values() if e["type"] == "Class"))
        for d in diagrams
    ] == [("Library", ["Book"]), ("Shop", ["Item"])]


def test_a_comment_only_section_hands_its_title_to_the_banner_right_below_it():
    content = (
        '# STRUCTURAL MODEL 1: "Library" #\n\n'
        "##########################\n# STRUCTURAL MODEL #\n##########################\n"
        "a = 1\n"
    )

    assert _extract_all_sections(content, "STRUCTURAL") == [("Library", "##########################\na = 1")]


def test_a_title_is_not_carried_past_a_section_of_another_kind():
    content = (
        '# STRUCTURAL MODEL 1: "Library" #\n'
        "# OBJECT MODEL #\nobj = 1\n"
        "# STRUCTURAL MODEL #\na = 1\n"
    )

    assert _extract_all_sections(content, "STRUCTURAL") == [(None, "a = 1")]
