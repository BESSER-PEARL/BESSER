"""
Banner-immunity tests for ``project_to_json``.

The importer must derive the number and identity of diagrams from the
authoritative ``Project(models=[...])`` list and the Python AST — NOT from
counting ``# ... MODEL #`` comment banners. This guards WME issue #161, where a
stray/duplicate object-model banner caused each object model to yield one real
diagram plus one spurious empty one (2 models -> 4 diagrams).
"""

import re

from besser.BUML.metamodel.structural.structural import (
    Class, Property, DomainModel, Metadata, StringType, IntegerType,
)
from besser.BUML.metamodel.object import (
    Object, AttributeLink, DataValue, ObjectModel,
)
from besser.BUML.metamodel.project import Project
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
    project_to_json,
)


def _multi_object_project_code(tmp_path) -> str:
    """Generate a real BUML file for a project with 1 domain + 2 object models."""
    title = Property(name="title", type=StringType)
    pages = Property(name="pages", type=IntegerType)
    book = Class(name="Book", attributes={title, pages})
    library = Class(name="Library", attributes={Property(name="name", type=StringType)})
    dm = DomainModel(name="LibraryModel", types={library, book}, associations=set())

    om1 = ObjectModel(name="ObjectModelOne", objects={Object(
        name="BookOne", classifier=book,
        slots=[AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Book One")),
               AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=100))])})
    om2 = ObjectModel(name="ObjectModelTwo", objects={Object(
        name="BookTwo", classifier=book,
        slots=[AttributeLink(attribute=title, value=DataValue(classifier=StringType, value="Book Two")),
               AttributeLink(attribute=pages, value=DataValue(classifier=IntegerType, value=200))])})

    project = Project(name="MultiObjectProject", models=[dm, om1, om2], owner="tester",
                      metadata=Metadata(description="banner immunity"))

    file_path = str(tmp_path / "multi_object.py")
    project_to_code(project, file_path)
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _object_diagrams(code):
    return project_to_json(code)["diagrams"].get("ObjectDiagram", [])


def test_baseline_two_object_diagrams(tmp_path):
    """A clean 1-domain + 2-object project imports as exactly two object diagrams."""
    code = _multi_object_project_code(tmp_path)
    object_diagrams = _object_diagrams(code)

    assert len(object_diagrams) == 2
    assert {d["title"] for d in object_diagrams} == {"ObjectModelOne", "ObjectModelTwo"}
    for d in object_diagrams:
        assert d["model"]["elements"], f"object diagram {d['title']!r} came back empty"


def test_stray_duplicate_object_banner_is_ignored(tmp_path):
    """Injecting a stray ``# OBJECT MODEL 3 #`` banner must NOT add a diagram.

    The number of object diagrams is fixed by ``models=[...]`` (two entries), so a
    third banner — with no corresponding model variable in the list — is inert.
    """
    code = _multi_object_project_code(tmp_path)

    stray_banner = (
        "####################################\n"
        '# OBJECT MODEL 3: "Ghost" #\n'
        "####################################\n\n"
    )
    poisoned = code.replace("# PROJECT DEFINITION #", stray_banner + "# PROJECT DEFINITION #")
    # Sanity: the poison actually introduced a third OBJECT MODEL banner.
    assert len(re.findall(r"#\s*OBJECT\s+MODEL[^\n#]*#", poisoned, flags=re.IGNORECASE)) == 3

    object_diagrams = _object_diagrams(poisoned)
    assert len(object_diagrams) == 2, [d["title"] for d in object_diagrams]
    assert {d["title"] for d in object_diagrams} == {"ObjectModelOne", "ObjectModelTwo"}
    for d in object_diagrams:
        assert d["model"]["elements"], f"object diagram {d['title']!r} came back empty"


def test_leftover_import_only_block_is_ignored(tmp_path):
    """A leftover import-only block (no model in ``models=[...]``) must not add a diagram."""
    code = _multi_object_project_code(tmp_path)

    leftover = (
        "####################################\n"
        "# OBJECT MODEL #\n"
        "####################################\n"
        "from besser.BUML.metamodel.object import ObjectModel\n"
        "import datetime\n\n"
    )
    poisoned = code.replace("# PROJECT DEFINITION #", leftover + "# PROJECT DEFINITION #")

    object_diagrams = _object_diagrams(poisoned)
    assert len(object_diagrams) == 2
    for d in object_diagrams:
        assert d["model"]["elements"], f"object diagram {d['title']!r} came back empty"


def test_renumbered_banners_do_not_change_diagram_set(tmp_path):
    """Reformatting/renumbering the banners leaves the diagram set untouched."""
    code = _multi_object_project_code(tmp_path)

    # Rewrite every "# OBJECT MODEL N: "X" #" banner to a bare, wrongly-numbered one.
    poisoned = re.sub(
        r'#\s*OBJECT\s+MODEL[^\n#]*#',
        "# OBJECT MODEL 99 #",
        code,
        flags=re.IGNORECASE,
    )

    object_diagrams = _object_diagrams(poisoned)
    # Titles still come from each ObjectModel(name=...) call, not the banner text.
    assert len(object_diagrams) == 2
    assert {d["title"] for d in object_diagrams} == {"ObjectModelOne", "ObjectModelTwo"}
