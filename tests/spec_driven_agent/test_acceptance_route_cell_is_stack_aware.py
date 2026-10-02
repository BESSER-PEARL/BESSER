"""The route cell may only report an absence it actually measured.

``build_endpoint_manifest`` parses FastAPI decorators out of ``.py`` files. On
an Express, Django, Spring or axum backend it returns ``""``, so
``any(f in manifest_low ...)`` was False for every entity and the matrix
reported "no backend REST route" for routes the app genuinely serves -- one
fabricated finding per class, on every non-Python stack, recorded in the recipe
and shown to the model.

Stack metadata exists precisely so the agent can target those stacks
(``detect_idiom_stack`` covers .NET, Rust, Kotlin, Go), so this is not a
hypothetical configuration.

The cell is now None (unmeasured) when nothing was parsed. A measured absence
-- a manifest that exists and does not mention this entity -- must still report,
which is the whole point of the check.
"""
import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.validation.acceptance import (
    build_acceptance_matrix, matrix_issues,
)

StringType = PrimitiveDataType("str")

_PAGE = "export default function RoomPage(){ return fetch('/room',{method:'POST'}) }"

_FASTAPI = (
    "from fastapi import APIRouter\n"
    'router = APIRouter(prefix="/room")\n'
    '@router.get("/")\ndef list_rooms():\n    return []\n'
    '@router.post("/")\ndef create_room(p: dict):\n    return p\n'
)
# Each genuinely serves GET+POST /room; none is parsed by the manifest builder.
_OTHER_STACKS = {
    "express": ("server.js",
                "const app=require('express')();\n"
                "app.get('/room', listRooms);\napp.post('/room', createRoom);\n"),
    "django": ("urls.py",
               "urlpatterns=[path('room/', RoomList.as_view())]\n"),
    "spring": ("src/main/java/RoomController.java",
               '@RestController class RoomController {\n'
               '  @GetMapping("/room") List<Room> all(){}\n'
               '  @PostMapping("/room") Room create(){}\n}\n'),
    "axum": ("src/main.rs",
             'Router::new().route("/room", get(list_rooms).post(create_room));\n'),
}


def _model(*names):
    classes = set()
    for n in names:
        c = Class(name=n)
        c.attributes = {Property(name="name", type=StringType)}
        classes.add(c)
    return DomainModel(name="Test", types=classes)


def _workspace(tmp_path, files):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize("stack", sorted(_OTHER_STACKS))
def test_an_unparsed_backend_leaves_the_route_cell_unmeasured(tmp_path, stack):
    """The regression: every one of these used to report a missing route."""
    rel, body = _OTHER_STACKS[stack]
    ws = _workspace(tmp_path, {rel: body, "src/RoomPage.jsx": _PAGE})

    matrix = build_acceptance_matrix(ws, _model("Room"))

    assert matrix["Room"]["route"] is None, f"{stack}: route was measured, but nothing parsed it"
    assert not any("REST route" in i for i in matrix_issues(matrix)), matrix_issues(matrix)


# Express is excluded below on purpose: its server.js carries a frontend
# extension, so the matrix scans the backend's own app.post as a frontend create.
# That makes the create cell more permissive, never more accusing, so it is not
# the failure mode this file is about.
@pytest.mark.parametrize("stack", ["axum", "django", "spring"])
def test_the_other_two_cells_are_still_measured(tmp_path, stack):
    """Only the route signal is stack-bound. Dropping all three would hide real gaps."""
    rel, body = _OTHER_STACKS[stack]
    ws = _workspace(tmp_path, {rel: body, "src/RoomPage.jsx": "export default () => <div/>;"})

    matrix = build_acceptance_matrix(ws, _model("Room"))

    assert matrix["Room"]["page"] is True
    assert matrix["Room"]["create"] is False
    assert any("no frontend create path" in i for i in matrix_issues(matrix))


def test_a_measured_absence_still_reports(tmp_path):
    """The check's reason for existing: a parsed manifest that omits the entity."""
    ws = _workspace(tmp_path, {"backend/room.py": _FASTAPI, "src/RoomPage.jsx": _PAGE})

    matrix = build_acceptance_matrix(ws, _model("Room", "Author"))

    assert matrix["Room"]["route"] is True
    assert matrix["Author"]["route"] is False
    assert any("no backend REST route" in i and "Author" in i
               for i in matrix_issues(matrix))


def test_it_stays_advisory(tmp_path):
    """Report-only by design: a scoped UI is a legitimate choice, not a defect."""
    from besser.spec_driven_agent.validation.issues import _classify_issue

    ws = _workspace(tmp_path, {"backend/room.py": _FASTAPI})
    issues = matrix_issues(build_acceptance_matrix(ws, _model("Author")))

    assert issues
    assert all(_classify_issue(i).severity == "warning" for i in issues)
