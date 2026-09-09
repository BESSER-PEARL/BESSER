"""Tests for the modular FastAPI layout produced by BackendGenerator.

Historically BackendGenerator emitted a single ~1,600-line ``main_api.py``
containing the FastAPI app, every CRUD/relationship/method endpoint for
every class, and the database session setup, all in one namespace. This
module verifies the split into:

    main_api.py        - slim app setup + router includes (keeps its
                          historical filename/``app`` object so existing
                          tooling that runs ``uvicorn main_api:app`` keeps
                          working unmodified)
    database.py         - shared engine/session setup + get_db dependency
    bal_stdlib.py        - BESSER Action Language standard-library helpers
    routers/<class>.py  - one APIRouter per resource, with all of that
                          class's CRUD/relationship/method endpoints

...and that the resulting files are valid, importable-shaped Python across
a few structurally different models (multi-class with associations, and a
single self-associated class).
"""

import ast
import os

import pytest

from besser.generators.backend import BackendGenerator
from besser.generators.backend.api_generator import cross_router_calls


# ---------------------------------------------------------------------------
# File layout + router wiring
# ---------------------------------------------------------------------------

def test_modular_file_layout_and_router_wiring(library_book_author_model, tmp_path):
    """BackendGenerator must emit a slim main_api.py, a shared database.py,
    bal_stdlib.py, and one routers/<class>.py per class - and main_api.py
    must wire every router in with app.include_router()."""
    output_dir = str(tmp_path / "output")
    generator = BackendGenerator(model=library_book_author_model, output_dir=output_dir)
    generator.generate()

    # Top-level files
    assert os.path.isfile(os.path.join(output_dir, "main_api.py"))
    assert os.path.isfile(os.path.join(output_dir, "database.py"))
    assert os.path.isfile(os.path.join(output_dir, "bal_stdlib.py"))
    assert os.path.isfile(os.path.join(output_dir, "sql_alchemy.py"))
    assert os.path.isfile(os.path.join(output_dir, "pydantic_classes.py"))

    # One router per class
    routers_dir = os.path.join(output_dir, "routers")
    assert os.path.isfile(os.path.join(routers_dir, "__init__.py"))
    for class_name in ("library", "book", "author"):
        router_file = os.path.join(routers_dir, f"{class_name}.py")
        assert os.path.isfile(router_file), f"Missing router file for {class_name}"
        with open(router_file, "r", encoding="utf-8") as f:
            router_code = f.read()
        assert "router = APIRouter()" in router_code
        assert f'@router.get("/{class_name}/"' in router_code
        assert f'@router.post("/{class_name}/"' in router_code

    with open(os.path.join(output_dir, "main_api.py"), "r", encoding="utf-8") as f:
        api_code = f.read()

    # main_api.py no longer defines any routes itself (besides the fixed
    # system endpoints) - it only wires up the per-class routers.
    assert "@app.get(\"/name1/\"" not in api_code
    for class_name in ("library", "book", "author"):
        assert f"from routers import {class_name} as {class_name}_router" in api_code
        assert f"app.include_router({class_name}_router.router)" in api_code

    # System endpoints stay in main_api.py
    assert '@app.get("/", tags=["System"])' in api_code
    assert '@app.get("/health", tags=["System"])' in api_code
    assert '@app.get("/statistics", tags=["System"])' in api_code


@pytest.mark.parametrize("model_fixture", ["library_book_author_model", "employee_self_assoc_model"])
def test_generated_files_parse_as_valid_python(model_fixture, tmp_path, request):
    """Every emitted .py file (main_api.py, database.py, bal_stdlib.py, and
    each routers/<class>.py) must be syntactically valid Python, for both a
    multi-class model with associations and a single self-associated class."""
    model = request.getfixturevalue(model_fixture)
    output_dir = str(tmp_path / "output")
    generator = BackendGenerator(model=model, output_dir=output_dir)
    generator.generate()

    py_files = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith(".py"):
                py_files.append(os.path.join(root, name))

    assert py_files, "Expected at least one generated .py file"
    for path in py_files:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source, filename=path)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path} failed to parse: {exc}")


# ---------------------------------------------------------------------------
# cross_router_calls() heuristic (used to import a sibling class's endpoint
# functions into a method-endpoint body without relying on module-level
# imports, which could deadlock on circular router references).
# ---------------------------------------------------------------------------

def test_cross_router_calls_ignores_same_class_calls():
    """A method's own class's endpoint functions are already in scope in its
    own router module, so no import should be generated for them."""
    code = "inst_to_update = _manager_object\nawait update_manager(inst_to_update.id, ManagerCreate(x=1), database)"
    assert cross_router_calls(code, "Manager", ["Manager", "Department"]) == []


def test_cross_router_calls_detects_other_class_functions():
    """A method body referencing another class's create/update/get functions
    should produce an import for that class's router module."""
    code = (
        "inst_to_update = (await get_department(_employee_object.department.id, database))\n"
        "await update_department(inst_to_update.id, DepartmentCreate(title='x'), database)\n"
    )
    result = cross_router_calls(code, "Employee", ["Employee", "Department"])
    assert ("department", "get_department") in result
    assert ("department", "update_department") in result
    # No spurious self-import
    assert not any(target == "employee" for target, _ in result)


def test_cross_router_calls_detects_method_and_relationship_helpers():
    code = (
        "x = (await execute_department_close(inst_to_update.id, {}, database))\n"
        "y = (await get_employees_of_department(inst_to_update.id, database))['employees']\n"
    )
    result = cross_router_calls(code, "Employee", ["Employee", "Department"])
    assert ("department", "execute_department_close") in result
    assert ("department", "get_employees_of_department") in result


def test_cross_router_calls_empty_for_no_method_code():
    assert cross_router_calls("", "Employee", ["Employee", "Department"]) == []
    assert cross_router_calls(None, "Employee", ["Employee", "Department"]) == []
