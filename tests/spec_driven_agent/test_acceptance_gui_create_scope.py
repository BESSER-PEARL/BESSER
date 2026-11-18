"""With a GUI model, the create cell follows the screens the user modelled.

A class the GUI model gives no create component (a table with action buttons,
a form or a create button bound to it) is not expected to be creatable from
the frontend: its records may come from the backend (an import, a parent's
endpoint). Its ``create`` cell is ``None``, not a "no frontend create path"
warning; a class the GUI model does offer to create keeps the check.
"""
from __future__ import annotations

from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.gui.binding import DataBinding
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix, matrix_issues

_STR = PrimitiveDataType("str")


def _model() -> DomainModel:
    classes = set()
    for name in ("Order", "OrderLine"):
        cls = Class(name=name)
        cls.attributes = {Property(name="label", type=_STR)}
        classes.add(cls)
    return DomainModel(name="Shop", types=classes)


def _gui(model: DomainModel, **tables: bool) -> GUIModel:
    """One screen with a table per named class; the value is its action_buttons flag."""
    by_name = {c.name: c for c in model.get_classes()}
    elements = {Table(f"{name}Table", action_buttons=buttons,
                      data_binding=DataBinding(domain_concept=by_name[name]))
                for name, buttons in tables.items()}
    screen = Screen(name="Home", description="", view_elements=elements)
    return GUIModel(name="ShopGUI", package="shop", versionCode="1", versionName="1.0",
                    modules={Module(name="Main", screens={screen})}, description="")


def _workspace(tmp_path) -> None:
    files = {
        "backend/routers/order.py": (
            "from fastapi import APIRouter\n"
            'router = APIRouter(prefix="/orders")\n'
            '@router.get("/")\ndef list_orders():\n    return []\n'
            '@router.get("/orderlines/")\ndef list_lines():\n    return []\n'),
        "frontend/src/pages/Orders.tsx": (
            "export default function Orders() {\n"
            "  // shows each Order with its OrderLine rows, read only\n"
            "  return <div>Orders</div>;\n}\n"),
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_a_class_the_gui_model_offers_no_create_for_is_not_expected_to_be_creatable(tmp_path):
    _workspace(tmp_path)
    model = _model()
    matrix = build_acceptance_matrix(str(tmp_path), model, gui_model=_gui(model, Order=True))
    assert matrix["OrderLine"]["create"] is None
    assert not any("OrderLine" in issue for issue in matrix_issues(matrix))


def test_a_read_only_table_does_not_ask_for_a_create_path(tmp_path):
    _workspace(tmp_path)
    model = _model()
    matrix = build_acceptance_matrix(str(tmp_path), model, gui_model=_gui(model, Order=True, OrderLine=False))
    assert matrix["OrderLine"]["create"] is None


def test_a_class_the_gui_model_offers_to_create_keeps_the_warning(tmp_path):
    _workspace(tmp_path)
    model = _model()
    matrix = build_acceptance_matrix(str(tmp_path), model, gui_model=_gui(model, Order=True))
    assert matrix["Order"]["create"] is False
    assert any(issue.startswith("acceptance: entity Order ") and "no frontend create path" in issue
               for issue in matrix_issues(matrix))


def test_without_a_gui_model_every_class_is_still_checked(tmp_path):
    _workspace(tmp_path)
    matrix = build_acceptance_matrix(str(tmp_path), _model())
    assert matrix["OrderLine"]["create"] is False
