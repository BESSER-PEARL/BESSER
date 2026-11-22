"""Prompt sections must describe the run they are shown to.

* Rule 15 ("a domain-model app needs a COMPLETE, NAVIGABLE CRUD frontend")
  was shown to backend-only runs, ordering a React frontend nobody asked for.
* A GUI stylesheet over 16k characters was cut mid-rule with no marker, under
  the instruction "add it unchanged as src/design.css".
"""
from types import SimpleNamespace

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.agent import design_system
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt

CRUD_RULE = "COMPLETE, NAVIGABLE CRUD frontend"
BACKEND_INVENTORY = (
    "Generator `generate_fastapi_backend` produced 2 files:\n"
    "  - main_api.py (2,000 bytes)\n  - sql_alchemy.py (900 bytes)"
)


def _model() -> DomainModel:
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return DomainModel(name="M", types={user})


def _prompt(instructions, inventory=BACKEND_INVENTORY, gui_model=None):
    return build_system_prompt(
        _model(), gui_model, None, inventory=inventory,
        instructions=instructions, max_turns=10,
    )


def test_a_backend_only_run_is_not_told_to_build_a_frontend():
    assert CRUD_RULE not in _prompt("Add a /stats endpoint returning user counts")


def test_a_web_app_request_keeps_the_crud_rule():
    assert CRUD_RULE in _prompt("Build a library web app")


def test_a_scaffold_with_a_frontend_keeps_the_crud_rule():
    inventory = BACKEND_INVENTORY + "\n  - frontend/src/App.tsx (1,200 bytes)"
    assert CRUD_RULE in _prompt("Add a /stats endpoint", inventory=inventory)


def test_a_gui_model_keeps_the_crud_rule():
    assert CRUD_RULE in _prompt("Add a /stats endpoint", gui_model=SimpleNamespace(name="GUI", modules=set()))


def test_a_cut_design_stylesheet_says_it_was_cut(monkeypatch):
    monkeypatch.setattr(design_system, "_MAX_INLINE_CSS", 200)
    css = ":root{--ds-primary:#17324D}\n" + "".join(
        f".ds-rule-{n}{{margin:{n}px}}\n" for n in range(40))
    gui = SimpleNamespace(stylesheet=css)

    section = design_system.design_system_section(None, gui)

    block = section.split("```css\n", 1)[1].split("\n```", 1)[0]
    assert block.rstrip().endswith("}"), "cut in the middle of a rule"
    assert "truncated" in section and str(len(css)) in section, section
