"""The request intro claims only what the Phase 1 generator produced.

r9-web1: the intro said the generator handled "CRUD, ORM, schemas, pages" while
checklist item 1 said no frontend existed. The "pages" claim became conditional
in a later round; the "CRUD, ORM, schemas" claim still went out for every
generator, including ones that write plain classes.
"""
import pytest

from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from tests.spec_driven_agent.shelf_app import shelf_model

FASTAPI = ("Generator `generate_fastapi_backend` produced 9 files:\n  - main_api.py\n  - routers/book.py\n\n"
           "Backend: FastAPI with SQLAlchemy ORM + Pydantic schemas (modular)")
PYTHON_CLASSES = "Generator `generate_python_classes` produced 1 files:\n  - classes.py: Book@4, Library@20"


def _request_part(inventory: str) -> str:
    prompt = build_system_prompt(domain_model=shelf_model(), gui_model=None, agent_model=None, inventory=inventory,
                                 instructions="Add a search box for books.", max_turns=40)
    return prompt.split("## User request", 1)[1].split("> Add a search box", 1)[0]


@pytest.mark.parametrize("inventory", [FASTAPI, PYTHON_CLASSES])
def test_the_intro_names_no_part_the_generator_may_not_have_made(inventory):
    intro = _request_part(inventory)
    for claim in ("pages", "CRUD", "ORM", "schemas"):
        assert claim not in intro, intro


def test_the_intro_still_says_a_scaffold_exists():
    assert "generator" in _request_part(PYTHON_CLASSES).lower()
    assert "No generator ran" in _request_part("")
