"""A Menu's items keep the editor's order in the exported B-UML script (#679).

``Menu.menuItems`` is a set of identity-hashed ``MenuItem``s, so iterating it
directly wrote the items in an order that changed between processes: the same
project exported three times gave three different ``.py`` files. The builder
now writes them in creation order (their timestamps), which is the editor's.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

LABELS = ["Home", "Products", "Pricing", "About", "Contact", "Blog", "Careers", "Help"]

EXPORT = textwrap.dedent('''
    import sys
    from besser.BUML.metamodel.gui import GUIModel, Module, Screen
    from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_processors.component_parsers import (
        parse_menu,
    )

    labels = sys.argv[2].split(",")
    nav = {"type": "navbar", "components": [
        {"type": "link", "attributes": {"href": "/" + label.lower()},
         "components": [{"type": "textnode", "content": label}]}
        for label in labels]}
    menu = parse_menu(nav, None, "main_menu", {"tagName": None})
    screen = Screen(name="Home", description="", view_elements={menu})
    gui = GUIModel(name="App", package="app", versionCode="1", versionName="1",
                   modules={Module(name="Main", screens={screen})}, description="")
    gui_model_to_code(gui, sys.argv[1])
''')

SEEDS = ("0", "1", "2", "3")


def _export(path: Path, seed: str) -> bytes:
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)}
    subprocess.run([sys.executable, "-c", EXPORT, str(path), ",".join(LABELS)], env=env, check=True,
                   cwd=repo, capture_output=True)
    return path.read_bytes()


def test_menu_items_are_written_in_editor_order(tmp_path):
    source = _export(tmp_path / "gui.py", "0").decode()
    lines = [line for line in source.splitlines() if "= MenuItem(" in line]
    assert [line.split('label="')[1].split('"')[0] for line in lines] == LABELS


def test_menu_export_is_identical_across_processes(tmp_path):
    runs = {seed: _export(tmp_path / f"gui_{seed}.py", seed) for seed in SEEDS}
    assert len(set(runs.values())) == 1, "the exported GUI script differs between runs"
