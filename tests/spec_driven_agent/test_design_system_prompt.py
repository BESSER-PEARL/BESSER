"""The design-system prompt section, for runs whose GUI model carries a design.

The Phase-2 prompt told the agent to write "one shared stylesheet", to define
concrete CSS for a theme, to ``write_file`` a page back after refused edits and,
in class-driven runs, to treat the GUI as a "loose hint" - against a generated
``design.css`` that six live runs never needed to touch. With a design, the
section names the stylesheet's real classes and switches those rules off.
"""

from __future__ import annotations

from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.design_system import design_system_section
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.test_design_fidelity import (
    APP,
    DESIGN_CSS,
    SRC,
    _Client,
    _designed_app,
    _write,
)


# ---------------------------------------------------------------- prompt section

def test_no_design_renders_no_section(tmp_path):
    _write(str(tmp_path), {f"{SRC}/App.tsx": APP})
    assert design_system_section(str(tmp_path)) == ""
    assert design_system_section(str(tmp_path), SimpleNamespace(stylesheet="")) == ""
    assert design_system_section(None, None) == ""


def test_section_lists_the_real_classes_and_variables(tmp_path):
    section = design_system_section(_designed_app(tmp_path))

    assert f"`{SRC}/design.css` is the design's source of truth" in section
    assert "primary, accent, space-md" in section
    for name in ("`ds-field`", "`ds-label`", "`ds-input`", "`ds-btn-primary`", "`hotel-shell`"):
        assert name in section
    # A compound selector's second class is a modifier, not a class of its own.
    assert "`hotel-btn` (+outline, danger)" in section
    assert "`outline`" not in section
    assert "a field is `ds-field` > `ds-label` + `ds-input`" in section
    assert "Never edit design.css" in section
    assert "never\n  `write_file` over an existing page" in section
    assert "`MethodButton`" in section and "`TableBlock`" in section


def test_section_falls_back_to_the_gui_model_stylesheet(tmp_path):
    """No design.css in the scaffold, but the GUI model carries the design."""
    section = design_system_section(str(tmp_path), SimpleNamespace(stylesheet=DESIGN_CSS))

    assert "add it unchanged as `src/design.css`" in section
    assert ".detail-grid{display:grid}" in section
    assert "`detail-grid`" in section


# ---------------------------------------------------------------- prompt wiring

def _prompt(design_system="", primary_kind="class"):
    gui = SimpleNamespace(modules=[SimpleNamespace(
        name="Main", screens=[SimpleNamespace(name="Home", view_elements=set())])])
    return build_system_prompt(
        domain_model=DomainModel(name="Hotel", types={Class(name="Booking")}),
        gui_model=gui, agent_model=None, inventory="Generated 5 files",
        instructions="Build the hotel app", max_turns=20,
        primary_kind=primary_kind, design_system=design_system,
    )


def test_without_a_design_the_prompt_keeps_its_stylesheet_rules():
    prompt = _prompt()
    assert "LOOSE HINT" in prompt
    assert "One shared stylesheet applied across the whole app" in prompt
    assert "define them as concrete CSS" in prompt
    assert "## Design system" not in prompt


def test_a_design_replaces_the_conflicting_rules(tmp_path):
    section = design_system_section(_designed_app(tmp_path))
    prompt = _prompt(section)

    assert section in prompt
    assert "LOOSE HINT" not in prompt
    assert "finished visual design (see Design system)" in prompt
    assert "One shared stylesheet" not in prompt
    assert "no second stylesheet" in prompt
    assert "define them as concrete CSS" not in prompt
    assert "overriding `--ds-*` values" in prompt
    assert "except a designed page (see Design system)" in prompt
    assert "(a designed page:\n   `replace_file_lines` on the block instead)" in prompt


def test_the_orchestrator_hands_the_design_to_the_prompt(tmp_path):
    workspace = _designed_app(tmp_path)
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Hotel", types={Class(name="Booking")}),
        output_dir=workspace, enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    assert "## Design system (from the GUI model)" in orch._build_system_prompt("hotel")
