"""The design-system prompt section, for runs whose GUI model carries a design.

The Phase-2 prompt told the agent to write "one shared stylesheet", to define
concrete CSS for a theme, to ``write_file`` a page back after refused edits and,
in class-driven runs, to treat the GUI as a "loose hint" - against a generated
``design.css`` that six live runs never needed to touch. With a design, the
section names the stylesheet's real classes and switches those rules off.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.design_system import design_system_section
from besser.spec_driven_agent.agent.prompt_builder import build_system_prompt
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

SRC = "web_app/frontend/src"

DESIGN_CSS = """\
/* Design stylesheet carried over from the GUI model (GUIModel.stylesheet). */
:root{--ds-primary:#17324D;--ds-accent:#218B83;--ds-space-md:0.75rem}
.ds-page{margin:0}
.ds-card{background:#fff}
.ds-field{display:flex}
.ds-label{font-weight:600}
.ds-input{width:100%}
.ds-btn{padding:0.5rem}
.ds-btn-primary{background:#17324D}
.hotel-shell{min-height:100vh}
.page-head{display:flex}
.detail-grid{display:grid}
.record-card{border:1px solid #eee}
.hotel-btn{border:0}
.hotel-btn.outline{background:#fff}
.hotel-btn.danger{background:#B84A4A}
.guest-nav a{margin-left:22px}
"""

INDEX = "import './index.css';\nimport App from './App';\nimport './design.css';\n"

APP = """\
import { Routes, Route, Navigate } from "react-router-dom";
import BookingDetail from "./pages/BookingDetail";
export default function App() {
  return <Routes><Route path="/booking" element={<BookingDetail />} />
    <Route path="/" element={<Navigate to="/booking" replace />} /></Routes>;
}
"""

BUTTONS = """\
          <MethodButton id="b1" className="hotel-btn" endpoint="/booking/{booking_id}/methods/check_in/" label="Check in" />
          <MethodButton id="b2" className="hotel-btn outline" endpoint="/booking/{booking_id}/methods/check_out/" label="Check out" />
          <MethodButton id="b3" className="hotel-btn danger" endpoint="/booking/{booking_id}/methods/cancel/" label="Cancel" />
"""

DETAIL = """\
import React from "react";
const BookingDetail: React.FC = () => (
  <div id="page" className="ds-page">
    <nav className="guest-nav"><a href="/booking">Bookings</a></nav>
    <main className="hotel-shell">
      <header className="page-head"><h1>Booking</h1></header>
      <section className="detail-grid">
        <article className="record-card ds-card">
          <TableBlock id="table-booking" dataBinding={{"entity": "Booking"}} />
        </article>
        <div className="action-row">
%s        </div>
      </section>
    </main>
  </div>
);
export default BookingDetail;
""" % BUTTONS


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - no test here calls the LLM
        raise AssertionError("no LLM call expected")


def _write(root, files):
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)


def _designed_app(tmp_path, **overrides):
    files = {
        f"{SRC}/design.css": DESIGN_CSS,
        f"{SRC}/index.tsx": INDEX,
        f"{SRC}/App.tsx": APP,
        f"{SRC}/pages/BookingDetail.tsx": DETAIL,
    }
    files.update({f"{SRC}/{k}": v for k, v in overrides.items()})
    _write(str(tmp_path), files)
    return str(tmp_path)


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
