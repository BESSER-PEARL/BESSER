"""The ``design regression:`` check: the GUI design survives the agent.

Evidence (six live runs on an AI-designed hotel GUI, 2026-09-25): design.css
was never edited, but one GPT run overwrote two designed pages with
``write_file`` - BookingDetail.tsx 358 -> 37 lines, losing the nav, the layout
wrappers and five ``MethodButton``s - and another filled an empty form with bare
``<label>/<input>`` instead of the stylesheet's ``ds-field``/``ds-input``.
Every other validator stayed green.
"""

from __future__ import annotations

import json
import os

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.validation.design_fidelity import (
    BASELINE_FILENAME,
    capture_design_baseline,
    collect_design_fidelity_issues,
    save_design_baseline,
)
from besser.spec_driven_agent.validation.frontend_contract import (
    collect_frontend_contract_issues,
)
from besser.spec_driven_agent.validation.issues import _classify_issue

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

# gpt1's BookingDetail after its write_file: no nav, no wrappers, no generated
# components, bare controls and inline styles.
REWRITTEN = """\
import React, { useState } from "react";
const BookingDetail: React.FC = () => {
  const [id, setId] = useState("");
  const act = (m: string) => fetch(`/booking/${id}/methods/${m}/`, { method: "POST" });
  return <main className="assistant-main" style={{ maxWidth: 760, color: "#17324D" }}>
    <h1>Booking</h1>
    <label>Booking id <input value={id} onChange={(e) => setId(e.target.value)} /></label>
    <button className="hotel-btn" onClick={() => act("check_in")}>Check in</button>
  </main>;
};
export default BookingDetail;
"""


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


def _edit(root, rel, text):
    _write(root, {f"{SRC}/{rel}": text})


def _blockers(issues):
    return [i for i in issues if _classify_issue(i).severity == "blocker"]


def _warnings(issues):
    return [i for i in issues if _classify_issue(i).severity == "warning"]


# ---------------------------------------------------------------- validator

def test_silent_without_a_design(tmp_path):
    _write(str(tmp_path), {f"{SRC}/App.tsx": APP, f"{SRC}/pages/X.tsx": REWRITTEN})
    assert capture_design_baseline(str(tmp_path)) is None
    assert save_design_baseline(str(tmp_path)) is None
    assert not os.path.exists(os.path.join(str(tmp_path), BASELINE_FILENAME))
    assert collect_design_fidelity_issues(str(tmp_path)) == []


def test_an_untouched_design_is_clean(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    assert collect_design_fidelity_issues(root) == []


def test_a_rewritten_page_is_a_blocker_naming_everything_it_lost(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx", REWRITTEN)

    issues = collect_design_fidelity_issues(root)
    [blocker] = _blockers(issues)

    assert blocker.startswith(f"design regression: {SRC}/pages/BookingDetail.tsx:")
    assert "its <header> and <nav> are gone" in blocker
    assert "design classes are gone" in blocker
    assert "3 MethodButton, 1 TableBlock" in blocker
    assert "Its Phase-1 version is saved at .besser_design_1_BookingDetail.tsx.txt" in blocker
    [warning] = _warnings(issues)
    assert warning.startswith("design drift:")
    assert "+1 inline style" in warning and "+1 hard-coded hex" in warning
    # <label> and <input> are bare; the button kept its hotel-btn class.
    assert "+2 form control(s) without a design class" in warning


def test_the_saved_copy_restores_the_page(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx", REWRITTEN)
    with open(os.path.join(root, ".besser_design_1_BookingDetail.tsx.txt"), encoding="utf-8") as fh:
        _edit(root, "pages/BookingDetail.tsx", fh.read())
    assert collect_design_fidelity_issues(root) == []


def test_the_saved_copies_are_invisible_to_source_scanners(tmp_path):
    """A copy of the scaffold's empty form must not raise a dead-form blocker."""
    form = DETAIL.replace("<section", '<form onSubmit={(e) => { e.preventDefault(); }}></form><section')
    root = _designed_app(tmp_path, **{"pages/BookingDetail.tsx": form})
    before = collect_frontend_contract_issues(root)
    save_design_baseline(root)
    assert collect_frontend_contract_issues(root) == before


def test_losing_only_the_nav_is_a_blocker(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx",
          DETAIL.replace('<nav className="guest-nav"><a href="/booking">Bookings</a></nav>', ""))
    [blocker] = collect_design_fidelity_issues(root)
    assert "its <nav> is gone" in blocker and "design classes" not in blocker


def test_a_nav_moved_into_a_shared_layout_is_not_lost(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx",
          DETAIL.replace('<nav className="guest-nav"><a href="/booking">Bookings</a></nav>', ""))
    _edit(root, "components/Layout.tsx",
          'export const Layout = () => <nav className="guest-nav"><a href="/">Home</a></nav>;\n')
    assert collect_design_fidelity_issues(root) == []


def test_removed_method_buttons_are_a_blocker(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace(BUTTONS, '          <button className="hotel-btn">Check in</button>\n'))
    [blocker] = collect_design_fidelity_issues(root)
    assert "generated components were removed (3 MethodButton)" in blocker


def test_components_moved_to_a_child_file_or_mapped_are_kept(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace(BUTTONS, "          <BookingActions />\n"))
    _edit(root, "components/BookingActions.tsx", f"export const BookingActions = () => <>\n{BUTTONS}</>;\n")
    assert collect_design_fidelity_issues(root) == []

    mapped = (
        '          {["/booking/{booking_id}/methods/check_in/", "/booking/{booking_id}/methods/check_out/",\n'
        '            "/booking/{booking_id}/methods/cancel/"].map((e) =>\n'
        '            <MethodButton key={e} className="hotel-btn" endpoint={e} label={e} />)}\n'
    )
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace(BUTTONS, mapped))
    os.remove(os.path.join(root, SRC, "components/BookingActions.tsx"))
    assert collect_design_fidelity_issues(root) == []


def test_a_rewrite_that_keeps_the_design_is_clean(tmp_path):
    """gpt2 rewrote BookingDetail with write_file but kept every class, the nav
    and the MethodButtons (one added): that is not a regression."""
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    extra = '          <MethodButton id="b4" className="hotel-btn" endpoint="/booking/{booking_id}/methods/invoice/" label="Invoice" />\n'
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace(BUTTONS, BUTTONS + extra).replace("  ", "    "))
    assert collect_design_fidelity_issues(root) == []


def test_a_small_class_loss_is_tolerated(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace("record-card ds-card", "record-card"))
    assert collect_design_fidelity_issues(root) == []


def test_removing_the_design_import_is_a_blocker(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "index.tsx", "import './index.css';\nimport App from './App';\n")
    [blocker] = collect_design_fidelity_issues(root)
    assert blocker.startswith("design regression: nothing imports")


def test_deleting_or_emptying_design_css_is_a_blocker(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    _edit(root, "design.css", "/* cleaned up */\n")
    [blocker] = collect_design_fidelity_issues(root)
    assert "design.css was deleted or emptied" in blocker
    assert "saved at .besser_design_0_design.css.txt" in blocker

    os.remove(os.path.join(root, SRC, "design.css"))
    assert "deleted or emptied" in collect_design_fidelity_issues(root)[0]


def test_deleting_a_designed_page_is_a_blocker(tmp_path):
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    os.remove(os.path.join(root, SRC, "pages/BookingDetail.tsx"))
    issues = collect_design_fidelity_issues(root)
    assert any(i.startswith(f"design regression: designed page {SRC}/pages/BookingDetail.tsx was deleted")
               for i in issues)


def test_bare_form_controls_are_a_warning_not_a_blocker(tmp_path):
    """gpt3 filled the empty scaffold form with bare <label>/<input>/<button>."""
    root = _designed_app(tmp_path)
    save_design_baseline(root)
    bare = '<form onSubmit={save}><label>Name</label><input name="n" /><button type="submit">Save</button></form>\n'
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace("        <div className=\"action-row\">", bare + '        <div className="action-row">'))
    issues = collect_design_fidelity_issues(root)
    assert _blockers(issues) == []
    [warning] = issues
    assert "+3 form control(s) without a design class" in warning

    styled = ('<form onSubmit={save}><div className="ds-field"><label className="ds-label">Name</label>'
              '<input className="ds-input" name="n" /></div><button className="ds-btn ds-btn-primary">Save</button></form>\n')
    _edit(root, "pages/BookingDetail.tsx", DETAIL.replace("        <div className=\"action-row\">", styled + '        <div className="action-row">'))
    assert collect_design_fidelity_issues(root) == []


def test_severity_prefixes():
    assert _classify_issue("design regression: x").severity == "blocker"
    assert _classify_issue("design drift: x").severity == "warning"


def test_phase_3_reports_the_regression_and_phase_1_captures_the_baseline(tmp_path):
    root = _designed_app(tmp_path)
    orch = LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="Hotel", types={Class(name="Booking")}),
        output_dir=root, enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    orch._capture_design_baseline()
    with open(os.path.join(root, BASELINE_FILENAME), encoding="utf-8") as fh:
        assert list(json.load(fh)["pages"]) == [f"{SRC}/pages/BookingDetail.tsx"]
    _edit(root, "pages/BookingDetail.tsx", REWRITTEN)

    blockers = [i.message for i in orch._collect_validation_issues() if i.severity == "blocker"]
    assert [b for b in blockers if b.startswith("design regression:")], blockers
