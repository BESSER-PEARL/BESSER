"""A citation of Python code locates whatever the file's layout; changed code still does not.

Live evidence (2026-11-14, claude-sonnet-5-5, runs r9-p81 and r9-p21): three
``requirement unverified`` blockers, each on code that does implement the
requirement when executed. The judge quoted a statement on one line; on disk
the harness's own ``ruff format`` had wrapped it (``patterns = (`` / call /
``)``), split a regex into two implicitly concatenated raw strings, or split a
route decorator over five lines. The text matcher compares lines, so it
refused all three. A quote that parses as Python is now also compared by its
syntax tree, and a decorator is located but proves nothing on its own.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence

# backend/services.py of run r9-p81, the cited parts verbatim (as ruff formatted them).
SERVICES = '''import re
from typing import List

STRONGLY_NEGATIVE_SENTIMENT = -0.6

AGGRESSIVE_RE = re.compile(
    r"\\b(kill|hate you|idiot|stupid|shut\\s*up|die|loser|ugly|moron|dumb|trash|worthless|"
    r"hurt you|find you|threat\\w*|destroy you)\\b",
    re.I,
)


def _detect_all(db: Session, account: SocialAccount) -> List[dict]:
    outgoing = db.query(Interaction).filter(Interaction.source_id == account.id).all()
    sent = db.query(Message).filter(Message.sender_id == account.id).all()
    reports = db.query(Report).filter(Report.reported_id == account.id).all()
    patterns = (
        db.query(ActivityPattern).filter(ActivityPattern.account_id == account.id).all()
    )
    found: List[dict] = []

    # R2 aggressive communication
    aggressive_msgs = [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]
    return found


def total_price(items: list) -> float:
    return (
        sum(item.price for item in items)
    )


def refuse_self_report(reporter_id: int, reported_id: int) -> None:
    if reporter_id == reported_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "an account cannot report "
                "itself"
            ),
        )


def flag_account(account) -> None:
    pass


def score_account(account) -> float: ...


def describe() -> str:
    """Describe the analysis."""
    return "analysis"


def create_booking(db, payload):
    booking = Booking(**payload)
    db.add(booking)
    db.commit()
    return booking
'''
OTHER = '''def unrelated():
    return 1
'''

# The two r9-p81 citations an analyst proved correct, verbatim from the recipe.
R1_CITATION_5 = "patterns = db.query(ActivityPattern).filter(ActivityPattern.account_id == account.id).all()"
R2_CITATION_1 = ('AGGRESSIVE_RE = re.compile(r"\\b(kill|hate you|idiot|stupid|shut\\s*up|die|loser|ugly|moron|'
                 'dumb|trash|worthless|hurt you|find you|threat\\w*|destroy you)\\b", re.I,)')
R2_CITATION_3 = 'aggressive_msgs = [m for m in sent if AGGRESSIVE_RE.search(m.content or "")]'


def _verdict(*citations, kind="action", text="The system detects aggressive communication"):
    evidence = [{"path": path, "line": line} for path, line in citations]
    return {"id": 1, "kind": kind, "text": text, "status": "implemented", "evidence": evidence}


def _check(tmp_path, *citations, **kwargs):
    (tmp_path / "backend").mkdir(exist_ok=True)
    (tmp_path / "backend" / "services.py").write_text(SERVICES, encoding="utf-8")
    (tmp_path / "backend" / "other.py").write_text(OTHER, encoding="utf-8")
    return verify_evidence([_verdict(*citations, **kwargs)], str(tmp_path))[0]


def _svc(line):
    return ("backend/services.py", line)


# --- the formatter's layout no longer hides real code -----------------------------------------

def test_a_statement_the_formatter_wrapped_in_parentheses_verifies(tmp_path):
    """r9-p81 R1 citation 5: on disk ``patterns = (`` / the call / ``)``."""
    verdict = _check(tmp_path, _svc(R1_CITATION_5), text="Analyse interaction patterns")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_regex_split_into_two_raw_strings_quoted_as_one_locates(tmp_path):
    """r9-p81 R2 citation 1: two implicitly concatenated raw strings, merged by the judge."""
    verdict = _check(tmp_path, _svc(R2_CITATION_1), _svc(R2_CITATION_3))
    assert verdict["status"] == "implemented", verdict.get("note")


def test_the_located_regex_still_proves_no_behaviour_on_its_own(tmp_path):
    """Located is not proving: a module-level declaration does not perform an action."""
    verdict = _check(tmp_path, _svc(R2_CITATION_1))
    assert verdict["status"] == "unverified"
    assert "does not perform the required behaviour" in verdict["note"]
    assert "not found" not in verdict["note"]


def test_a_small_function_quoted_on_one_line_verifies(tmp_path):
    verdict = _check(tmp_path, _svc("def total_price(items: list) -> float: return sum(item.price for item in items)"),
                     kind="computed", text="The total is the sum of the item prices")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_collapsed_multi_line_call_verifies(tmp_path):
    verdict = _check(tmp_path, _svc('raise HTTPException(status_code=409, detail="an account cannot report itself")'),
                     kind="rule", text="An account must not report itself")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_compound_statement_quoted_on_one_line_verifies(tmp_path):
    verdict = _check(tmp_path, _svc('if reporter_id == reported_id: raise HTTPException(status_code=409, '
                                    'detail="an account cannot report itself")'),
                     kind="rule", text="An account must not report itself")
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_run_of_consecutive_statements_verifies(tmp_path):
    verdict = _check(tmp_path, _svc("sent = db.query(Message).filter(Message.sender_id == account.id).all(); "
                                    "reports = db.query(Report).filter(Report.reported_id == account.id).all(); "
                                    "patterns = db.query(ActivityPattern).filter("
                                    "ActivityPattern.account_id == account.id).all()"))
    assert verdict["status"] == "implemented", verdict.get("note")


# --- what must still be refused ----------------------------------------------------------------

@pytest.mark.parametrize("citation", [
    pytest.param(_svc("aggressive = detect_aggression(account)"), id="invented"),
    pytest.param(("backend/other.py", R1_CITATION_5), id="real-statement-wrong-file"),
    pytest.param(_svc("# R2 aggressive communication"), id="comment-only"),
    pytest.param(_svc('"""Describe the analysis."""'), id="docstring"),
    pytest.param(_svc(R1_CITATION_5.replace("account_id", "owner_id")), id="one-token-changed"),
    pytest.param(_svc(R2_CITATION_1.replace("destroy you", "maim you")), id="one-regex-word-changed"),
    pytest.param(_svc(R1_CITATION_5.removesuffix(".all()")), id="call-dropped"),
    pytest.param(_svc("def flag_account(account) -> None: pass"), id="stub-pass"),
    pytest.param(_svc("def score_account(account) -> float: ..."), id="stub-ellipsis"),
])
def test_a_citation_that_is_not_the_code_stays_unverified(tmp_path, citation):
    verdict = _check(tmp_path, citation)
    assert verdict["status"] == "unverified", verdict


def test_every_citation_of_a_verdict_must_still_locate(tmp_path):
    """Four good citations do not carry an altered fifth (r9-p81 R1 with one token changed)."""
    verdict = _check(tmp_path, _svc(R2_CITATION_3), _svc("outgoing = db.query(Interaction).filter("
                                                         "Interaction.source_id == account.id).all()"),
                     _svc(R1_CITATION_5.replace("account_id", "owner_id")))
    assert verdict["status"] == "unverified"
    assert "citation 3" in verdict["note"]


# --- feedback for the citation-repair re-judge -------------------------------------------------

def test_the_note_shows_where_a_quote_leaves_a_long_line(tmp_path):
    """The r9-p81 note was cut at 80 characters, before the point where quote and file part."""
    verdict = _check(tmp_path, _svc(R2_CITATION_1.replace("worthless|hurt", "worthless|hurts")))
    note = verdict["note"]
    assert verdict["status"] == "unverified"
    assert 'worthless|"' in note, note
    assert "hurts you" in note, note


def test_the_note_names_the_closest_statement_when_the_first_line_is_short(tmp_path):
    """r9-p81 R1 got only "no source line matches": ``patterns = (`` starts no quote."""
    verdict = _check(tmp_path, _svc(R1_CITATION_5.replace("account_id", "owner_id")))
    note = verdict["note"]
    assert verdict["status"] == "unverified"
    assert "lines 17-19" in note, note
    assert "account_id" in note and "owner_id" in note, note


# --- the judge is told how to cite a multi-line statement --------------------------------------

def test_the_judge_cites_one_physical_line_of_a_multi_line_statement():
    assert "never rejoined" in ledger._JUDGE_SYSTEM_PROMPT
    evidence = ledger._SUBMIT_VERDICTS_TOOL["input_schema"]["properties"]["verdicts"]["items"]["properties"]["evidence"]
    assert "never the statement rejoined" in evidence["description"]


# --- review of round 7: the role check reads the file, not the quote ----------------------------

@pytest.mark.parametrize("kind,text,line", [
    pytest.param("uniqueness", "A user's email must be unique",
                 "db.add(booking)  # raise HTTPException 409 if the email is not unique", id="trailing-comment"),
    pytest.param("validation", "The booking count must not exceed the room capacity",
                 "# raise HTTPException when capacity exceeded\nreturn booking", id="leading-comment-line"),
    pytest.param("uniqueness", "A user's email must be unique", "db.add(booking)", id="control-no-comment"),
])
def test_a_comment_in_the_quote_supplies_no_enforcement(tmp_path, kind, text, line):
    """The tree match drops the judge's comment; its words must not pass the enforcement check."""
    verdict = _check(tmp_path, _svc(line), kind=kind, text=text)
    assert verdict["status"] == "unverified", verdict
    assert "does not demonstrate enforcement" in verdict["note"], verdict["note"]


def test_a_tree_matched_enforcing_statement_still_verifies(tmp_path):
    verdict = _check(tmp_path, _svc('raise HTTPException(status_code=409, detail="an account cannot report itself")'
                                    "  # refuses a self-report"),
                     kind="rule", text="An account must not report itself")
    assert verdict["status"] == "implemented", verdict.get("note")


# --- review of round 7: a pathological quote or file is no match, never a crash -------------------

@pytest.mark.parametrize("line", [
    pytest.param("x = " + "+".join(["a"] * 998), id="binop-chain-999"),
    pytest.param("x = a" + ".b" * 997, id="attribute-chain-999"),
    pytest.param("x = " + "-" * 999 + "1", id="unary-chain-999"),
    pytest.param("@a" + ".b" * 997, id="decorator-attribute-chain"),
])
def test_a_deeply_nested_quote_is_no_match(tmp_path, line):
    assert _check(tmp_path, _svc(line))["status"] == "unverified"


@pytest.mark.parametrize("terms", [500, 1000, 3000])
def test_a_deeply_nested_file_statement_is_no_match(tmp_path, terms):
    (tmp_path / "backend").mkdir(exist_ok=True)
    (tmp_path / "backend" / "deep.py").write_text(
        "def g(db):\n    y = " + " + ".join(f"'s{i}'" for i in range(terms)) + "\n    return y\n", encoding="utf-8")
    verdict = _check(tmp_path, ("backend/deep.py", "y = compute(db)"))
    assert verdict["status"] == "unverified"


# --- review of round 7: a line cited in the wrong file is named where it is -----------------------

# Run r10-p61, R1 "Dashboard displays a map": the excerpts of the two files its citations name, verbatim.
LUXMAP = """import { useEffect, useRef } from 'react';

export default function LuxMap({ geojson, plants, selection, onSelectRegion, onSelectPlant }) {
  const elRef = useRef(null);
  const mapRef = useRef(null);

  // create / destroy the map
  useEffect(() => {
    const map = L.map(elRef.current).setView([49.8153, 6.1296], 9);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 18,
    }).addTo(map);
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  return <div ref={elRef} className="map" data-testid="lux-map" />;
}
"""
DASHBOARD = """import { useEffect, useMemo, useState } from 'react';
import LuxMap from '../components/LuxMap.jsx';

export default function Dashboard() {
  return (
    <div>
      <section className="card">
        <h2>Treatment plants map</h2>
        <LuxMap
          geojson={geojson}
          plants={plants}
        />
      </section>
    </div>
  );
}
"""
R10_P61_R1 = [
    {"path": "frontend/src/components/LuxMap.jsx",
     "line": "const map = L.map(elRef.current).setView([49.8153, 6.1296], 9);"},
    {"path": "frontend/src/components/LuxMap.jsx",
     "line": "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {"},
    {"path": "frontend/src/pages/Dashboard.jsx", "line": "<LuxMap"},
    {"path": "frontend/src/pages/Dashboard.jsx",
     "line": "return <div ref={elRef} className=\"map\" data-testid=\"lux-map\" />;"},
]


def test_a_line_cited_in_the_wrong_file_names_the_file_it_is_in(tmp_path):
    for rel, content in {"frontend/src/components/LuxMap.jsx": LUXMAP,
                         "frontend/src/pages/Dashboard.jsx": DASHBOARD}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(content, encoding="utf-8")
    verdict = verify_evidence([{"id": 1, "kind": "ui", "text": "Dashboard displays a map", "status": "implemented",
                                "evidence": R10_P61_R1}], str(tmp_path))[0]
    note = verdict["note"]
    assert verdict["status"] == "unverified", "a misfiled citation is still rejected, never relocated"
    assert "citation 4" in note
    assert "frontend/src/components/LuxMap.jsx line 21" in note, note
    assert "must name the file the line is in" in note, note
    assert verdict[ledger.EVIDENCE_UNLOCATED] == ledger.CITATION_MALFORMED
