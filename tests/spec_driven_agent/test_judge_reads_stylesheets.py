"""The requirements judge is shown the stylesheets a requirement about appearance rests on.

A requirement such as "each card has its own accent colour and a hover effect" is
implemented in a stylesheet. Stylesheets were not judged source: the judge never saw
them, a citation of a CSS rule could not be checked, and an inspection path such as
``src/**/*.css`` resolved to nothing, so working styling stayed an "unverified" blocker
("the stylesheet ... was not supplied") on every pass.
"""
import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

PAGE = "frontend/src/pages/Catalog.tsx"
STYLES = "frontend/src/index.css"
CARDS = "frontend/src/styles/cards.css"
HOVER_RULE = ".book-card:hover { transform: translateY(-2px); }"
STYLESHEET = (
    "/* Catalog cards */\n"
    "#root {\n"
    "  min-height: 100vh;\n"
    "}\n"
    ".book-card { cursor: pointer; }\n"
    f"{HOVER_RULE}\n"
    "/* .book-card--old { cursor: wait; } */\n"
)
HOVER = {"id": 1, "text": "Each book card shows a pointer cursor and a hover effect", "kind": "ui"}
# Names nothing about looks: the whole-app digest leaves the stylesheets out.
DISTINCT = {"id": 1, "text": "Each book genre is easy to tell apart on the catalog", "kind": "ui"}


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def app(tmp_path):
    _write(tmp_path, "backend/pydantic_classes.py", "class Book:\n    title: str\n")
    _write(tmp_path, PAGE, "import '../index.css';\n\nexport default function Catalog({ books }) {\n"
                           "  return books.map((b) => <a className={`book-card book-card--${b.genre}`}>{b.title}</a>);\n}\n")
    _write(tmp_path, STYLES, STYLESHEET)
    _write(tmp_path, CARDS, ".book-card--novel { border-left: 4px solid #2563eb; }\n")
    return tmp_path


def _verify(app, line, path=STYLES, requirement=HOVER):
    verdict = dict(requirement, status="implemented", evidence=[{"path": path, "line": line}], note="")
    return ledger.verify_evidence([verdict], str(app))[0]


def test_a_stylesheet_rule_is_a_citation(app):
    assert _verify(app, HOVER_RULE)["status"] == "implemented"


def test_a_commented_out_rule_is_not_a_citation(app):
    checked = _verify(app, "/* .book-card--old { cursor: wait; } */")
    assert checked["status"] == "unverified" and "comment" in checked["note"]


@pytest.mark.parametrize("text", [
    "Background jobs send the invoices nightly", "A beer has a style",
    "A project has a theme", "The API is responsive", "Payments are aligned with invoices",
])
def test_a_requirement_that_only_shares_a_word_with_styling_does_not_add_the_stylesheets(text):
    assert not ledger.shows_stylesheets([{"text": text}])


@pytest.mark.parametrize("text", [
    "Each card has its own accent colour", "Buttons show a pointer cursor on hover",
    "Titles use a larger font", "The table rows are styled in zebra stripes", "Cards have a drop shadow",
])
def test_a_styling_requirement_adds_the_stylesheets(text):
    assert ledger.shows_stylesheets([{"text": text}])


def test_a_glob_does_not_crowd_out_an_exact_path(app):
    for index in range(12):
        _write(app, f"frontend/src/pages/Page{index:02}.tsx", "export default () => null;\n")
    files = ledger._source_files(str(app))
    focus = ledger.focus_paths_in_turn([["frontend/src/**/*.tsx"], ["backend/pydantic_classes.py"]])
    resolved = ledger.resolved_focus_paths(files, focus)
    assert "backend/pydantic_classes.py" in resolved and len(resolved) == 10


def test_an_appearance_requirement_digest_shows_the_stylesheets(app):
    assert ledger.shows_stylesheets([HOVER])
    digest, gaps = ledger.build_app_digest_and_gaps(str(app), stylesheets=True)
    assert f"### {STYLES}" in digest and HOVER_RULE in digest
    # In a stylesheet a line starting with "#" is an id selector, not a comment.
    assert "#root {" in digest
    assert not gaps


def test_other_digests_list_the_stylesheets_as_omitted(app):
    assert not ledger.shows_stylesheets([DISTINCT])
    digest, gaps = ledger.build_app_digest_and_gaps(str(app))
    assert f"### {STYLES}" not in digest and f"### {PAGE}" in digest
    assert STYLES in digest.split("Omitted files:", 1)[1]
    assert {STYLES, CARDS} <= gaps


def test_a_glob_names_every_stylesheet_it_matches(app):
    files = ledger._source_files(str(app))
    assert ledger.resolved_focus_paths(files, ["frontend/src/**/*.css"]) == [STYLES, CARDS]
    assert ledger.resolved_focus_paths(files, ["src/**/*.css"]) == [STYLES, CARDS]
    gaps = ledger.build_app_digest_and_gaps(str(app))[1]
    assert ledger.named_digest_gaps({"inspection_paths": ["frontend/src/**/*.css"], "note": ""}, gaps) == [
        STYLES, CARDS]


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - judge_coverage is replaced
        raise AssertionError("no LLM call expected")


def _orchestrator(app, requirement):
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Book")}),
                           output_dir=str(app), enable_tracing=False, enable_checkpointing=False)
    orch._instructions = requirement["text"]
    orch._generator_used = "generate_web_app"
    orch._requirements = [dict(requirement)]
    return orch


def _judge(monkeypatch, requirement, *answers):
    digests, script = [], list(answers)

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        return [dict(requirement, **script.pop(0))]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    return digests


def test_a_styling_requirement_is_judged_with_its_stylesheet(app, monkeypatch):
    digests = _judge(monkeypatch, HOVER, {"status": "implemented", "note": "",
                                          "evidence": [{"path": STYLES, "line": HOVER_RULE}]})
    issues = _orchestrator(app, HOVER)._collect_requirement_issues()
    assert HOVER_RULE in digests[0]
    assert not [issue for issue in issues if issue.startswith("requirement")], issues


def test_a_verdict_naming_a_stylesheet_glob_is_judged_again_with_them(app, monkeypatch):
    orch = _orchestrator(app, DISTINCT)
    digests = _judge(monkeypatch, DISTINCT,
                     {"status": "unverified", "evidence": "", "inspection_paths": ["frontend/src/**/*.css"],
                      "note": "Catalog.tsx sets a class per genre; the stylesheet was not supplied."},
                     {"status": "implemented", "note": "",
                      "evidence": [{"path": CARDS, "line": ".book-card--novel { border-left: 4px solid #2563eb; }"}]})
    issues = orch._collect_requirement_issues()
    assert len(digests) == 2 and f"### {CARDS}" not in digests[0] and f"### {CARDS}" in digests[1]
    assert orch._requirement_verdicts[0]["status"] == "implemented", issues


def test_a_stylesheet_edit_does_not_re_judge_a_page_citation(app, monkeypatch):
    """The page imports the global stylesheet; its verified verdict rests on the page, not the CSS."""
    line = "return books.map((b) => <a className={`book-card book-card--${b.genre}`}>{b.title}</a>);"
    digests = _judge(monkeypatch, DISTINCT, {"status": "implemented", "note": "",
                                             "evidence": [{"path": PAGE, "line": line}]})
    orch = _orchestrator(app, DISTINCT)
    orch._collect_requirement_issues()
    _write(app, STYLES, STYLESHEET + ".book-card { color: red; }\n")
    orch._collect_requirement_issues()
    assert len(digests) == 1
