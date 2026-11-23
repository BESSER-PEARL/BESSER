"""A second focused re-judge still shows the files the first one relied on.

Each requirement's focus held only what its own first verdict named. When the first
focused round showed a page because ANOTHER requirement named it, and the answer for
this requirement relied on that page while naming a file the round left out, the
second round showed only the left-out file: the judge then answered "the page is not
included in the supplied excerpts" about a page it had just cited, and the verdict
stayed an "unverified" blocker.
"""
from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.providers.llm_client import UsageTracker

DASHBOARD = "frontend/src/pages/Dashboard.tsx"
ENTRY = "frontend/src/main.tsx"
REQS = [{"id": 1, "text": "The dashboard lists open orders", "kind": "ui"},
        {"id": 2, "text": "Each order tile links to its order", "kind": "ui"}]


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover - judge_coverage is replaced
        raise AssertionError("no LLM call expected")


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_the_second_round_shows_the_page_the_answer_relied_on(tmp_path, monkeypatch):
    _write(tmp_path, "backend/pydantic_classes.py", "class Order:\n    status: str\n")
    _write(tmp_path, DASHBOARD, "export default function Dashboard({ orders }) {\n"
                                "  return orders.map((o) => <a href={`/orders/${o.id}`}>{o.status}</a>);\n}\n")
    _write(tmp_path, ENTRY, "import Dashboard from './pages/Dashboard';\nrender(<Dashboard />);\n")
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={Class(name="Order")}),
                           output_dir=str(tmp_path), enable_tracing=False, enable_checkpointing=False)
    orch._instructions = "Show open orders on a dashboard."
    orch._generator_used = "generate_web_app"
    orch._requirements = [dict(r) for r in REQS]
    answers = [
        # The whole-app judgment: R1 names the page, R2 names nothing that resolves.
        [{"status": "unverified", "evidence": "", "inspection_paths": [DASHBOARD], "note": "see the page"},
         {"status": "unverified", "evidence": "", "inspection_paths": ["the order tiles"], "note": "not shown"}],
        # Round 1 shows the page for R1; R2's answer relies on it and names the entry file it left out.
        [{"status": "unverified", "evidence": "", "note": "still unclear"},
         {"status": "unverified", "evidence": "", "inspection_paths": [ENTRY],
          "note": "Dashboard.tsx renders a link per order; main.tsx must mount it"}],
        [{"status": "unverified", "evidence": "", "note": "unclear"}],
    ]
    digests = []

    def judge(requirements, digest, client, **kwargs):
        digests.append(digest)
        by_id = {r["id"]: r for r in requirements}
        return [dict(by_id[r["id"]], **answer) for r, answer in zip(requirements, answers.pop(0))]

    monkeypatch.setattr(ledger, "judge_coverage", judge)
    orch._collect_requirement_issues()
    assert len(digests) == 3
    assert f"### {DASHBOARD}" in digests[1]
    assert f"### {ENTRY}" in digests[2] and f"### {DASHBOARD}" in digests[2]
