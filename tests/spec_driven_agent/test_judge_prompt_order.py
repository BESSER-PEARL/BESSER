"""The requirements judge's prompt goes from most stable to least stable.

A run judges coverage again after each repair. The prompt put the requirement
subset and the previous round's feedback (both change between calls) before
the code digest (about 47k tokens, mostly unchanged), and ordered the digest
by how much of each file the agent wrote, which an edit changes. So no judge
call could reuse the cached prompt prefix of the one before. Now: the request,
the code in path order, then the feedback and the requirements.
"""

import pytest

import besser.spec_driven_agent.planning.requirements_ledger as ledger
from besser.spec_driven_agent.state.generator_output import store_generator_output

REQUIREMENTS = [{"id": 1, "text": "Orders list their customer", "kind": "behavior"},
                {"id": 2, "text": "A customer has an email", "kind": "data"}]


class _Client:
    model = "gpt-6-sol"

    def __init__(self):
        self.prompts = []


@pytest.fixture
def sent(monkeypatch):
    client = _Client()
    monkeypatch.setattr(ledger, "_is_real_provider", lambda c: True)

    def call(llm_client, system, prompt, tool, **kwargs):
        llm_client.prompts.append(prompt)
        return {"verdicts": [{"id": 1, "status": "unverified", "evidence": "", "note": ""}]}
    monkeypatch.setattr(ledger, "_call_with_tool", call)
    return client


def test_the_requirements_and_feedback_follow_the_code(sent):
    ledger.judge_coverage(REQUIREMENTS, "### backend/orders.py\ndef list_orders(): ...", sent,
                          original_spec="Track orders.",
                          previous_verdicts=[{"id": 1, "evidence": "backend/orders.py: x", "note": "not found"}])
    [prompt] = sent.prompts
    order = [prompt.index(heading) for heading in (
        "## Original user specification", "## Generated code", "## Previous citation verification",
        "## Requirements")]
    assert order == sorted(order), prompt


def _write(root, rel, text):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _headers(root) -> list[str]:
    return [line[4:] for line in ledger.build_app_digest(str(root)).splitlines() if line.startswith("### ")]


def test_an_edit_to_one_file_does_not_move_the_others_in_the_digest(tmp_path):
    """Ordered by written share, the file an edit rewrote jumped ahead of every other router."""
    _write(tmp_path, "backend/sql_alchemy.py", "class Order(Base):\n    id = Column(Integer)\n")
    generated = {}
    for name in ("customer", "invoice", "order"):
        body = "".join(f"@router.get('/{name}/{n}/')\ndef read_{name}_{n}(): return []\n" for n in range(20))
        generated[f"backend/routers/{name}.py"] = body.encode()
        _write(tmp_path, f"backend/routers/{name}.py", body)
    store_generator_output(str(tmp_path), "backend", generated, "8.0.0")
    before = _headers(tmp_path)

    _write(tmp_path, "backend/routers/order.py", "def order_totals():\n    return compute()\n" * 20)

    assert _headers(tmp_path) == before == [
        "backend/sql_alchemy.py", "backend/routers/customer.py", "backend/routers/invoice.py",
        "backend/routers/order.py"]
