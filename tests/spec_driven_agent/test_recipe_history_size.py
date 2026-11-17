"""Each run rewrites the recipe's history, so what it carries over is bounded like a new entry.

A modify run starts from a base whose recipe can come from any repository.
"""
import json

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

WRITE_BUDGET = 128 * 1024


class _Client:
    model = "mock-model"
    usage = UsageTracker("mock-model")

    def chat(self, **kwargs):
        raise AssertionError("no LLM call expected")


def _orchestrator(tmp_path) -> LLMOrchestrator:
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    return LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="Shop", types={order}),
                           output_dir=str(tmp_path), enable_checkpointing=False)


def test_a_modify_run_on_a_base_with_a_long_history_stays_under_the_budget(tmp_path):
    base_history = [{
        "instructions": f"Change {n}: " + "add invoices " * 5000,
        "saved_at": "2026-01-01T00:00:00+00:00",
        "mode": "modify",
        "files_touched": [f"backend/invoice_{n}_{i}.py" for i in range(400)],
        "notes": "x" * 20_000,
    } for n in range(40)]
    (tmp_path / ".besser_recipe.json").write_text(json.dumps({"history": base_history}), encoding="utf-8")
    orchestrator = _orchestrator(tmp_path)
    orchestrator._modify_mode = True

    orchestrator._save_recipe("Add a due date to Order", 3.0)

    text = (tmp_path / ".besser_recipe.json").read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) <= WRITE_BUDGET
    history = json.loads(text)["history"]
    assert len(history) == 10
    assert history[-1]["instructions"] == "Add a due date to Order" and history[-1]["mode"] == "modify"
    for entry in history[:-1]:
        assert set(entry) == {"instructions", "saved_at", "mode", "files_touched"}
        assert len(entry["instructions"]) == 300 and len(entry["files_touched"]) == 20
    assert history[0]["instructions"].startswith("Change 31: ")


def test_a_recipe_within_the_budget_is_written_as_before(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.tool_calls_log.append(
        {"turn": 1, "tool": "write_file", "input": {"path": "app.py", "content": "print(1)\n" * 40},
         "success": True, "status": "ok"})

    orchestrator._save_recipe("Track orders", 1.0)

    text = (tmp_path / ".besser_recipe.json").read_text(encoding="utf-8")
    recipe = json.loads(text)
    assert "recipe_shortened" not in recipe
    assert text == json.dumps(recipe, indent=2)
    assert recipe["tool_calls"][0]["input"]["content"] == "print(1)\n" * 40
    assert orchestrator.saved_recipe == recipe
