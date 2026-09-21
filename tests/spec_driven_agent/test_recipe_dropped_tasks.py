"""A checklist item the model dropped as not requested is visible afterwards.

The frontend shows no per-item state, so the recipe is where a reviewer sees
what the agent declined to build and the reason it gave.
"""

import json
import os

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
from besser.spec_driven_agent.llm_client import UsageTracker
from besser.spec_driven_agent.orchestrator import LLMOrchestrator


def test_dropped_items_land_in_the_recipe_with_their_reason(tmp_path):
    user = Class(name="User")
    user.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    model = DomainModel(name="M", types={user})

    class Client:
        model = "mock"
        usage = UsageTracker("mock")

        def chat(self, **kwargs):
            return {"stop_reason": "end_turn", "content": []}

    orch = LLMOrchestrator(llm_client=Client(), domain_model=model, output_dir=str(tmp_path))
    orch.executor.set_tasks(["Add authentication", "Implement User.deactivate"])
    orch.executor._task_list({"action": "drop", "id": 1, "reason": "the request never mentions auth"})
    orch.executor._task_list({"action": "done", "id": 2})

    orch._save_recipe("Build the user app", 1.0)

    recipe = json.load(open(os.path.join(str(tmp_path), ".besser_recipe.json"), encoding="utf-8"))
    assert recipe["dropped_tasks"] == [
        {"id": 1, "text": "Add authentication", "reason": "the request never mentions auth"},
    ]
