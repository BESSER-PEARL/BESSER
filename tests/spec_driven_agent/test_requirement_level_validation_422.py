"""A call that FastAPI refused before the handler ran confirms nothing the handler does.

A passing workflow's negative check posted an invalid body and expected FastAPI's
request-validation 422. That call counted as exercising the route, so every
requirement cited in the route's handler could read "confirmed by running the app's
API" although the handler never ran. A request-validation 422 (a ``detail`` list of
``loc``/``msg`` errors) no longer counts; a 4xx the handler raised still does.
"""

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.validation.api_probe import request_validation_refusal
from tests.spec_driven_agent.test_requirement_verification_level import _Client, _recipe, _verdict

ROUTER = "backend/routers/order.py"
SOURCE = '''from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/order")


@router.post("/")
def create_order(quantity: int):
    if quantity > 100:
        raise HTTPException(status_code=409, detail="at most 100 items")
    return {"id": 1, "quantity": quantity}
'''
RULE = "if quantity > 100:"
VALIDATION_422 = {"detail": [{"type": "int_parsing", "loc": ["body", "quantity"], "msg": "Input should be a valid "
                                                                                         "integer", "input": "x"}]}


def _level(tmp_path, *responses) -> str:
    (tmp_path / ROUTER).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ROUTER).write_text(SOURCE, encoding="utf-8")
    orch = LLMOrchestrator(llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="Order")}),
                           output_dir=str(tmp_path))
    orch._requirement_verdicts = [_verdict(1, "rule", ROUTER, line=RULE)]
    orch._api_scenarios["named:orders"] = {
        "scenario": {"requests": [{"method": "POST", "path": "/order/"}]}, "scenario_id": "orders",
        "revision": orch._workspace_revision(),
        "report": {"status": "passed", "responses": [
            {"index": i, "method": "POST", "path": "/order/", **response} for i, response in enumerate(responses)]}}
    return _recipe(orch)["requirements"][0]["verification"]


def test_a_route_reached_only_by_a_request_validation_422_is_not_api_verified(tmp_path):
    assert _level(tmp_path, {"status": 422, "json": VALIDATION_422}) == "unverified"


def test_a_refusal_the_handler_raised_still_counts(tmp_path):
    assert _level(tmp_path, {"status": 409, "json": {"detail": "at most 100 items"}}) == "verified"


def test_a_422_the_handler_raised_from_its_own_validation_errors_still_counts(tmp_path):
    # ``raise HTTPException(422, detail=exc.errors())``: loc/msg errors whose loc names no request part.
    handler_422 = {"detail": [{"type": "greater_than", "loc": ["quantity"], "msg": "Input should be greater "
                                                                                  "than 0", "input": -1}]}
    assert not request_validation_refusal({"status": 422, "json": handler_422})
    assert _level(tmp_path, {"status": 422, "json": handler_422}) == "verified"


def test_a_validation_body_cut_at_the_report_limit_is_still_recognised():
    assert request_validation_refusal({"status": 422, "json": '{"detail": [{"type": "missing", "loc": ["bo'
                                                                      ' [truncated]'})
    assert not request_validation_refusal({"status": 422, "json": {"detail": "quantity must be positive"}})
    assert not request_validation_refusal({"status": 200, "json": VALIDATION_422})
