"""A modelled action with no parameters must be callable with no arguments.

The library spec says a loan "can be renewed, which extends the due date"
without saying by how much, and the model declares ``Loan.renew()`` with an
empty parameter list. Faced with the gap the agent asked the caller instead of
choosing: 20 of the 22 gpt-5.6-terra library runs in
``verification/spec-iterations`` generated a ``renew`` handler that answers
422 "A later dueDate is required" / "renew requires params.dueDate". The
generated React MethodButton POSTs ``{}`` for a zero-parameter method
(MethodButton.tsx.j2), so the delivered Renew button was uncallable.

It is not obstinacy: the model's entire statement of the signature was
``{"name": "renew", "implementation": "none"}`` - ``_method_entry`` omitted
``parameters`` when the list was empty, so "takes nothing" was expressed only
by the absence of a key. Three layers, cheapest first:

1. the serializer always states the parameter list (``"parameters": []``),
2. the generated stub's docstring says it in the file being edited,
3. ``contract_checks`` fails a handler that demands a value anyway - the only
   one of the three that is enforcement rather than advice.

Measured over 257 run workspaces spanning 16 cases: 1093 handlers for 48
distinct zero-parameter actions, of which 22 read a request-body key - every
one of them ``Loan.renew``. The check fires on 21 (the 22nd defaults the
value, which is the wanted behaviour) and on none of the other 1072,
hotel's 529 zero-argument handlers included, so a blocker is affordable.
"""

import json

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    Method,
    Parameter,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.contract_checks import build_data_contract, lint_file
from besser.generators.llm.model_serializer import serialize_domain_model
from besser.generators.llm.orchestrator import _classify_issue

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
DateType = PrimitiveDataType("date")


def _loan_model() -> DomainModel:
    """Loan.renew() takes nothing; Loan.reschedule(dueDate) takes one."""
    loan = Class(name="Loan")
    loan.attributes = {Property(name="id", type=IntegerType, is_id=True)}
    loan.methods = {
        Method(name="renew", type=PrimitiveDataType("bool")),
        Method(name="reschedule", type=PrimitiveDataType("bool"), parameters=[
            Parameter(name="dueDate", type=DateType),
        ]),
    }
    return DomainModel(name="Library", types={loan})


# --------------------------------------------------------------------------- #
# 1. The serialized model states the parameter list instead of omitting it
# --------------------------------------------------------------------------- #

def test_a_zero_parameter_method_serializes_an_explicit_empty_list():
    methods = {
        m["name"]: m
        for cls in serialize_domain_model(_loan_model())["classes"]
        for m in cls.get("methods", [])
    }
    # The whole point: not merely a missing key the model must guess about.
    assert methods["renew"]["parameters"] == []
    assert "parameters" in json.dumps(methods["renew"])
    assert methods["reschedule"]["parameters"] == [
        {"name": "dueDate", "type": "date"}
    ]


# --------------------------------------------------------------------------- #
# 2. The contract is extracted from the model
# --------------------------------------------------------------------------- #

def test_the_data_contract_records_each_action_s_declared_arity():
    contract = build_data_contract(_loan_model())
    assert contract.action_arity == {"Loan": {"renew": 0, "reschedule": 1}}


# --------------------------------------------------------------------------- #
# 3. The lint fails a handler that demands a value the model never declared
# --------------------------------------------------------------------------- #

_HEADER = (
    "from fastapi import APIRouter, Body, Depends, HTTPException\n"
    "router = APIRouter()\n\n"
)


def _handler(body: str, route: str = "/loan/{loan_id}/methods/renew/") -> str:
    return (
        _HEADER
        + f'@router.post("{route}", response_model=None)\n'
        "async def execute_loan_renew(\n"
        "    loan_id: int,\n"
        "    params: dict = Body(default=None, embed=True),\n"
        "    database: Session = Depends(get_db),\n"
        "):\n" + body
    )


def _messages(content: str, model: DomainModel | None = None) -> list:
    contract = build_data_contract(model or _loan_model())
    return [f.message for f in lint_file("routers/loan_methods.py", content, contract)]


# Verbatim from verification/spec-iterations/gpt-5.6-terra-6oj40oh_.
_GUARD_THEN_422 = (
    '    if not params or not params.get("dueDate"):\n'
    '        raise HTTPException(status_code=422, detail="A later dueDate is required to renew a loan")\n'
    '    _loan.dueDate = date.fromisoformat(params["dueDate"])\n'
    "    return {\"success\": True}\n"
)

# Verbatim shape from gpt-5.6-terra-cjg6cn0s: read through `(params or {})`,
# then a 400 on the missing key.
_INDIRECT_READ = (
    '    new_due_date = (params or {}).get("dueDate")\n'
    "    if not new_due_date:\n"
    '        raise HTTPException(status_code=400, detail="A new dueDate is required for renewal")\n'
    "    return {\"success\": True}\n"
)

# gpt-5.6-terra-s3gwc522: a required Pydantic body model in the signature.
_REQUIRED_BODY_MODEL = (
    _HEADER
    + '@router.post("/loan/{loan_id}/methods/renew/", response_model=None)\n'
    "async def execute_loan_renew(\n"
    "    loan_id: int,\n"
    "    renewal: RenewLoanRequest,\n"
    "    database: Session = Depends(get_db),\n"
    ") -> dict:\n"
    "    loan.dueDate = renewal.newDueDate\n"
    "    return {\"success\": True}\n"
)


def test_a_guard_that_422s_on_a_missing_body_key_is_a_blocker():
    contract = build_data_contract(_loan_model())
    findings = lint_file("routers/loan_methods.py", _handler(_GUARD_THEN_422), contract)
    assert len(findings) == 1
    assert findings[0].blocker
    assert "Loan.renew" in findings[0].message
    assert "dueDate" in findings[0].message
    # It must reach the Phase 3 fix loop as a blocker, not a warning.
    assert _classify_issue(f"data contract: {findings[0].message}").severity == "blocker"


def test_a_key_read_through_an_or_default_is_still_a_demand():
    assert len(_messages(_handler(_INDIRECT_READ))) == 1


def test_a_required_request_model_in_the_signature_is_a_blocker():
    messages = _messages(_REQUIRED_BODY_MODEL)
    assert len(messages) == 1
    assert "renewal" in messages[0]


def test_an_alias_fallback_chain_is_a_demand():
    """gpt-5.6-terra-l49zhafg: `.get("dueDate") or .get("due_date")`."""
    messages = _messages(_handler(
        "    payload = params or {}\n"
        '    value = payload.get("dueDate") or payload.get("due_date")\n'
        "    if not value:\n"
        '        raise HTTPException(status_code=400, detail="Renewal requires a later dueDate")\n'
        "    return {\"success\": True}\n"
    ))
    assert len(messages) == 1
    assert "dueDate" in messages[0] and "due_date" in messages[0]


def test_a_conditional_read_falling_back_to_none_is_a_demand():
    """gpt-5.6-terra-v47n8me4: `params.get("dueDate") if params else None`."""
    assert len(_messages(_handler(
        '    raw = params.get("dueDate") if params else None\n'
        "    if raw is None:\n"
        '        raise HTTPException(status_code=422, detail="dueDate is required")\n'
        "    return {\"success\": True}\n"
    ))) == 1


def test_a_conditional_read_falling_back_to_a_value_is_accepted():
    assert _messages(_handler(
        '    days = params.get("extensionDays") if params else 14\n'
        "    if days <= 0:\n"
        '        raise HTTPException(status_code=400, detail="extensionDays must be positive")\n'
        "    return {\"success\": True}\n"
    )) == []


def test_a_refusal_dressed_as_http_200_is_still_a_demand():
    """gpt-5.6-terra-053ydac9, the run made to confirm this very fix.

    It answers the empty body with 200 and {"success": false, "message": "An
    extended dueDate is required"}, leaving dueDate untouched. The button is
    as dead as on a 422, and both the check and the case oracle scored it a
    pass while it did nothing.
    """
    messages = _messages(_handler(
        '    payload = params or {}\n'
        '    raw_due_date = payload.get("dueDate") or payload.get("due_date")\n'
        "    if not raw_due_date:\n"
        '        return {"success": False, "message": "An extended dueDate is required"}\n'
        "    return {\"success\": True}\n"
    ))
    assert len(messages) == 1
    assert "dueDate" in messages[0]


def test_a_state_refusal_returning_success_false_is_accepted():
    """The modelled return type is bool; refusing on state is correct."""
    assert _messages(_handler(
        "    if _loan.status != LoanStatus.ACTIVE:\n"
        '        return {"success": False, "message": "Only an active loan can be renewed"}\n'
        "    _loan.dueDate = _loan.dueDate + timedelta(days=14)\n"
        "    return {\"success\": True}\n"
    )) == []


def test_a_subscript_read_of_the_body_is_a_demand():
    assert len(_messages(_handler(
        '    _loan.dueDate = params["dueDate"]\n    return {"success": True}\n'
    ))) == 1


# --------------------------------------------------------------------------- #
# 4. No false positives - a false blocker burns billable fix turns
# --------------------------------------------------------------------------- #

def test_an_implementation_that_chooses_a_default_is_accepted():
    """The wanted behaviour: pick a value, accept an optional override.

    Verbatim from gpt-5.6-terra-s7a5e4er, the one library run that got this
    right. The 400 guard is a validation of a supplied value, not a demand.
    """
    assert _messages(_handler(
        '    extension_days = (params or {}).get("extensionDays", 14)\n'
        "    if not isinstance(extension_days, int) or extension_days <= 0:\n"
        '        raise HTTPException(status_code=400, detail="extensionDays must be a positive integer")\n'
        "    _loan.dueDate = _loan.dueDate + timedelta(days=extension_days)\n"
        "    return {\"success\": True}\n"
    )) == []


def test_defaulting_on_the_miss_and_validating_the_hit_is_accepted():
    """The shape the fix itself takes, and a false positive it once produced.

    The `else` branch's 422 rejects a malformed supplied value; the branch
    taken when nothing was supplied picks a default. Reading a 4xx anywhere
    under the `if` condemned exactly the handler the check is asking for.
    """
    assert _messages(_handler(
        '    raw = (params or {}).get("dueDate")\n'
        "    if raw is None:\n"
        "        new_due_date = _loan.dueDate + timedelta(days=14)\n"
        "    else:\n"
        "        try:\n"
        "            new_due_date = date.fromisoformat(raw)\n"
        "        except (TypeError, ValueError):\n"
        '            raise HTTPException(status_code=422, detail="dueDate must be an ISO date")\n'
        "    _loan.dueDate = new_due_date\n"
        "    return {\"success\": True}\n"
    )) == []


def test_a_refusal_that_depends_only_on_state_is_accepted():
    assert _messages(_handler(
        "    if _loan.status != LoanStatus.ACTIVE:\n"
        '        raise HTTPException(status_code=400, detail="Only an active loan can be renewed")\n'
        "    _loan.dueDate = _loan.dueDate + timedelta(days=14)\n"
        "    return {\"success\": True}\n"
    )) == []


def test_the_generated_501_stub_is_not_flagged():
    """action_inventory owns unimplemented handlers; do not double-report."""
    assert _messages(_handler(
        "    raise HTTPException(\n"
        "        status_code=501,\n"
        '        detail="Method \'renew\' of Loan is modeled but has no implementation",\n'
        "    )\n"
    )) == []


def test_a_method_that_declares_a_parameter_may_require_it():
    assert _messages(_handler(
        '    if not params or not params.get("dueDate"):\n'
        '        raise HTTPException(status_code=422, detail="dueDate is required")\n'
        "    return {\"success\": True}\n",
        route="/loan/{loan_id}/methods/reschedule/",
    )) == []


def test_an_action_name_that_matches_no_modelled_method_is_left_alone():
    assert _messages(_handler(
        '    if not params.get("dueDate"):\n'
        '        raise HTTPException(status_code=422, detail="dueDate is required")\n'
        "    return {\"success\": True}\n",
        route="/loan/{loan_id}/methods/extend/",
    )) == []


def test_an_unparseable_file_reports_nothing():
    assert _messages("def broken(:\n    pass\n") == []


def test_a_model_with_no_methods_costs_nothing():
    book = Class(name="Book")
    book.attributes = {Property(name="isbn", type=StringType, is_id=True)}
    model = DomainModel(name="Library", types={book})
    assert build_data_contract(model).action_arity == {}
    assert _messages(_handler(_GUARD_THEN_422), model=model) == []
