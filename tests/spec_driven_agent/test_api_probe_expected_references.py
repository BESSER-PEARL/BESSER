"""``expected_fields`` values may name an earlier response, like request bodies do.

A workflow check often has to say "the order lists the customer created two
requests ago". The ids are minted by the app, so the expected value can only be
written as a reference.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from besser.spec_driven_agent.validation import api_probe

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _shop_app():
    from fastapi import FastAPI

    app = FastAPI()
    customers: list[dict] = []

    @app.post("/customer/")
    def create_customer(body: dict):
        customer = {"id": 100 + len(customers), "name": body["name"], "api_token": "tok-5f4e3d2c1b0a9f8e"}
        customers.append(customer)
        return {"customer": customer, "session": {"access_token": "sess-9d8c7b6a5f4e3d2c", "expires_in": 3600}}

    @app.get("/template/")
    def template():
        return {"body": "Dear {{9.name}}, your order {{0}} has shipped"}

    @app.post("/order/")
    def create_order(body: dict):
        return {"order": {"id": 1, "label": f"Order for {body['customer']}"},
                "customer_ids": [body["customer"]], "owner": {"id": body["customer"]}}

    return app


def _run(requests):
    return asyncio.run(api_probe._run_requests(_shop_app(), requests))


_CUSTOMER = {"method": "POST", "path": "/customer/", "json": {"name": "Ada"}}


def _order(expected_fields):
    return {"method": "POST", "path": "/order/", "json": {"customer": "{{0.customer.id}}"},
            "expected_fields": expected_fields}


@pytest.mark.parametrize("expected_fields", [
    {"customer_ids.0": "{{0.customer.id}}"},
    {"customer_ids": ["{{0.customer.id}}"]},
    {"owner": {"id": "{{0.customer.id}}"}},
    {"order.label": "Order for {{0.customer.id}}"},
])
def test_an_expected_value_can_reference_an_earlier_response(expected_fields):
    report = _run([_CUSTOMER, _order(expected_fields)])
    assert report["status"] == "passed", report["assertion_failures"]


def test_a_referenced_expectation_still_fails_on_a_different_value():
    report = _run([_CUSTOMER, _CUSTOMER, _order({"customer_ids": ["{{1.customer.id}}"]})])
    assert report["status"] == "failed"
    assert report["responses"][2]["failures"] == ["customer_ids: expected [101], got [100]"]


def test_an_unresolvable_expected_reference_fails_that_assertion_only():
    report = _run([_CUSTOMER, _order({"customer_ids.0": "{{0.customer.missing}}", "order.id": 1})])
    assert report["responses"][1]["failures"] == ["customer_ids.0: expected '{{0.customer.missing}}', got 100"]


def test_a_credential_referenced_as_an_expected_value_is_not_quoted():
    report = _run([_CUSTOMER, _order({"order.label": "{{0.customer.api_token}}"})])
    assert report["status"] == "failed"
    assert "tok-5f4e3d2c1b0a9f8e" not in json.dumps(report)


@pytest.mark.parametrize("expected", ["{{0.session}}", "{{0}}", {"id": "{{0.session}}"}, ["{{0.session}}"],
                                      {"session": "{{0.session}}"}])
def test_a_referenced_object_is_quoted_without_its_credentials(expected):
    """The expected value is quoted in the failure: a whole response or an
    object in it carries tokens that a reference to one field would not."""
    report = _run([_CUSTOMER, _order({"owner": expected})])
    assert report["status"] == "failed"
    quoted = json.dumps(report)
    assert "sess-9d8c7b6a5f4e3d2c" not in quoted and "tok-5f4e3d2c1b0a9f8e" not in quoted
    assert "expires_in" in report["responses"][1]["failures"][0]


def test_text_that_only_looks_like_a_reference_is_compared_as_written():
    """A stored message template holds braces of its own."""
    template = {"method": "GET", "path": "/template/"}
    written = "Dear {{9.name}}, your order {{0}} has shipped"
    assert _run([_CUSTOMER, dict(template, expected_fields={"body": written})])["status"] == "passed"


@pytest.mark.parametrize("written", [
    "Dear {{9.name}}",                     # names no earlier response
    "Dear {{0.customer}}, shipped",        # an object inside text
    "{{1.order.id}}",                      # the response it checks
])
def test_text_with_braces_that_does_not_match_fails_as_a_mismatch(written):
    """Braces that resolve to nothing are text: a wrong expectation is reported
    as the mismatch it is, not as a reference error."""
    report = _run([_CUSTOMER, {"method": "GET", "path": "/template/", "expected_fields": {"body": written}}])
    assert report["responses"][1]["failures"] == [
        f"body: expected {written!r}, got 'Dear {{{{9.name}}}}, your order {{{{0}}}} has shipped'"]


def test_a_rejected_request_names_its_unsupported_keys():
    with pytest.raises(ValueError) as error:
        api_probe._validate_requests([_CUSTOMER, {"method": "GET", "path": "/order/", "expect_status": 200}])
    assert "request 1: unsupported key(s) 'expect_status'" in str(error.value)
    assert "expected_status" in str(error.value)


def test_too_many_requests_says_how_many_and_what_to_do():
    with pytest.raises(ValueError) as error:
        api_probe._validate_requests([_CUSTOMER] * 22)
    assert "1 to 20 request objects, got 22" in str(error.value)
    assert "scenario_id" in str(error.value)
