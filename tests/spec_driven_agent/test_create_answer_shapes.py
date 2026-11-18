"""The probes read the id of a created record from both shapes a generated backend's create answers.

Without the backend's ``uniform_responses`` option, ``POST /x/`` answers the bare object for a class without a
to-many end and ``{"x": object, "<end>_ids": [...]}`` for one with; with it, always the latter.
"""
import pytest

from besser.spec_driven_agent.validation.api_probe import confirmed_create_paths
from besser.spec_driven_agent.validation.constructibility import _extract_id

BARE = {"id": 7, "amount": 2.0}
WRAPPED = {"deal": {"id": 7, "amount": 2.0}}
WRAPPED_WITH_IDS = {"deal": {"id": 7, "amount": 2.0}, "items_ids": [1, 2]}


@pytest.mark.parametrize("body", [BARE, WRAPPED, WRAPPED_WITH_IDS])
def test_extract_id_reads_either_shape(body):
    assert _extract_id(body) == 7


@pytest.mark.parametrize("created", [BARE, WRAPPED, WRAPPED_WITH_IDS])
@pytest.mark.parametrize("read", [WRAPPED, WRAPPED_WITH_IDS])
def test_a_create_read_back_through_get_is_confirmed_in_either_shape(created, read):
    report = {"boot": "ok", "responses": [
        {"method": "POST", "path": "/deal/", "status": 200, "json": created, "failures": []},
        {"method": "GET", "path": "/deal/7/", "status": 200, "json": read, "failures": []},
    ]}
    assert confirmed_create_paths(report) == {"/deal"}
