"""Who may stop a BESSER Lab run through ``POST /spec-driven/cancel/{run_id}``: its starter, the change's owner or
an admin; another allowed login is refused (403) and the run goes on. A run id that is not a lab run behaves as
without the lab."""

import asyncio
import uuid

import pytest

from besser.utilities.web_modeling_editor.backend.backend import app
from tests.spec_driven_agent.lab.conftest import lab_layout, lab_layout_template  # noqa: F401
from tests.utilities.web_modeling_editor.backend.lab.test_lab_service import ADMIN, lab_env, make_service  # noqa: F401

OWNER, STARTER, OTHER = "jordi", "ana", "ben"


@pytest.fixture
def lab_run(lab_env, route_settings, app_state, spec_driven_runtime):  # noqa: F811
    """A run of ``STARTER`` on a change of ``OWNER``, reserved in the runtime like a streaming run."""
    route_settings.lab_enabled = True
    route_settings.lab_allowed_logins = ",".join((OWNER, STARTER, OTHER, ADMIN))
    route_settings.lab_admin_logins = ADMIN
    for login in (OWNER, STARTER, OTHER, ADMIN):
        app_state.github_sessions.user_tokens.set(f"sess-{login}", {"access_token": f"gho_{login}",
                                                                    "username": login})
    service = make_service(lab_env.workspace(), route_settings)
    change = service.workspace.create_draft(owner=OWNER)
    run_id = uuid.uuid4().hex
    service.begin_run(change.id, run_id, STARTER)
    event = asyncio.run(spec_driven_runtime.active_runs.reserve(run_id))
    app.state.lab = service
    yield run_id, event
    del app.state.lab


def cancel(client, run_id: str, login: str):
    return client.post(f"/besser_api/spec-driven/cancel/{run_id}", headers={"X-GitHub-Session": f"sess-{login}"})


def test_another_logins_lab_run_cannot_be_stopped(client, lab_run):
    run_id, event = lab_run

    response = cancel(client, run_id, OTHER)

    assert response.status_code == 403
    assert "started" in response.json()["detail"] and "admin" in response.json()["detail"]
    assert not event.is_set()


@pytest.mark.parametrize("login", [STARTER, OWNER, ADMIN])
def test_the_starter_the_owner_and_an_admin_may_stop_it(client, lab_run, login):
    run_id, event = lab_run

    response = cancel(client, run_id, login)

    assert response.status_code == 200 and response.json() == {"status": "cancelled", "runId": run_id}
    assert event.is_set()


def test_a_run_id_that_is_not_a_lab_run_is_answered_as_before(client, lab_run):
    response = cancel(client, uuid.uuid4().hex, OTHER)

    assert response.status_code == 200 and response.json()["status"] == "not_found"


def test_without_the_lab_the_cancel_route_is_unchanged(client, route_settings, spec_driven_runtime):
    route_settings.lab_enabled = False
    run_id = uuid.uuid4().hex
    event = asyncio.run(spec_driven_runtime.active_runs.reserve(run_id))

    response = client.post(f"/besser_api/spec-driven/cancel/{run_id}")

    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    assert event.is_set()
