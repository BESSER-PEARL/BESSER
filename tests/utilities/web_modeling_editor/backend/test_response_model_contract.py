"""The response models state the contract the backend actually sends.

The editor types every response from the OpenAPI schema, so a field typed
looser than what the code sends (``str`` for a fixed set of values, ``dict``
for a fixed shape) forces a cast on the editor side, and a field typed tighter
than what the code sends turns a working response into a 500. These tests pin
both directions against the code that fills each field.
"""

from __future__ import annotations

import asyncio
import inspect
import typing

import pytest
from pydantic import TypeAdapter, ValidationError

from besser.BUML.notations.kg_to_buml import consistency
from besser.utilities.web_modeling_editor.backend.models.responses import (
    DeploymentUrls,
    DeployToGitHubResponse,
    KGConsistencyIssueResponse,
    KGConsistencyReportResponse,
    KGSeverityCountsResponse,
)
from besser.utilities.web_modeling_editor.backend.models.spec_driven import (
    CodeAnalysisResponse,
    ImportGitHubRunResponse,
    PushSmartToGitHubResponse,
)
from besser.utilities.web_modeling_editor.backend.routers.github_deploy_api import (
    PendingChangeStatus,
    SaveProjectResponse,
    _pending_status,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import analyze_repository


def _literal_values(model, field: str) -> set:
    """The values of the ``Literal`` in ``model.field`` (``Optional`` unwrapped)."""
    def values(tp) -> set:
        if typing.get_origin(tp) is typing.Literal:
            return set(typing.get_args(tp))
        return set().union(*(values(arg) for arg in typing.get_args(tp)))
    return values(model.model_fields[field].annotation)


@pytest.mark.parametrize("model, field, values", [
    (PendingChangeStatus, "pull_request_state", {"open", "closed", "merged"}),
    (SaveProjectResponse, "status", {"pushed", "merged", "conflict"}),
    (PushSmartToGitHubResponse, "status", {"pushed", "merged", "conflict", "pull_request"}),
    (DeployToGitHubResponse, "deployment_type", {"webapp", "chatbot"}),
    (KGConsistencyIssueResponse, "severity", {"info", "warning", "violation"}),
    (KGConsistencyReportResponse, "inferenceUsed", {"rdfs", "owlrl"}),
])
def test_closed_value_sets(model, field, values):
    assert _literal_values(model, field) == values
    for value in values:
        TypeAdapter(model.model_fields[field].annotation).validate_python(value)
    with pytest.raises(ValidationError):
        TypeAdapter(model.model_fields[field].annotation).validate_python("bogus")


def test_kg_literals_cover_what_the_consistency_check_produces():
    severities = _literal_values(KGConsistencyIssueResponse, "severity")
    assert set(consistency._SEVERITY_MAP.values()) <= severities
    # Unmapped pyshacl severities fall back to "violation"; skipped patterns are "warning".
    assert {"violation", "warning"} <= severities
    assert set(consistency._count_severities([])) == set(KGSeverityCountsResponse.model_fields)
    # The route never passes ``inference``, so the report carries the default.
    default = inspect.signature(consistency.check_kg_consistency).parameters["inference"].default
    assert default in _literal_values(KGConsistencyReportResponse, "inferenceUsed")


def test_real_code_analysis_validates(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text(
        "from sqlalchemy import Column, Integer\n"
        "from sqlalchemy.orm import declarative_base\n"
        "Base = declarative_base()\n"
        "class Room(Base):\n    __tablename__ = 'room'\n    id = Column(Integer, primary_key=True)\n",
        encoding="utf-8",
    )
    analysis, _ = analyze_repository(str(tmp_path))
    payload = analysis.to_dict()

    parsed = CodeAnalysisResponse.model_validate(payload)

    assert parsed.model_dump() == payload
    assert set(CodeAnalysisResponse.model_fields) == set(payload)


def test_import_response_round_trips_analysis_and_diagram():
    body = {
        "run_id": "r", "has_model": False, "owner": "o", "repo": "x", "branch": "main",
        "analysis": {"stack": ["FastAPI"], "model_source": None, "source_files": [], "entities": [],
                     "ai": False, "skipped_relationships": []},
        "proposed_diagram": {"title": "x model", "model": {"type": "ClassDiagram", "elements": {}},
                             "diagramType": "ClassDiagram"},
    }

    dumped = ImportGitHubRunResponse.model_validate(body).model_dump()

    assert dumped["analysis"] == body["analysis"]
    assert dumped["proposed_diagram"] == body["proposed_diagram"]


def test_deployment_urls_keep_only_the_links_that_were_set():
    urls = GitHubService("token").get_deployment_urls("octocat", "shop")
    response = dict(success=True, repo_url="u", repo_name="shop", owner="octocat",
                    files_uploaded=1, message="m", is_first_deploy=True, deployment_type="chatbot")

    first = DeployToGitHubResponse(**response, deployment_urls=urls).model_dump()
    redeploy = DeployToGitHubResponse(
        **response, deployment_urls={**urls, "live_chatbot": "https://c", "render_dashboard": "https://d"},
    ).model_dump()

    assert first["deployment_urls"] == urls
    assert set(redeploy["deployment_urls"]) == {"github", "render", "live_chatbot", "render_dashboard"}
    with pytest.raises(ValidationError):
        TypeAdapter(DeploymentUrls).validate_python({"render_dashboard": "https://d"})


class _Pulls:
    def __init__(self, pull):
        self.pull = pull

    async def get_branch_head(self, owner, repo, branch):
        return "a" * 40

    async def get_pull_request_for_branch(self, owner, repo, branch):
        return self.pull


@pytest.mark.parametrize("pull, state", [
    (None, None),
    ({"number": 1, "html_url": "h", "state": "open", "merged": False}, "open"),
    ({"number": 1, "html_url": "h", "state": "closed", "merged": False}, "closed"),
    ({"number": 1, "html_url": "h", "state": "closed", "merged": True}, "merged"),
    # ``get_pull_request_for_branch`` defaults a missing state to "".
    ({"number": 1, "html_url": "h", "state": "", "merged": False}, None),
])
def test_pending_status_states(pull, state):
    status = asyncio.run(_pending_status(_Pulls(pull), "o", "r", "besser/sync-1"))

    assert status.pull_request_state == state


def test_openapi_marks_sent_defaults_required_and_keeps_request_schemas(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    assert "status" in schemas["SaveProjectResponse"]["required"]
    assert "pending" in schemas["SyncStatusResponse"]["required"]
    assert set(schemas["DeploymentUrls"]["required"]) == {"github", "render"}
    assert {"is_first_deploy", "deployment_type"} <= set(schemas["DeployToGitHubResponse"]["required"])
    # Shared with the KGResolutionInput request body: one schema, defaults optional.
    assert not any(name.endswith(("-Input", "-Output")) for name in schemas)
    assert "affectedNodeIds" not in schemas["KGIssueResponse"]["required"]
