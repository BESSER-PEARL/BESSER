"""Who added a class or attribute: stamped at save, kept by the merge, invisible to the converters."""

import asyncio
import copy
import json
from datetime import datetime, timezone

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import github_deploy_api
from besser.utilities.web_modeling_editor.backend.services.deployment import repo_model
from besser.utilities.web_modeling_editor.backend.services.deployment.github_service import GitHubService
from besser.utilities.web_modeling_editor.backend.services.deployment.model_attribution import (
    attribute_project,
    attributed_ids,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.model_merge import merge_documents
from tests.utilities.web_modeling_editor.backend.fake_github import FakeGitHub, use_fake_github

SESSION = {"X-GitHub-Session": "sess"}
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
ANA = {"addedBy": "ana", "addedAt": "2026-11-04T12:00:00Z"}


@pytest.fixture
def github(tmp_path, monkeypatch):
    fake = FakeGitHub(root=str(tmp_path / "github"))
    use_fake_github(monkeypatch, fake)
    monkeypatch.setattr(github_deploy_api, "get_user_token", lambda _sessions, sid: fake.token if sid == "sess" else None)
    return fake


def _call(method: str, url: str, **kwargs) -> httpx.Response:
    async def run():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, url, timeout=60, **kwargs)
    return asyncio.run(run())


def _model(*classes: str, attributes: tuple = ()) -> dict:
    """An Apollon class diagram; each name in ``attributes`` is an attribute of the first class."""
    elements = {}
    for index, name in enumerate(classes):
        elements[f"id-{name}"] = {
            "id": f"id-{name}", "type": "Class", "name": name, "owner": None,
            "bounds": {"x": index * 300, "y": 0, "width": 200, "height": 100},
            "attributes": [f"id-{a}" for a in attributes] if index == 0 else [], "methods": [],
        }
    for attribute in attributes:
        elements[f"id-{attribute}"] = {
            "id": f"id-{attribute}", "type": "ClassAttribute", "name": f"+ {attribute}: str", "owner": f"id-{classes[0]}",
            "bounds": {"x": 0, "y": 40, "width": 200, "height": 30},
        }
    return {"version": "3.0.0", "type": "ClassDiagram", "size": {"width": 800, "height": 600},
            "interactive": {"elements": {}, "relationships": {}}, "elements": elements, "relationships": {},
            "assessments": {}}


def _envelope(model: dict, attribution: dict | None = None) -> dict:
    project = {
        "id": "proj-1", "type": "Project", "name": "Library", "createdAt": "2026-04-15T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "cd1", "title": "Library", "model": model}]},
        "currentDiagramIndices": {"ClassDiagram": 0},
    }
    if attribution is not None:
        project["attribution"] = attribution
    return {"project": project, "exportedAt": "2026-11-04T00:00:00Z", "version": "2.0.0"}


def _save(github, login: str, envelope: dict, base_sha: str | None) -> dict:
    github.login = login
    body = {"owner": "ana", "repo": "library", "commit_message": "Save",
            "project_data": envelope["project"], "project_export": envelope}
    if base_sha:
        body["base_commit_sha"] = base_sha
    response = _call("POST", "/besser_api/github/project/save", headers=SESSION, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _in_repo(github, repo) -> dict:
    return json.loads(github.files(repo)[repo_model.MODEL_FILE][1])["project"]


def test_two_people_each_credited_for_the_class_they_added(github):
    """A pushes, B adds a class, A adds another from the older commit: the merge keeps both credits."""
    repo = github.create_repo("library")
    first = _save(github, "ana", _envelope(_model("Book")), None)
    assert first["status"] in ("pushed", "merged")
    assert set(_in_repo(github, repo)["attribution"]) == {"id-Book"}

    # Ben started from Ana's commit and sends what his editor holds, her record included.
    ana_model = _in_repo(github, repo)
    ben = _save(github, "ben", _envelope(_model("Book", "Loan"), ana_model["attribution"]), first["commit_sha"])
    assert ben["status"] == "pushed"

    merged = _save(github, "ana", _envelope(_model("Book", "Member"), ana_model["attribution"]), first["commit_sha"])
    assert merged["status"] == "merged"

    records = _in_repo(github, repo)["attribution"]
    assert {element: record["addedBy"] for element, record in records.items()} == {
        "id-Book": "ana", "id-Loan": "ben", "id-Member": "ana",
    }
    assert records["id-Loan"]["addedAt"].endswith("Z")


def test_an_attribute_added_to_an_existing_class_is_credited_alone(github):
    repo = github.create_repo("library")
    first = _save(github, "ana", _envelope(_model("Book")), None)

    _save(github, "ben", _envelope(_model("Book", attributes=("title",))), first["commit_sha"])

    records = _in_repo(github, repo)["attribution"]
    assert records["id-Book"]["addedBy"] == "ana"
    assert records["id-title"]["addedBy"] == "ben"


def test_a_repo_saved_before_attribution_credits_only_what_is_new(github):
    """The old classes were not added by whoever saves next, and nothing else in the old model changes."""
    repo = github.create_repo("library", files={
        repo_model.MODEL_FILE: json.dumps(_envelope(_model("Book", "Author"))),
    })
    head = github.ref(repo, "main")

    _save(github, "ben", _envelope(_model("Book", "Author", "Loan")), head)

    assert set(_in_repo(github, repo)["attribution"]) == {"id-Loan"}


def test_an_unchanged_old_model_gets_no_attribution_at_all(github):
    repo = github.create_repo("library", files={
        repo_model.MODEL_FILE: json.dumps(_envelope(_model("Book"))),
    })

    _save(github, "ben", _envelope(_model("Book")), github.ref(repo, "main"))

    assert "attribution" not in _in_repo(github, repo)


def test_a_model_an_older_editor_saved_is_the_base(github):
    repo = github.create_repo("library", files={
        repo_model.LEGACY_PROJECT_FILE: json.dumps(_envelope(_model("Book"))),
    })

    _save(github, "ben", _envelope(_model("Book", "Loan")), github.ref(repo, "main"))

    assert set(_in_repo(github, repo)["attribution"]) == {"id-Loan"}


def test_a_sender_that_dropped_the_records_does_not_erase_them(github):
    """An older editor, or a Spec-Driven export, may hold no records; the three-way merge would delete them."""
    repo = github.create_repo("library")
    first = _save(github, "ana", _envelope(_model("Book")), None)

    _save(github, "ben", _envelope(_model("Book", "Loan")), first["commit_sha"])

    assert set(_in_repo(github, repo)["attribution"]) == {"id-Book", "id-Loan"}


def test_a_github_failure_while_reading_the_base_does_not_fail_the_save(github, monkeypatch):
    repo = github.create_repo("library", files={"README.md": "# library\n"})
    original = GitHubService.get_file_content

    async def fail_on_model(self, owner, repo_name, file_path, branch="main", ref=None):
        if file_path == repo_model.MODEL_FILE and ref == head:
            request = httpx.Request("GET", "https://api.github.com")
            raise httpx.HTTPStatusError("boom", request=request, response=httpx.Response(500, request=request))
        return await original(self, owner, repo_name, file_path, branch, ref)

    head = github.ref(repo, "main")
    monkeypatch.setattr(GitHubService, "get_file_content", fail_on_model)

    saved = _save(github, "ben", _envelope(_model("Book")), head)

    assert saved["status"] == "pushed"
    assert "attribution" not in _in_repo(github, repo)


def test_a_github_failure_while_asking_who_saves_does_not_fail_the_save(github, monkeypatch):
    repo = github.create_repo("library", files={"README.md": "# library\n"})
    head = github.ref(repo, "main")

    async def no_user(self):
        request = httpx.Request("GET", "https://api.github.com/user")
        raise httpx.HTTPStatusError("boom", request=request, response=httpx.Response(502, request=request))

    monkeypatch.setattr(GitHubService, "get_authenticated_user", no_user)

    saved = _save(github, "ben", _envelope(_model("Book")), head)

    assert saved["status"] == "pushed"
    assert "attribution" not in _in_repo(github, repo)


def test_create_repo_credits_the_person_who_created_it(github):
    envelope = _envelope(_model("Book"))
    response = _call("POST", "/besser_api/github/project/create-repo", headers=SESSION, json={
        "repo_name": "fresh", "project_data": envelope["project"], "project_export": envelope,
    })
    assert response.status_code == 200, response.text
    assert _in_repo(github, github.repos["ana/fresh"])["attribution"]["id-Book"]["addedBy"] == "ana"


# ---------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------


def test_the_merge_keeps_records_added_on_both_sides():
    base = _envelope(_model("Book"), {"id-Book": ANA})
    ours = _envelope(_model("Book", "Member"), {"id-Book": ANA, "id-Member": ANA})
    theirs = _envelope(_model("Book", "Loan"), {"id-Book": ANA, "id-Loan": {**ANA, "addedBy": "ben"}})

    merged = merge_documents(base, ours, theirs)["project"]

    assert merged["attribution"]["id-Member"]["addedBy"] == "ana"
    assert merged["attribution"]["id-Loan"]["addedBy"] == "ben"


def test_two_records_for_one_element_keep_the_one_already_in_the_repository():
    """A run seeded before a pull re-adds what a teammate added: the push must not take the credit."""
    ours = {"attribution": {"x": {"addedBy": "ana", "addedAt": "2026-11-04T13:00:00Z"}}}
    theirs = {"attribution": {"x": {"addedBy": "ben", "addedAt": "2026-11-04T12:00:00Z"}}}

    assert merge_documents({"attribution": {}}, ours, theirs) == theirs


def test_a_commit_that_dropped_the_records_does_not_conflict_with_a_new_one():
    # An older editor's save, or a hand edit, removed the block while this save added to it.
    base = _envelope(_model("Book"), {"id-Book": ANA})
    theirs = _envelope(_model("Book", "Loan"))
    ours = _envelope(_model("Book", "Member"), {"id-Book": ANA, "id-Member": ANA})

    files = repo_model.ModelMerge(ours).merge(json.dumps(base), json.dumps(theirs))

    assert files is not None
    assert set(json.loads(files.files[repo_model.MODEL_FILE])["project"]["attribution"]) == {"id-Book", "id-Member"}


def test_a_login_with_an_underscore_keeps_its_credit():
    # GitHub's managed users are named like this; the record used to vanish on the next save.
    record = {"addedBy": "jane_acme", "addedAt": ANA["addedAt"]}
    base = _envelope(_model("Book"), {"id-Book": record})["project"]

    assert attribute_project(base, base, "ben", NOW)["attribution"] == {"id-Book": record}


def test_an_unreadable_model_file_keeps_no_record_the_sender_claims(github):
    # With nothing to compare against, a record in the request is only the sender's word.
    repo = github.create_repo("library", files={"models/shop.json": "{not json"})
    forged = {"id-Book": {"addedBy": "torvalds", "addedAt": ANA["addedAt"]}}
    envelope = _envelope(_model("Book"), forged)
    github.login = "ben"

    response = _call("POST", "/besser_api/github/project/save", headers=SESSION, json={
        "owner": "ana", "repo": "library", "commit_message": "Save", "project_data": envelope["project"],
        "project_export": envelope, "base_commit_sha": github.ref(repo, "main"), "legacy_file_path": "models/shop.json",
    })

    assert response.status_code == 200, response.text
    assert "attribution" not in _in_repo(github, repo)
    assert response.json()["attribution"] == {}


# ---------------------------------------------------------------------
# What the records leave alone
# ---------------------------------------------------------------------


def test_records_change_no_b_uml_source():
    """Generators read the B-UML model: with and without records they must get the same one."""
    model = _model("Book", "Loan", attributes=("title",))

    without = repo_model.model_files(_envelope(copy.deepcopy(model)))
    with_records = repo_model.model_files(_envelope(copy.deepcopy(model), {"id-Book": ANA, "id-title": ANA}))

    sources = sorted(path for path in without if path.endswith(".py"))
    assert sources, "the model should build"
    assert sources == sorted(path for path in with_records if path.endswith(".py"))
    for path in sources:
        assert without[path] == with_records[path], path
    assert "attribution" in json.loads(with_records[repo_model.MODEL_FILE])["project"]


def test_stamping_touches_no_diagram():
    project = _envelope(_model("Book", "Loan"))["project"]
    before = copy.deepcopy(project["diagrams"])

    stamped = attribute_project(project, None, "ana", NOW)

    assert stamped["diagrams"] == before
    assert stamped["attribution"]["id-Book"] == ANA
    assert "attribution" not in project


def test_records_of_removed_elements_go():
    project = _envelope(_model("Book"), {"id-Book": ANA, "id-Gone": ANA})["project"]

    assert attribute_project(project, project, "ben", NOW)["attribution"] == {"id-Book": ANA}


def test_a_record_stays_with_its_first_author():
    base = _envelope(_model("Book"), {"id-Book": ANA})["project"]
    project = _envelope(_model("Book"), {"id-Book": ANA})["project"]

    assert attribute_project(project, base, "ben", NOW)["attribution"] == {"id-Book": ANA}


def test_a_sender_cannot_claim_an_element_the_base_lacks():
    forged = {"id-Loan": {"addedBy": "torvalds", "addedAt": "2020-01-01T00:00:00Z"}}
    base = _envelope(_model("Book"))["project"]
    project = _envelope(_model("Book", "Loan"), forged)["project"]

    assert attribute_project(project, base, "ben", NOW)["attribution"] == {
        "id-Loan": {"addedBy": "ben", "addedAt": ANA["addedAt"]},
    }


def test_a_sender_cannot_rewrite_a_record_the_base_holds():
    base = _envelope(_model("Book"), {"id-Book": ANA})["project"]
    project = _envelope(_model("Book"), {"id-Book": {"addedBy": "mallory", "addedAt": ANA["addedAt"]}})["project"]

    assert attribute_project(project, base, "ben", NOW)["attribution"] == {"id-Book": ANA}


@pytest.mark.parametrize("record", [
    {"addedBy": 5, "addedAt": ANA["addedAt"]},
    {"addedBy": "ana"},
    {"addedBy": "<!-- besser:end -->", "addedAt": ANA["addedAt"]},
    {"addedBy": "ana", "addedAt": "yesterday"},
    "ana",
])
def test_a_malformed_record_in_the_repo_is_not_kept(record):
    base = _envelope(_model("Book"), {"id-Book": record})["project"]

    assert attribute_project(base, base, "ben", NOW)["attribution"] == {}


def test_only_the_two_fields_of_a_record_are_kept():
    base = _envelope(_model("Book"), {"id-Book": {**ANA, "email": "ana@example.com"}})["project"]

    assert attribute_project(base, base, "ben", NOW)["attribution"] == {"id-Book": ANA}


def test_removing_the_last_credited_element_is_no_conflict_with_a_new_record():
    base = _envelope(_model("Book"), {"id-Book": ANA})
    theirs = _envelope(_model("Book", "Loan"), {"id-Book": ANA, "id-Loan": ANA})
    ours = copy.deepcopy(base)
    ours["project"]["diagrams"]["ClassDiagram"][0]["model"] = _model()
    del ours["project"]["attribution"]
    ours["project"] = attribute_project(ours["project"], base["project"], "ben", NOW)

    merged = merge_documents(base, ours, theirs)

    assert "id-Loan" in merged["project"]["attribution"]


def test_a_save_without_records_is_no_conflict_with_a_new_record():
    # What a failed lookup sends: the merge keeps the records rather than treating them as deleted.
    base = _envelope(_model("Book"), {"id-Book": ANA})
    theirs = _envelope(_model("Book", "Loan"), {"id-Book": ANA, "id-Loan": ANA})
    ours = _envelope(_model("Book", "Member"))

    files = repo_model.ModelMerge(ours).merge(json.dumps(base), json.dumps(theirs))

    assert files is not None
    assert set(json.loads(files.files[repo_model.MODEL_FILE])["project"]["attribution"]) == {"id-Book", "id-Loan"}


def test_a_deeply_nested_repo_model_does_not_fail_the_save(github):
    repo = github.create_repo("library", files={repo_model.MODEL_FILE: "[" * 100_000 + "]" * 100_000})

    saved = _save(github, "ben", _envelope(_model("Book")), github.ref(repo, "main"))

    assert saved["commit_sha"]


def test_a_model_in_an_older_editors_folder_is_the_base_when_no_path_is_given(github):
    # Everything used to be credited to whoever pushed first after the older editor.
    repo = github.create_repo("library", files={"models/shop.json": json.dumps(_envelope(_model("Book")))})

    _save(github, "ben", _envelope(_model("Book", "Loan")), github.ref(repo, "main"))

    assert set(_in_repo(github, repo)["attribution"]) == {"id-Loan"}


def test_a_save_returns_the_records_it_wrote(github):
    github.create_repo("library")

    saved = _save(github, "ana", _envelope(_model("Book")), None)

    assert saved["attribution"]["id-Book"]["addedBy"] == "ana"


def test_a_save_of_a_model_without_classes_returns_no_records(github):
    github.create_repo("library")

    assert _save(github, "ana", _envelope({"elements": {}}), None)["attribution"] == {}


def test_react_flow_classes_and_rows_are_attributed():
    """The same record works after the React Flow migration: members are rows of their class node."""
    model = {"nodes": [
        {"id": "c1", "type": "class", "data": {
            "name": "Book", "attributes": [{"id": "a1", "name": "title"}], "methods": [{"id": "m1", "name": "lend"}],
        }},
        {"id": "n1", "type": "colorDescription", "data": {"description": "a note"}},
    ], "edges": []}
    project = {"diagrams": {"ClassDiagram": [{"id": "cd1", "model": model}]}}

    assert attributed_ids(project) == {"c1", "a1", "m1"}


def test_relationships_and_other_diagrams_are_not_attributed():
    model = _model("Book")
    model["relationships"]["r1"] = {"id": "r1", "type": "ClassBidirectional"}
    project = {"diagrams": {"ClassDiagram": [{"model": model}], "StateMachineDiagram": [{"model": {
        "elements": {"s1": {"id": "s1", "type": "State"}},
    }}]}}

    assert attributed_ids(project) == {"id-Book"}
