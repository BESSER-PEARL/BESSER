"""Once an app has accounts, what can one account do to another's records?

The constructibility probe logs in and creates records; nothing asked what a
second account could then do to them. An app that guards the routes of one
entity still accepts, on every other router of the scaffold, a relationship
list that re-parents that entity's records.

Every finding is a warning in the run's report, worded as what was observed:
it never blocks, never reaches repair, never ranks a tree or withholds
verified completion. The observation logic is tested against a fake HTTP
layer; the generated-app tests boot the repository's own scaffold with a
hand-written login beside it.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys

import pytest

from besser.spec_driven_agent.validation import authorization_probe as authz
from besser.spec_driven_agent.validation.authorization_probe import (
    COVERAGE_PREFIX,
    PREFIX,
    _effect,
    _judge,
    _landed,
    _param_entity,
    authorization_issues,
)
from besser.spec_driven_agent.validation.constructibility import OPEN_ACCOUNT_PREFIX
from besser.spec_driven_agent.validation.issues import _classify_issue, is_completion_issue


# ------------------------------------------------- what counts as a change


@pytest.mark.parametrize("before, after, given, expected", [
    # The field reads back the value the other account sent under its name.
    ({"title": "mine"}, {"title": "theirs"}, {"sent": {"title": "theirs"}}, ["title"]),
    ({"displayName": "mine"}, {"displayName": "theirs"}, {"sent": {"display_name": "theirs"}}, ["displayName"]),
    # Nothing moved: a refusal, or a write that touched other data.
    ({"title": "mine"}, {"title": "mine"}, {"sent": {"title": "theirs"}}, []),
    # A stamp, a counter and a derived value move on their own.
    ({"title": "mine", "updatedAt": "t1"}, {"title": "mine", "updatedAt": "t2"}, {"sent": {"title": "theirs"}}, []),
    ({"loanCount": 1, "shelf_id": 4}, {"loanCount": 2, "shelf_id": 4},
     {"linked": (2,), "link_keys": ("shelf_id",)}, []),
    ({"label": "A1"}, {"label": "A1 (edited)"}, {"sent": {"title": "theirs"}}, []),
    # The link now points at the other account's record.
    ({"shelf_id": 4}, {"shelf_id": 9}, {"linked": (9,), "link_keys": ("shelf_id",)}, ["shelf_id"]),
    ({"shelf_id": "a1"}, {"shelf_id": "b7"}, {"linked": ("b7",), "link_keys": ("shelf_id",)}, ["shelf_id"]),
    # It moved, but not to a record of the other account.
    ({"shelf_id": 4}, {"shelf_id": 5}, {"linked": (9,), "link_keys": ("shelf_id",)}, []),
    # A response that shows the link under another name: any ``*id`` field, never a count.
    ({"shelfId": 4, "views": 8}, {"shelfId": 9, "views": 9}, {"linked": (9,), "link_keys": ("shelf_id",)}, ["shelfId"]),
    # A flag or a null equals what was sent too easily to prove anything.
    ({"done": False}, {"done": True}, {"sent": {"done": True}}, []),
    ({"note": "x"}, {"note": None}, {"sent": {"note": None}}, []),
])
def test_only_what_the_other_account_supplied_counts_as_its_change(before, after, given, expected):
    assert _landed(before, after, **given) == expected


class _OneRecord:
    """A fake HTTP layer serving one record of account A; ``on_write`` is what the app does with
    another account's request: ``(status, record afterwards)``, None for a deleted record, a
    status for a record its owner is no longer served."""

    def __init__(self, record, on_write):
        self.record, self.on_write = record, on_write
        self.calls: list = []

    async def call(self, actor, method, path, **body):
        self.calls.append((actor, method, path))
        if method == "GET":
            if self.record is None:
                return 404, {"detail": "not found"}
            if isinstance(self.record, int):
                return self.record, {"detail": "not authenticated"}
            return 200, {"book": dict(self.record), "pages_ids": [1, 2]}
        status, self.record = self.on_write(dict(self.record), body.get("json"))
        return status, {"shelf": {"id": 9}}


async def _read(api):
    status, answer = await api.call("a", "GET", "/book/1/")
    if status == 404:
        return None
    if status in (401, 403):
        return authz._LOCKED_OUT
    return dict(answer["book"]) if status == 200 else authz._UNREADABLE


_BOOK = {"id": 1, "title": "mine", "shelf_id": 4, "updatedAt": "t1", "loans": 0}


@pytest.mark.parametrize("sent, app, given, expected", [
    # A refusal is a pass.
    ({"title": "theirs"}, lambda record, body: (403, record), {}, None),
    # A 2xx that changed nothing of account A's is not a finding.
    ({"name": "a shelf"}, lambda record, body: (200, record), {}, None),
    ({"title": "theirs"}, lambda record, body: (200, {**record, "updatedAt": "t2", "loans": 1}), {}, None),
    # The record took the other account's value.
    ({"title": "theirs"}, lambda record, body: (200, {**record, **body}), {}, {"keys": ["title"]}),
    # The status is not the oracle: a failed request that still wrote is a change.
    ({"title": "theirs"}, lambda record, body: (500, {**record, **body}), {}, {"keys": ["title"]}),
    # Another entity's relationship list moved the record to the other account's parent.
    ({"name": "a shelf", "books": [1]}, lambda record, body: (200, {**record, "shelf_id": 9, "updatedAt": "t2"}),
     {"linked": (9,), "link_keys": ("shelf_id",)}, {"keys": ["shelf_id"]}),
    # The list was accepted and ignored.
    ({"name": "a shelf", "books": [1]}, lambda record, body: (200, record),
     {"linked": (9,), "link_keys": ("shelf_id",)}, None),
    # The record is gone.
    (None, lambda record, body: (200, None), {}, {"gone": True}),
    (None, lambda record, body: (404, record), {}, None),
    # The owner is refused its own record right after the other account's accepted request.
    ({"email": "theirs"}, lambda record, body: (200, 401), {}, {"locked": True}),
    # Refused both: the owner's session ended for a reason of its own.
    ({"email": "theirs"}, lambda record, body: (403, 401), {}, None),
    # The read-back failed otherwise: no verdict.
    ({"title": "theirs"}, lambda record, body: (200, 500), {}, None),
])
def test_the_record_read_back_decides_not_the_status(sent, app, given, expected):
    api = _OneRecord(dict(_BOOK), app)
    method = "DELETE" if sent is None else "PUT"
    seen = asyncio.run(_effect(api, lambda: _read(api), "b", method, "/book/1/", sent))
    assert [actor for actor, _, _ in api.calls] == ["a", "b", "a"]  # read as its owner either side
    assert _judge(seen, sent=sent, **given) == expected


def test_a_record_its_owner_cannot_read_is_not_tested():
    api = _OneRecord(None, lambda record, body: (200, record))
    assert asyncio.run(_effect(api, lambda: _read(api), "b", "DELETE", "/book/1/")) is None
    assert api.calls == [("a", "GET", "/book/1/")]  # the other account's request was never sent


@pytest.mark.parametrize("field, value, leaked", [
    ("passwordHash", "pbkdf2_sha256$240000$c2FsdA==$ZGlnZXN0", True),
    ("password", "$2b$12$abcdefghijklmnopqrstuv", True),
    ("password", "Probe-Passw0rd-2817", True),          # the password the account registered with
    ("apiKey", "sk4f9a0c2e7b1d4f6a8c0e2b4d", True),
    ("accessToken", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln", True),
    # Named like a credential, holding none.
    ("token_balance", 120, False), ("loyaltyToken", "GOLD", False), ("has_password", "yes", False),
    ("password_reset_required", "false", False), ("apiKey", "********", False), ("password", "[hidden]", False),
    ("secret", "the title of a long novel", False),
])
def test_a_leaked_credential_needs_a_credential_shaped_value_beside_the_account(field, value, leaked):
    email = "authz.a@example.com"
    body = [{"id": 1, "email": email, field: value}, {"id": 2, "email": "other@example.com", "password": "$2b$12$x"}]
    assert authz._credentials_beside(body, email, "Probe-Passw0rd-2817") == ({field} if leaked else set())


@pytest.mark.parametrize("param, entity, expected", [
    ("book_id", "Book", "Book"), ("id", "Book", "Book"),
    ("shelf_id", "Book", "Shelf"),          # another entity, by name
    ("borrowedBy_id", "Book", "Member"),    # another entity, by the route entity's relationship
    ("isbn", "Book", None),
])
def test_a_path_parameter_is_filled_from_the_record_it_names(param, entity, expected):
    orm = {"Book": {"borrowedBy": ("Member", False)}, "Shelf": {}, "Member": {}}
    assert _param_entity(param, entity, orm) == expected


# ------------------------------------------------------- the wording of each warning


def _report(**parts) -> dict:
    return {"boot": "ok", "account_entity": None, "entities": {}, "effects": [], "anonymous": [],
            "exposed": [], "stale_token": [], "uncovered": {}, "requests": 10, **parts}


def _effect_entry(route, entity, kind="update", **extra) -> dict:
    return {"route": route, "entity": entity, "kind": kind, "keys": ["title"], "status": 200, **extra}


def _walled(rel="backend", **extra) -> dict:
    return {"backend": rel, "boot": "ok", "auth": {"source": "its register/login routes"}, **extra}


def _child(monkeypatch, *reports):
    """Make the child answer with ``reports``, one per run."""
    queue = list(reports)
    monkeypatch.setattr(authz, "_run_child", lambda *args, **kwargs: queue.pop(0))


def _said(monkeypatch, tmp_path, report, request="", backend=None) -> list[str]:
    """What the production entry point says about one child report."""
    _child(monkeypatch, report)
    return authorization_issues(str(tmp_path), [backend or _walled()], request)


_SIDE_DOOR = _effect_entry("PUT /shelf/{shelf_id}/", "Book", "link", keys=["shelf_id"], field="books", holder="Shelf")
_RULE = {"rule": {"route": "PUT /book/{book_id}/", "refused": 403}}
# Words a warning must not use: it states what happened, it does not judge or instruct.
_JUDGING = re.compile(r"goes around|take ?over|does not own|remove|stop accepting|fix site|apply here|"
                      r"keep the|blocker|restrict", re.I)


def test_every_finding_is_a_warning_that_blocks_nothing(monkeypatch, tmp_path):
    report = _report(
        account_entity="Member", entities={"Book": _RULE}, effects=[
            _SIDE_DOOR, _effect_entry("DELETE /member/bulk/", "Member", "delete", keys=[])],
        anonymous=[{"route": "POST /order/", "status": 201}],
        exposed=[{"route": "GET /member/", "fields": ["email"], "credentials": ["password"]}],
        stale_token=[{"route": "POST /genre/", "status": 200}],
        uncovered={"Loan": "POST /loan/ answered 422"}, stopped="the 60s time limit")
    issues = _said(monkeypatch, tmp_path, report)
    assert len(issues) == 8
    for issue in issues:
        classified = _classify_issue(issue)
        assert classified.severity == "warning" and not is_completion_issue(classified), issue
        assert issue.startswith((f"{PREFIX} backend: ", f"{COVERAGE_PREFIX} backend: ")), issue
        assert not _JUDGING.search(issue), issue


def test_a_side_door_is_stated_as_observed(monkeypatch, tmp_path):
    report = _report(entities={"Book": _RULE}, effects=[_SIDE_DOOR])
    assert _said(monkeypatch, tmp_path, report) == [
        f"{PREFIX} backend: another account could change Book.shelf_id through PUT /shelf/{{shelf_id}}/; "
        "PUT /book/{book_id}/ refused that account"]


@pytest.mark.parametrize("effect, said", [
    (_effect_entry("PATCH /book/{book_id}/", "Book"), "another account could change Book.title through PATCH /book/{book_id}/"),
    (_effect_entry("DELETE /book/bulk/", "Book", "delete", keys=[]),
     "another account could delete a Book through DELETE /book/bulk/"),
    (_effect_entry("POST /book/{book_id}/tags/{tag_id}/", "Book", "relation", keys=[], relation="tags", target="Tag"),
     "another account could change Book.tags through POST /book/{book_id}/tags/{tag_id}/"),
    (_effect_entry("PATCH /book/{book_id}/", "Book", "lockout", keys=[]),
     "after another account's request through PATCH /book/{book_id}/ was accepted, the account that created a Book "
     "could no longer read it"),
])
def test_every_route_that_changed_a_record_the_app_refused_elsewhere_is_named(monkeypatch, tmp_path, effect, said):
    assert _said(monkeypatch, tmp_path, _report(entities={"Book": _RULE}, effects=[effect])) == [
        f"{PREFIX} backend: {said}; PUT /book/{{book_id}}/ refused that account"]


def test_a_record_the_app_refuses_no_one_on_yields_nothing(monkeypatch, tmp_path):
    """A shared record, or one locked for everyone: with no refusal of the other account alone,
    every change of it is the app's design."""
    effects = [_effect_entry("PUT /book/{book_id}/", "Book"), _effect_entry("DELETE /book/{book_id}/", "Book", "delete"),
               _SIDE_DOOR]
    assert _said(monkeypatch, tmp_path, _report(entities={"Book": {}}, effects=effects)) == []


def test_the_route_that_refused_is_not_its_own_side_door(monkeypatch, tmp_path):
    """The rule's route and the changing route are the same only when the app changed between two reads."""
    report = _report(entities={"Book": _RULE}, effects=[_effect_entry("PUT /book/{book_id}/", "Book")])
    assert _said(monkeypatch, tmp_path, report) == []


def test_a_refused_single_delete_makes_the_bulk_delete_a_finding(monkeypatch, tmp_path):
    """Delete-only ownership: everyone edits, only the creator deletes; the bulk route lets anyone delete."""
    report = _report(entities={"Book": {"delete_rule": {"route": "DELETE /book/{book_id}/", "refused": 403}}},
                     effects=[_effect_entry("PUT /book/{book_id}/", "Book"), _SIDE_DOOR,
                              _effect_entry("DELETE /book/bulk/", "Book", "delete", keys=[])])
    assert _said(monkeypatch, tmp_path, report) == [
        f"{PREFIX} backend: another account could delete a Book through DELETE /book/bulk/; "
        "DELETE /book/{book_id}/ refused that account"]


@pytest.mark.parametrize("effect, said", [
    (_effect_entry("PUT /member/{member_id}/", "Member", keys=["email", "password"]),
     "another account could change Member.email, Member.password through PUT /member/{member_id}/; "
     "Member.email, Member.password identify the account"),
    (_effect_entry("POST /member/{member_id}/rename/", "Member", keys=["nickname"]),
     "another account could change Member.nickname through POST /member/{member_id}/rename/"),
    (_effect_entry("DELETE /member/{member_id}/", "Member", "delete", keys=[]),
     "another account could delete a Member account through DELETE /member/{member_id}/"),
    (_effect_entry("PUT /member/{member_id}/", "Member", "lockout", keys=[]),
     "after another account's request through PUT /member/{member_id}/ was accepted, the account could no "
     "longer read its own record"),
])
def test_a_change_of_an_account_record_is_named_with_no_rule_needed(monkeypatch, tmp_path, effect, said):
    """Only identity fields, or an owner locked out, are described as touching the account itself."""
    assert _said(monkeypatch, tmp_path, _report(account_entity="Member", effects=[effect])) == [
        f"{PREFIX} backend: {said}"]


_TEAM_LINK = _effect_entry("POST /team/", "Member", "link", keys=["team_id"], field="members", holder="Team")


def test_a_link_on_an_account_record_needs_a_refusal_elsewhere(monkeypatch, tmp_path):
    assert _said(monkeypatch, tmp_path, _report(account_entity="Member", effects=[_TEAM_LINK])) == []
    rule = {"Member": {"rule": {"route": "PUT /member/{member_id}/", "refused": 403}}}
    assert _said(monkeypatch, tmp_path, _report(account_entity="Member", entities=rule, effects=[_TEAM_LINK]),
                 request="Any member can create a team and add other members to it.") == [
        f"{PREFIX} backend: another account could change Member.team_id through POST /team/; "
        "PUT /member/{member_id}/ refused that account"]


def test_a_write_route_open_to_visitors_is_named(monkeypatch, tmp_path):
    report = _report(anonymous=[{"route": "POST /book/{book_id}/view/", "status": 200},
                                {"route": "POST /order/", "status": 201}])
    assert _said(monkeypatch, tmp_path, report) == [
        f"{PREFIX} backend: a caller who is not logged in could successfully send POST /book/{{book_id}}/view/, "
        "POST /order/"]


_EXPOSED = [{"route": "GET /member/", "fields": ["email", "phone"], "credentials": []},
            {"route": "GET /book/", "fields": ["email"], "credentials": []}]


def test_contact_data_shown_to_visitors_is_named_unless_the_request_makes_it_public(monkeypatch, tmp_path):
    assert _said(monkeypatch, tmp_path, _report(exposed=_EXPOSED)) == [
        f"{PREFIX} backend: a caller who is not logged in could read an account's email and phone through "
        "GET /member/, GET /book/"]
    for request in ("Members have public profiles with their contact details.",
                    "A member's email and phone are visible to everyone on the club page."):
        assert _said(monkeypatch, tmp_path, _report(exposed=_EXPOSED), request) == []


@pytest.mark.parametrize("request_text, said", [
    ("", "`passwordHash`, `shareToken`"),
    # A share token of a public profile is meant to be seen; the password never is.
    ("Profiles are public and can be shared by link.", "`passwordHash`"),
])
def test_a_credential_beside_an_account_is_named_as_shown_to_other_callers(monkeypatch, tmp_path, request_text, said):
    exposed = [{"route": "GET /member/", "fields": [], "credentials": ["passwordHash", "shareToken"]}]
    assert _said(monkeypatch, tmp_path, _report(exposed=exposed), request_text) == [
        f"{PREFIX} backend: GET /member/ returned an account's {said} to other callers"]


def test_a_share_token_of_a_public_profile_is_no_finding(monkeypatch, tmp_path):
    exposed = [{"route": "GET /member/", "fields": ["email"], "credentials": ["shareToken"]}]
    assert _said(monkeypatch, tmp_path, _report(exposed=exposed), "Profiles are public.") == []


def test_a_deleted_accounts_token_that_still_writes_is_named(monkeypatch, tmp_path):
    assert _said(monkeypatch, tmp_path, _report(stale_token=[{"route": "POST /genre/", "status": 200}])) == [
        f"{PREFIX} backend: POST /genre/ still accepted the token of an account that had deleted itself and "
        "could no longer log in"]


@pytest.mark.parametrize("report, said", [
    ({"boot": "error", "error": "timed out after 90s"},
     "the check did not run (timed out after 90s); what one account can do to another's records was not checked"),
    (_report(skipped="two accounts could not be registered and logged in through the app's own routes "
                     "(POST /auth/login answered 401)"),
     "two accounts could not be registered and logged in through the app's own routes (POST /auth/login answered "
     "401); what one account can do to another's records was not checked"),
    (_report(stopped="the 60s time limit"),
     "the check stopped early (the 60s time limit) after 10 requests; the routes it had not reached were not checked"),
    (_report(uncovered={"Loan": "POST /loan/ answered 422", "Fine": "POST /fine/ needs a record that could not be created"}),
     "records of Fine (POST /fine/ needs a record that could not be created); Loan (POST /loan/ answered 422) could "
     "not be set up for the check, so what other accounts can do to them was not checked"),
])
def test_what_the_check_did_not_cover_is_one_plain_sentence(monkeypatch, tmp_path, report, said):
    assert _said(monkeypatch, tmp_path, report) == [f"{COVERAGE_PREFIX} backend: {said}"]


def test_a_route_the_probe_already_reported_is_not_named_again(monkeypatch, tmp_path):
    """The constructibility probe warns about anonymous writes to account records and about one logged-in
    account changing another's account fields; those routes are its findings."""
    report = _report(account_entity="Member", effects=[
        _effect_entry("PUT /member/{member_id}/", "Member", keys=["email"]),
        _effect_entry("DELETE /member/bulk/", "Member", "delete", keys=[])], anonymous=[
        {"route": "DELETE /member/{member_id}/", "status": 200}, {"route": "POST /genre/", "status": 200}])
    backend = _walled(open_account_writes=[{
        "table": "Member", "login": "/auth/login", "routes": ["DELETE /member/{member_id}/"],
        "fields": [{"route": "PUT /member/{member_id}/", "fields": ["email"]}]}])
    assert _said(monkeypatch, tmp_path, report, backend=backend) == [
        f"{PREFIX} backend: another account could delete a Member account through DELETE /member/bulk/",
        f"{PREFIX} backend: a caller who is not logged in could successfully send POST /genre/"]


def test_a_backend_without_accounts_is_not_checked(monkeypatch, tmp_path):
    def never(*args, **kwargs):
        raise AssertionError("no registration route and no login wall: the check must not run")

    monkeypatch.setattr(authz, "_run_child", never)
    for name in ("open", "broken"):
        (tmp_path / name / "routers").mkdir(parents=True)
        (tmp_path / name / "routers" / "event.py").write_text(
            '@router.post("/event/{event_id}/register")\ndef attend(): ...\n', encoding="utf-8")
    backends = [{"backend": "open", "boot": "ok"}, {"backend": "broken", "boot": "import_error", "auth": {"source": "x"}}]
    assert authorization_issues(str(tmp_path), backends) == []


def test_an_app_with_a_registration_route_is_checked_though_no_create_was_refused(monkeypatch, tmp_path):
    """Public creates beside protected edits: the probe meets no login wall, the app still has accounts."""
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text(
        'router = APIRouter(prefix="/auth")\n\n@router.post("/register")\ndef register(): ...\n', encoding="utf-8")
    backend = {"backend": "backend", "boot": "ok"}
    assert len(_said(monkeypatch, tmp_path, _report(entities={"Book": _RULE}, effects=[_SIDE_DOOR]),
                     backend=backend)) == 1
    # A registration literal the app serves under no such route: nothing to say when no wall was met.
    assert _said(monkeypatch, tmp_path, _report(skipped="the app has no open registration route", no_accounts=True),
                 backend=backend) == []


def test_a_crash_of_the_check_is_reported_and_never_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(authz, "_run_child", lambda *args, **kwargs: 1 / 0)
    [note] = authorization_issues(str(tmp_path), [_walled()])
    assert note.startswith(f"{COVERAGE_PREFIX} backend: the check did not run (it crashed: ZeroDivisionError")


def test_a_crash_while_preparing_the_check_is_reported_and_never_raised(monkeypatch, tmp_path):
    """Phase 3 calls this on every sweep: a failure before the child starts must not stop the run."""
    def unreadable(_domain_model):
        raise AttributeError("model unreadable")

    monkeypatch.setattr(authz._probe, "abstract_entities", unreadable)
    [note] = authorization_issues(str(tmp_path), [_walled()])
    assert note.startswith(f"{COVERAGE_PREFIX} backend: the check did not run (it crashed: AttributeError")


def test_the_child_starts_without_the_besser_package(tmp_path):
    """It runs where only the generated app's own dependencies are installed. A second package
    import in a sibling it loads would stop it there, and only there."""
    script = (
        "import runpy, sys\n"
        "class NoBesser:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'besser' or name.startswith('besser.'):\n"
        "            raise ImportError('besser is not installed here')\n"
        "sys.meta_path.insert(0, NoBesser())\n"
        f"sys.path.insert(0, {os.path.dirname(authz.__file__)!r})\n"
        f"runpy.run_path({authz.__file__!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert "the app did not start: ModuleNotFoundError: No module named 'main_api'" in result.stdout, result.stderr


# ----------------------------------------------------- a generated backend

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# Tokens are kept in memory. Reading is public; every write needs a token.
_LOGIN_MODULE = '''
import secrets
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from sql_alchemy import *

TOKENS: dict = {}
OWNERS: dict = {}
ACCOUNTS: dict = {}
bearer = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/auth")


def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if request.method == "GET":
        return None
    if credentials is None or credentials.credentials not in TOKENS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return TOKENS[credentials.credentials]


def require_owner(task_id: int, user) -> None:
    if OWNERS.get(task_id) != user:
        raise HTTPException(status_code=403, detail="Not your task")


class Credentials(BaseModel):
    email: str
    password: str
    phone: str

{register}

@router.post("/login")
def login(data: Credentials, database: Session = Depends(get_db)):
    user = check(data, database)
    if user is None:
        raise HTTPException(status_code=401, detail="bad credentials")
    token = secrets.token_urlsafe(16)
    TOKENS[token] = user
    return {"access_token": token, "token_type": "bearer"}
'''

_REGISTER_IN_MEMORY = '''
@router.post("/register", status_code=201)
def register(data: Credentials):
    ACCOUNTS[data.email] = data.password
    return {"email": data.email}


def check(data, database):
    return data.email if ACCOUNTS.get(data.email) == data.password else None
'''

# The account is a row of the modelled Member class; a token outlives the row it was issued for.
_REGISTER_AS_MEMBER = '''
@router.post("/register", status_code=201)
def register(data: Credentials, database: Session = Depends(get_db)):
    member = Member(email=data.email, phone=data.phone, password=data.password)
    database.add(member)
    database.commit()
    database.refresh(member)
    return {"id": member.id}


def check(data, database):
    member = database.query(Member).filter(Member.email == data.email, Member.password == data.password).first()
    return member and member.id
'''


def _write_login(backend, routers: int, register: str) -> None:
    """Put the login module beside a generated backend and require its token on every router."""
    (backend / "auth.py").write_text(_LOGIN_MODULE.replace("{register}", register), encoding="utf-8")
    main = backend / "main_api.py"
    text, guarded = re.subn(r"app\.include_router\((\w+_router)\.router\)",
                            r"app.include_router(\1.router, dependencies=[Depends(get_current_user)])",
                            main.read_text(encoding="utf-8"))
    assert guarded == routers
    main.write_text("import auth as _auth\nfrom auth import get_current_user\n" + text
                    + "\napp.include_router(_auth.router)\n", encoding="utf-8")


def _login_scaffold(tmp_path, *, member_accounts: bool = False, own_tasks: bool = True,
                    own_bulk_delete: bool = True, tags: bool = False, rating: bool = False) -> str:
    """A generator-fresh Project/Task(/Member/Tag) backend with a login beside it.

    Every write needs a token. With ``own_tasks`` the Task router alone also
    checks that the caller created the task (its bulk delete too, with
    ``own_bulk_delete``); the sibling routers stay as generated.
    """
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
    )
    from besser.generators.backend import BackendGenerator

    project = Class(name="Project", attributes={Property(name="name", type=StringType)})
    task = Class(name="Task", attributes={Property(name="title", type=StringType),
                                          *([Property(name="rating", type=IntegerType)] if rating else [])})
    types = {project, task}
    associations = {BinaryAssociation(name="project_tasks", ends={
        Property(name="project", type=project, multiplicity=Multiplicity(1, 1)),
        Property(name="tasks", type=task, multiplicity=Multiplicity(0, "*")),
    })}
    if member_accounts:
        types.add(Class(name="Member", attributes={
            Property(name="email", type=StringType), Property(name="phone", type=StringType),
            Property(name="password", type=StringType)}))
    if tags:
        tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
        types.add(tag)
        associations.add(BinaryAssociation(name="task_tags", ends={
            Property(name="taggedTasks", type=task, multiplicity=Multiplicity(0, "*")),
            Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
        }))
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=DomainModel(name="Board", types=types, associations=associations),
                     output_dir=str(backend)).generate()
    _write_login(backend, len(types), _REGISTER_AS_MEMBER if member_accounts else _REGISTER_IN_MEMORY)

    if own_tasks:
        router = backend / "routers" / "task.py"
        text, heads = re.subn(r"(async def (?:create|update|delete|bulk_delete)_task\([^)]*Depends\(get_db\))\)",
                              r"\1, user=Depends(get_current_user))", router.read_text(encoding="utf-8"))
        text, checks = re.subn(r"(async def (?:update|delete)_task\(.*\n)", r"\1    require_owner(task_id, user)\n", text)
        if own_bulk_delete:
            text = re.sub(r"(async def bulk_delete_task\(.*\n)",
                          r"\1    for task_id in ids:\n        require_owner(task_id, user)\n", text)
        text, records = re.subn(r"( +)database\.add\(db_task\)\n",
                                r"\1database.add(db_task)\n\1database.flush()\n\1OWNERS[db_task.id] = user\n",
                                text, count=1)
        assert (heads, checks, records) == (4, 2, 1) and ("for task_id in ids" in text) is own_bulk_delete
        router.write_text("from auth import OWNERS, get_current_user, require_owner\n" + text, encoding="utf-8")
    return str(tmp_path)


def _guard_members(backend) -> None:
    """A member may change and delete only its own row."""
    router = backend / "routers" / "member.py"
    text, heads = re.subn(r"(async def (?:update|delete|bulk_delete)_member\([^)]*Depends\(get_db\))\)",
                          r"\1, user=Depends(get_current_user))", router.read_text(encoding="utf-8"))
    text, checks = re.subn(
        r"(async def (?:update|delete)_member\(.*\n)",
        r'\1    if user != member_id:\n        raise HTTPException(status_code=403, detail="Not your account")\n', text)
    text, bulk = re.subn(
        r"(async def bulk_delete_member\(.*\n)",
        r'\1    if ids != [user]:\n        raise HTTPException(status_code=403, detail="Not your account")\n', text)
    assert (heads, checks, bulk) == (3, 2, 1)
    router.write_text("from auth import get_current_user\n" + text, encoding="utf-8")


def _club_scaffold(tmp_path) -> str:
    """Team 1 - * Member, the accounts being Member rows; a member may change only its own row."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
    )
    from besser.generators.backend import BackendGenerator

    team = Class(name="Team", attributes={Property(name="name", type=StringType)})
    member = Class(name="Member", attributes={
        Property(name="email", type=StringType), Property(name="phone", type=StringType),
        Property(name="password", type=StringType)})
    link = BinaryAssociation(name="team_members", ends={
        Property(name="team", type=team, multiplicity=Multiplicity(0, 1)),
        Property(name="members", type=member, multiplicity=Multiplicity(0, "*"))})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=DomainModel(name="Club", types={team, member}, associations={link}),
                     output_dir=str(backend)).generate()
    _write_login(backend, 2, _REGISTER_AS_MEMBER)
    _guard_members(backend)
    return str(tmp_path)


def _edit(workspace: str, name: str, change) -> None:
    path = os.path.join(workspace, "web_app", "backend", name)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    changed = change(text)
    assert changed != text, name
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(changed)


def _check(workspace: str, request: str = "", *, walled: bool = True) -> list[str]:
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    probe = collect_constructibility_report(workspace)
    assert bool((probe["backends"][0].get("auth") or {}).get("source")) is walled, probe
    return authorization_issues(workspace, probe["backends"], request)


def _said_of(issues) -> list[str]:
    return sorted(issue.removeprefix(f"{PREFIX} web_app/backend: ") for issue in issues)


_SIDE_DOORS = ["another account could change Task.project_id through POST /project/; "
               "PUT /task/{task_id}/ refused that account",
               "another account could change Task.project_id through PUT /project/{project_id}/; "
               "PUT /task/{task_id}/ refused that account"]


def test_an_owner_check_on_one_router_leaves_the_sibling_routers_side_door(tmp_path):
    """Account B is refused on A's task directly and re-parents it through Project's task list.
    Projects, which no route refuses to B, are shared and not named."""
    assert _said_of(_check(_login_scaffold(tmp_path))) == _SIDE_DOORS


def test_a_bulk_delete_left_beside_a_guarded_delete_is_named(tmp_path):
    issues = _check(_login_scaffold(tmp_path, own_bulk_delete=False))
    assert ("another account could delete a Task through DELETE /task/bulk/; DELETE /task/{task_id}/ refused "
            "that account") in _said_of(issues)


def test_delete_only_ownership_names_the_bulk_delete(tmp_path):
    """Any account edits any task; only its creator may delete it, but the generated bulk delete lets anyone."""
    workspace = _login_scaffold(tmp_path, own_bulk_delete=False)
    _edit(workspace, "routers/task.py", lambda text: re.sub(
        r"(async def update_task\(.*\n)    require_owner\(task_id, user\)\n", r"\1", text))
    assert _said_of(_check(workspace, "A shared board. Any member can edit any task; only the member who "
                                      "created a task can delete it.")) == [
        "another account could delete a Task through DELETE /task/bulk/; DELETE /task/{task_id}/ refused that "
        "account"]


def test_a_recollection_reports_the_same_findings(tmp_path):
    """Two runs over one tree must agree: no id or sampled value may be in a finding."""
    workspace = _login_scaffold(tmp_path)
    assert _check(workspace) == _check(workspace)


def test_a_login_wall_with_no_ownership_anywhere_yields_nothing(tmp_path):
    """Every account may edit every record: the other account's direct edit succeeds, and that is the design."""
    assert _check(_login_scaffold(tmp_path, own_tasks=False)) == []


def test_a_route_that_refuses_every_caller_is_no_rule(tmp_path):
    """Tasks are locked once created; moving one through its project is the designed feature."""
    workspace = _login_scaffold(tmp_path, own_tasks=False)
    _edit(workspace, "routers/task.py", lambda text: re.sub(
        r"(async def update_task\(.*\n)",
        r'\1    raise HTTPException(status_code=403, detail="Tasks are locked once created")\n', text))
    assert _check(workspace, "A shared board with login. Tasks cannot be edited once created; any member can "
                             "move a task to another project.") == []


def test_a_public_route_under_a_record_is_named(tmp_path):
    """A view counter anyone may call, beside a login on every update."""
    workspace = _login_scaffold(tmp_path, own_tasks=False)
    _edit(workspace, "main_api.py", lambda text: text + '''
VIEWS: dict = {}

@app.post("/task/{task_id}/view/")
def count_view(task_id: int):
    VIEWS[task_id] = VIEWS.get(task_id, 0) + 1
    return {"views": VIEWS[task_id]}
''')
    assert _said_of(_check(workspace)) == [
        "a caller who is not logged in could successfully send POST /task/{task_id}/view/"]


_UNGUARDED_ROUTES = '''

@router.patch("/task/{task_id}/", response_model=None, tags=["Task"])
async def patch_task(task_id: int, task_data: TaskCreate, database: Session = Depends(get_db)):
    db_task = database.query(Task).filter(Task.id == task_id).first()
    db_task.title = task_data.title
    database.commit()
    database.refresh(db_task)
    return db_task


@router.post("/task/{task_id}/rename/", response_model=None, tags=["Task"])
async def rename_task(task_id: int, task_data: TaskCreate, database: Session = Depends(get_db)):
    db_task = database.query(Task).filter(Task.id == task_id).first()
    db_task.title = task_data.title
    database.commit()
    database.refresh(db_task)
    return db_task
'''


def test_an_unguarded_second_update_route_is_named(tmp_path):
    """PUT checks the owner; an added PATCH and an added route under the record taking its body do not."""
    workspace = _login_scaffold(tmp_path)
    _edit(workspace, "routers/task.py", lambda text: text + _UNGUARDED_ROUTES)
    assert _said_of(_check(workspace)) == sorted(_SIDE_DOORS + [
        "another account could change Task.title through PATCH /task/{task_id}/; PUT /task/{task_id}/ refused "
        "that account",
        "another account could change Task.title through POST /task/{task_id}/rename/; PUT /task/{task_id}/ "
        "refused that account"])


def test_an_action_route_with_a_body_of_its_own_is_not_an_edit(tmp_path):
    """Any account may rate another's task, which updates the task's rating: what the route is for."""
    workspace = _login_scaffold(tmp_path, rating=True)
    # A new task starts unrated, whatever its create body says.
    _edit(workspace, "routers/task.py", lambda text: text.replace(
        "    database.add(db_task)\n    database.flush()\n",
        "    db_task.rating = 0\n    database.add(db_task)\n    database.flush()\n", 1))
    _edit(workspace, "main_api.py", lambda text: text + '''
from pydantic import BaseModel

RATINGS: dict = {}


class RatingIn(BaseModel):
    rating: int


@app.post("/task/{task_id}/rate/", tags=["Task"])
def rate_task(task_id: int, data: RatingIn, database: Session = Depends(get_db), user=Depends(get_current_user)):
    task = database.query(Task).filter(Task.id == task_id).first()
    RATINGS.setdefault(task_id, {})[user] = data.rating
    task.rating = sum(RATINGS[task_id].values()) // len(RATINGS[task_id])
    database.commit()
    return {"rating": task.rating}
''')
    assert _said_of(_check(workspace)) == _SIDE_DOORS


def _impersonation_guard_scaffold(tmp_path) -> str:
    """Any member edits any task, but a task's ``owner`` must be the member saving it."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, Multiplicity, Property, StringType,
    )
    from besser.generators.backend import BackendGenerator

    task = Class(name="Task", attributes={Property(name="title", type=StringType)})
    member = Class(name="Member", attributes={
        Property(name="email", type=StringType), Property(name="phone", type=StringType),
        Property(name="password", type=StringType)})
    owns = BinaryAssociation(name="member_tasks", ends={
        Property(name="owner", type=member, multiplicity=Multiplicity(1, 1)),
        Property(name="ownedTasks", type=task, multiplicity=Multiplicity(0, "*"))})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=DomainModel(name="Board", types={task, member}, associations={owns}),
                     output_dir=str(backend)).generate()
    _write_login(backend, 2, _REGISTER_AS_MEMBER)
    _guard_members(backend)
    router = backend / "routers" / "task.py"
    text, guarded = re.subn(
        r"(async def (?:create|update)_task\([^)]*Depends\(get_db\))\)\s*(->\s*\w+)?:\n",
        r"\1, user=Depends(get_current_user)) \2:\n"
        r'    if task_data.owner != user:\n        raise HTTPException(status_code=403, detail="Act as yourself")\n',
        router.read_text(encoding="utf-8"))
    assert guarded == 2
    router.write_text("from auth import get_current_user\n" + text, encoding="utf-8")
    return str(tmp_path)


def test_a_request_in_another_accounts_name_is_no_rule_about_the_record(tmp_path):
    """Account B's update names B as the owner, as a real member's would; the app accepts it, so no record
    rule exists and nothing is said about tasks."""
    issues = _check(_impersonation_guard_scaffold(tmp_path))
    assert not [issue for issue in issues if "Task" in issue], issues


_NESTED_TASK_CREATE = '''

@router.post("/project/{project_id}/tasks/", response_model=None, tags=["Task"])
async def add_task(project_id: int, task_data: TaskCreate, database: Session = Depends(get_db)):
    db_task = Task(title=task_data.title, project_id=project_id)
    database.add(db_task)
    database.commit()
    database.refresh(db_task)
    return db_task
'''


def test_a_nested_create_route_does_not_hide_the_entity(tmp_path):
    """An added POST /project/{project_id}/tasks/ takes the same body as the plain create; the check
    must still create its task through the plain one, or its findings vanish with the record."""
    workspace = _login_scaffold(tmp_path)
    _edit(workspace, "routers/task.py", lambda text: text + _NESTED_TASK_CREATE)
    assert _said_of(_check(workspace)) == _SIDE_DOORS



def test_a_record_account_a_cannot_create_is_named_as_not_covered(tmp_path):
    workspace = _login_scaffold(tmp_path)
    _edit(workspace, "routers/task.py", lambda text: re.sub(
        r"(async def create_task\(.*\n)",
        r'\1    raise HTTPException(status_code=409, detail="The board is full")\n', text))
    assert _check(workspace) == [
        f"{COVERAGE_PREFIX} web_app/backend: records of Task (POST /task/ answered 409) could not be set up for "
        "the check, so what other accounts can do to them was not checked"]


def test_a_link_route_under_a_protected_record_is_named(tmp_path):
    """Task *-* Tag: account B changes which of A's tags are on A's task through the generated link
    route. Attaching a record of its own instead is what such routes are for, and is not tried."""
    assert _said_of(_check(_login_scaffold(tmp_path, tags=True))) == sorted(_SIDE_DOORS + [
        "another account could change Task.tags through POST /task/{task_id}/tags/{tag_id}/; "
        "PUT /task/{task_id}/ refused that account"])


def test_an_app_with_public_creates_and_protected_edits_is_checked(tmp_path):
    """No create is refused, so the probe meets no login wall; the app still has accounts and an owner rule."""
    workspace = _login_scaffold(tmp_path)
    _edit(workspace, "auth.py", lambda text: text.replace(
        'if request.method == "GET":', 'if request.method == "GET" or (request.method == "POST" and credentials is None):'))
    said = _said_of(_check(workspace, walled=False))
    assert set(_SIDE_DOORS) <= set(said)
    assert any(line.startswith("a caller who is not logged in could successfully send POST /project/") for line in said)


def test_an_account_router_left_as_generated_is_reported(tmp_path):
    """The account is a modelled class whose own router nobody touched. Its update route is the
    constructibility probe's own ``open account writes:`` finding and is not repeated."""
    workspace = _login_scaffold(tmp_path, member_accounts=True)
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    probe = collect_constructibility_report(workspace)
    assert any(issue.startswith(f"{OPEN_ACCOUNT_PREFIX} web_app/backend: PUT /member/{{member_id}}/")
               for issue in probe["issues"])
    assert _said_of(authorization_issues(workspace, probe["backends"])) == sorted(_SIDE_DOORS + [
        "GET /member/, GET /member/paginated/, GET /member/search/, GET /member/{member_id}/ returned an "
        "account's `password` to other callers",
        "POST /project/, POST /task/ still accepted the token of an account that had deleted itself and could no "
        "longer log in",
        "a caller who is not logged in could read an account's email and phone through GET /member/, "
        "GET /member/paginated/, GET /member/search/, GET /member/{member_id}/",
        "another account could delete a Member account through DELETE /member/{member_id}/",
    ])


_USERNAME_LOGIN = '''
@router.post("/register", status_code=201)
def register(data: Credentials, database: Session = Depends(get_db)):
    member = Member(username=data.username, password=data.password)
    database.add(member)
    database.commit()
    database.refresh(member)
    return {"id": member.id}


def check(data, database):
    member = database.query(Member).filter(Member.username == data.username,
                                           Member.password == data.password).first()
    return member and member.id
'''


def test_a_password_shown_beside_a_username_account_is_named(tmp_path):
    """No e-mail anywhere: the account is recognised by the username it logs in with."""
    from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
    from besser.generators.backend import BackendGenerator

    member = Class(name="Member", attributes={Property(name="username", type=StringType),
                                              Property(name="password", type=StringType)})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=DomainModel(name="Board", types={member}), output_dir=str(backend)).generate()
    _write_login(backend, 1, _USERNAME_LOGIN)
    _edit(str(tmp_path), "auth.py", lambda text: text.replace(
        "    email: str\n    password: str\n    phone: str\n", "    username: str\n    password: str\n"))
    _guard_members(backend)
    assert ("GET /member/, GET /member/paginated/, GET /member/search/, GET /member/{member_id}/ returned an "
            "account's `password` to other callers") in _said_of(_check(str(tmp_path)))


def test_a_login_kept_in_a_secure_cookie_is_followed(tmp_path):
    """The login answers with a ``Secure`` session cookie and no token; the accounts must still be checked."""
    workspace = _login_scaffold(tmp_path)
    _edit(workspace, "auth.py", lambda text: text.replace(
        '    return {"access_token": token, "token_type": "bearer"}',
        '    from fastapi.responses import JSONResponse\n'
        '    response = JSONResponse({"ok": True})\n'
        '    response.set_cookie("session", token, httponly=True, secure=True, samesite="lax")\n'
        '    return response').replace(
        "    if credentials is None or credentials.credentials not in TOKENS:\n"
        '        raise HTTPException(status_code=401, detail="Not authenticated")\n'
        "    return TOKENS[credentials.credentials]",
        "    if request.cookies.get('session') not in TOKENS:\n"
        '        raise HTTPException(status_code=401, detail="Not authenticated")\n'
        "    return TOKENS[request.cookies['session']]"))
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    assert _said_of(authorization_issues(workspace, collect_constructibility_report(workspace)["backends"])) \
        == _SIDE_DOORS


def test_a_team_built_from_other_members_is_named_whatever_the_request_says(tmp_path):
    """A member may change only its own row, and a team's member list moves other members into it. The
    request may ask for exactly that; the warning states it and the user decides."""
    workspace = _club_scaffold(tmp_path)
    links = ["another account could change Member.team_id through POST /team/; PUT /member/{member_id}/ refused "
             "that account",
             "another account could change Member.team_id through PUT /team/{team_id}/; PUT /member/{member_id}/ "
             "refused that account"]
    for request in ("A club app with login.", "Any member can create a team and add other members to it."):
        assert [line for line in _said_of(_check(workspace, request)) if "team_id" in line] == links


def test_the_create_map_differs_from_the_probes_only_where_documented(tmp_path):
    """Both probes derive the app's create routes from its OpenAPI document and its ORM, abstract classes
    left out. Where an entity also has a nested create route, this check creates through the plain one and
    the constructibility probe through the last one declared."""
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, Class, DomainModel, Generalization, Multiplicity, Property, StringType,
    )
    from besser.generators.backend import BackendGenerator
    from besser.spec_driven_agent.validation.constructibility import collect_constructibility_report

    shape = Class(name="Shape", is_abstract=True, attributes={Property(name="label", type=StringType)})
    circle = Class(name="Circle", attributes={Property(name="colour", type=StringType)})
    layer = Class(name="Layer", attributes={Property(name="name", type=StringType)})
    model = DomainModel(name="Drawing", types={shape, circle, layer}, generalizations={Generalization(shape, circle)},
                        associations={BinaryAssociation(name="layer_circles", ends={
                            Property(name="layer", type=layer, multiplicity=Multiplicity(1, 1)),
                            Property(name="circles", type=circle, multiplicity=Multiplicity(0, "*"))})})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=model, output_dir=str(backend)).generate()
    _write_login(backend, 3, _REGISTER_IN_MEMORY)
    router = backend / "routers" / "circle.py"
    router.write_text(router.read_text(encoding="utf-8") + '''

@router.post("/layer/{layer_id}/new-circle/", response_model=None, tags=["Circle"])
async def add_circle(layer_id: int, circle_data: CircleCreate, database: Session = Depends(get_db)):
    db_circle = Circle(colour=circle_data.colour, label=circle_data.label, layer_id=layer_id)
    database.add(db_circle)
    database.commit()
    database.refresh(db_circle)
    return db_circle
''', encoding="utf-8")

    [probe] = collect_constructibility_report(str(tmp_path), model)["backends"]
    # The abstract class is not created, whether its router declares a create route or not.
    assert "Shape" not in probe["entities"] and probe["auth"]["source"]
    report = authz._run_child(str(backend), workspace=str(tmp_path), abstract=["Shape"])
    probe_creates = {name: entry["path"] for name, entry in probe["entities"].items()}
    assert probe_creates.pop("Circle") == "/layer/{layer_id}/new-circle/"
    assert report["creates"].pop("Circle") == "/circle/"
    assert report["creates"] == probe_creates
    assert "Circle" in report["created"]


class _Usage:
    estimated_cost = 0.0

    def summary(self) -> dict:
        return {"api_calls": 0, "cost_usd": 0.0}


class _Client:
    model = "test-model"
    usage = _Usage()
    max_tokens = 4096

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        return {"stop_reason": "end_turn", "content": []}


def _orchestrator(workspace: str):
    from besser.BUML.metamodel.structural import Class, DomainModel
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="Board", types={Class(name="Task")}),
        output_dir=workspace, enable_tracing=False, enable_checkpointing=False, enable_toolchain_validation=False)


def _sweep(orchestrator) -> tuple[list, tuple, bool]:
    """The Phase 3 sweep: its authorization findings, the tree score and whether completion is verified."""
    issues = orchestrator._collect_validation_issues()
    blockers = [issue for issue in issues if issue.severity == "blocker"]
    found = [issue for issue in issues if issue.message.startswith(("authorization", "open account"))]
    return found, orchestrator._phase3_tree_score(blockers), not any(map(is_completion_issue, issues))


def test_the_phase3_sweep_reports_the_findings_and_the_agent_never_sees_them(tmp_path):
    orchestrator = _orchestrator(_login_scaffold(tmp_path))
    found, _score, _verified = _sweep(orchestrator)
    assert [issue.severity for issue in found] == ["warning", "warning"]
    assert _said_of(issue.message for issue in found) == _SIDE_DOORS
    assert not [issue for issue in orchestrator._validate_app()["issues"] if "authorization" in issue]


def test_the_check_changes_no_score_and_no_verdict(tmp_path):
    """Removing the app's owner check makes the warnings go away and changes nothing else: the same score
    and the same verified completion, so a repair has nothing to gain from it."""
    workspace = _login_scaffold(tmp_path)
    orchestrator = _orchestrator(workspace)
    found, score, verified = _sweep(orchestrator)
    assert len(found) == 2

    _edit(workspace, "routers/task.py", lambda text: text.replace("    require_owner(task_id, user)\n", ""))
    assert _sweep(orchestrator) == ([], score, verified)


def test_a_sign_up_limit_reads_as_not_covered_and_changes_nothing_else(tmp_path):
    workspace = _login_scaffold(tmp_path)
    orchestrator = _orchestrator(workspace)
    _found, score, verified = _sweep(orchestrator)
    _edit(workspace, "auth.py", lambda text: text.replace(
        "    ACCOUNTS[data.email] = data.password\n",
        '    if len(ACCOUNTS) >= 2:\n        raise HTTPException(status_code=429, detail="Too many sign-ups")\n'
        "    ACCOUNTS[data.email] = data.password\n"))
    found, *rest = _sweep(orchestrator)
    assert rest == [score, verified]
    [note] = found
    assert note.severity == "warning" and note.message.startswith(
        f"{COVERAGE_PREFIX} web_app/backend: two accounts could not be registered and logged in")


def test_the_sweep_reports_an_account_route_once(tmp_path):
    """The probe's own account check and this one both see PUT /member/{member_id}/."""
    orchestrator = _orchestrator(_login_scaffold(tmp_path, member_accounts=True))
    found, _score, _verified = _sweep(orchestrator)
    named = [issue.message for issue in found if "PUT /member/{member_id}/" in issue.message]
    assert len(named) == 1 and named[0].startswith(OPEN_ACCOUNT_PREFIX), named
