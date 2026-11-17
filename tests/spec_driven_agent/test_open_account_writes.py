"""Account records that anyone can change, in an app with its own login.

Live runs r9-p81 and r11-p81 (risk-awareness app, "include user authentication"):
``/auth/register`` and ``/auth/login`` issue bearer tokens, yet the generic
``PUT /appuser/{appuser_id}/`` and ``DELETE /appuser/{appuser_id}/`` answer 200
to a request with no credentials, so any visitor can rename or delete any
account. No check said so. The probe now asks it of the account table (the one
the register route creates), and only answers 2xx with no credentials count.
An open ``POST`` is self-registration and is never listed.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.constructibility import (
    OPEN_ACCOUNT_PREFIX, _issues_from_report, collect_constructibility_report,
)
from besser.spec_driven_agent.validation.issues import _classify_issue

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

_AUTH_MODULE = '''
import secrets
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from sql_alchemy import Member

_tokens: set = set()
bearer = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/auth")


class Credentials(BaseModel):
    username: str
    password: str


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if credentials is None or credentials.credentials not in _tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return credentials.credentials


def header_user(authorization: str = Header(...)):
    if authorization.removeprefix("Bearer ") not in _tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return authorization


def writes_need_admin(request: Request, user: str = Depends(get_current_user)):
    if request.method in ("PUT", "PATCH", "DELETE"):
        raise HTTPException(status_code=403, detail="Admins only")
    return user


@router.post("/register", status_code=201)
def register(data: Credentials, database: Session = Depends(get_db)):
    member = Member(username=data.username, password=data.password, role="member")
    database.add(member)
    database.commit()
    return {"id": member.id, "username": member.username}


@router.post("/login")
def login(data: Credentials, database: Session = Depends(get_db)):
    member = database.query(Member).filter(Member.username == data.username).first()
    if member is None or member.password != data.password:
        raise HTTPException(status_code=401, detail="bad credentials")
    token = secrets.token_urlsafe(16)
    _tokens.add(token)
    return {"access_token": token, "token_type": "bearer"}
'''


def _app(tmp_path, *, member_guard: str | None, with_auth: bool = True) -> str:
    """Member (the account table) and Note; ``member_guard`` names the dependency
    put on the member router, None for none."""
    from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
    from besser.generators.backend import BackendGenerator

    member = Class(name="Member", attributes={Property(name="username", type=StringType),
                                              Property(name="password", type=StringType),
                                              Property(name="role", type=StringType)})
    note = Class(name="Note", attributes={Property(name="text", type=StringType)})
    backend = tmp_path / "web_app" / "backend"
    BackendGenerator(model=DomainModel(name="Club", types={member, note}), output_dir=str(backend)).generate()
    if not with_auth:
        return str(tmp_path)
    (backend / "auth.py").write_text(_AUTH_MODULE, encoding="utf-8")
    main = backend / "main_api.py"
    text = main.read_text(encoding="utf-8").replace(
        "app.include_router(note_router.router)",
        "app.include_router(note_router.router, dependencies=[Depends(get_current_user)])", 1)
    if member_guard:
        text = text.replace("app.include_router(member_router.router)",
                            f"app.include_router(member_router.router, dependencies=[Depends({member_guard})])", 1)
    head = "from fastapi import Depends\nimport auth as _auth\nfrom auth import get_current_user, header_user\n"
    main.write_text(head + text + "\napp.include_router(_auth.router)\n", encoding="utf-8")
    return str(tmp_path)


def test_account_writes_open_beside_a_login_are_a_warning_naming_them(tmp_path):
    report = collect_constructibility_report(_app(tmp_path, member_guard=None))

    [finding] = [issue for issue in report["issues"] if issue.startswith(OPEN_ACCOUNT_PREFIX)]
    assert "PUT /member/{member_id}/" in finding and "DELETE /member/{member_id}/" in finding, finding
    assert "POST /auth/login" in finding and "Member" in finding
    # Self-registration: an open create is never listed.
    assert "POST /member/" not in finding
    assert _classify_issue(finding).severity == "warning"


def test_a_logged_in_account_changing_another_ones_password_and_role_is_flagged(tmp_path):
    """Live run r12-p61: ``password`` and ``role`` on ``UserCreate``, so the generic
    ``PUT /user/{id}/`` behind login let any account set another's password and role."""
    report = collect_constructibility_report(_app(tmp_path, member_guard="get_current_user"))

    [finding] = [issue for issue in report["issues"] if issue.startswith(OPEN_ACCOUNT_PREFIX)]
    assert finding == (f"{OPEN_ACCOUNT_PREFIX} web_app/backend: PUT /member/{{member_id}}/ let one logged-in "
                       "account change password, role of another Member record"), finding
    assert _classify_issue(finding).severity == "warning"


def test_writes_only_an_admin_may_make_are_not_flagged(tmp_path):
    report = collect_constructibility_report(_app(tmp_path, member_guard="writes_need_admin"))
    assert not [issue for issue in report["issues"] if issue.startswith(OPEN_ACCOUNT_PREFIX)], report["issues"]


def test_a_guard_that_refuses_with_422_is_not_flagged(tmp_path):
    # ``Header(...)`` missing answers 422 before the guard body runs: not a write that got through.
    report = collect_constructibility_report(_app(tmp_path, member_guard="header_user"))
    assert not [issue for issue in report["issues"] if "no credentials" in issue], report["issues"]


def test_an_app_without_login_is_never_flagged(tmp_path):
    report = collect_constructibility_report(_app(tmp_path, member_guard=None, with_auth=False))
    assert report["issues"] == [], report["issues"]
    assert "open_account_writes" not in report["backends"][0]


def test_the_finding_is_rendered_from_the_report():
    report = {"boot": "ok", "entities": {}, "open_account_writes": [
        {"table": "AppUser", "login": "/auth/login",
         "routes": ["PUT /appuser/{appuser_id}/", "DELETE /appuser/{appuser_id}/"]}]}
    [finding] = _issues_from_report(report, "backend")
    assert finding.startswith(f"{OPEN_ACCOUNT_PREFIX} backend: PUT /appuser/{{appuser_id}}/, "
                              "DELETE /appuser/{appuser_id}/"), finding
