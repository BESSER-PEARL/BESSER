"""The account-write step must never cost the probe its other results.

Review of round 10: a register route that raises a plain exception on a taken
username (no handler, so ASGITransport re-raises it) crashed the new step when it
registered a second time. The whole child report became ``boot: probe_error``,
every create verdict was lost, and a real create blocker hid behind a warning.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from besser.spec_driven_agent.validation.constructibility import PREFIX, collect_constructibility_report

from tests.spec_driven_agent.test_open_account_writes import _AUTH_MODULE, _app

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# Tokens map to member ids, so a guard can tell an admin and the account itself.
_AUTH = _AUTH_MODULE.replace("_tokens: set = set()", "_tokens: dict = {}").replace(
    "_tokens.add(token)", "_tokens[token] = member.id") + '''

def current_member(credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                   database: Session = Depends(get_db)):
    if credentials is None or credentials.credentials not in _tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return database.get(Member, _tokens[credentials.credentials])


def admin_or_self(request: Request, me=Depends(current_member)):
    if request.method in ("PUT", "PATCH", "DELETE"):
        target = request.path_params.get("member_id")
        if me.role != "admin" and str(me.id) != str(target):
            raise HTTPException(status_code=403, detail="Only an admin or the account itself")
    return me
'''


def variant_app(tmp_path, variant: str) -> str:
    """The open-account fixture app, changed as the reviewer's ``fp_apps.py`` changes it."""
    guard = "admin_or_self" if variant in ("first_admin", "self_only") else "get_current_user"
    root = _app(tmp_path, member_guard=guard)
    backend = Path(root) / "web_app" / "backend"
    auth = _AUTH
    if variant == "first_admin":
        auth = auth.replace('role="member")', 'role="admin" if database.query(Member).count() == 0 else "member")')
    if variant == "single_account":  # a second registration is an unhandled error
        auth = auth.replace("    member = Member(username=data.username",
                            "    if database.query(Member).count():\n"
                            "        raise RuntimeError('one account only')\n"
                            "    member = Member(username=data.username", 1)
    if variant == "dup_crash":
        auth = auth.replace("    member = Member(username=data.username",
                            "    if database.query(Member).filter(Member.username == data.username).first():\n"
                            "        raise LookupError('username taken')\n"
                            "    member = Member(username=data.username", 1)
    (backend / "auth.py").write_text(auth, encoding="utf-8")
    main = backend / "main_api.py"
    main.write_text(main.read_text(encoding="utf-8").replace(
        "from auth import get_current_user, header_user",
        "from auth import admin_or_self, get_current_user, header_user"), encoding="utf-8")
    if variant == "echo":  # answers with the body it was sent, storing only the username
        router = backend / "routers" / "member.py"
        text = router.read_text(encoding="utf-8")
        text = text.replace("    setattr(db_member, 'password', member_data.password)\n"
                            "    setattr(db_member, 'role', member_data.role)\n", "")
        text = text.replace("    database.refresh(db_member)\n\n    return db_member\n",
                            "    return {**member_data.model_dump(), 'id': member_id}\n", 1)
        router.write_text(text, encoding="utf-8")
    return root


def test_an_app_exception_in_the_account_step_keeps_the_create_verdicts(tmp_path):
    root = variant_app(tmp_path, "dup_crash")
    # A real create blocker the probe must still report.
    note = Path(root) / "web_app" / "backend" / "routers" / "note.py"
    text = note.read_text(encoding="utf-8")
    head = text[text.index("async def create_note("):]
    head = head[:head.index("\n") + 1]
    note.write_text(text.replace(head, head + "    raise RuntimeError('note storage is broken')\n", 1),
                    encoding="utf-8")

    report = collect_constructibility_report(root)

    [backend] = report["backends"]
    assert backend["boot"] == "ok", backend
    assert backend["entities"]["Member"]["verdict"] == "created"
    assert backend["entities"]["Note"]["verdict"] == "crashed"
    assert [issue for issue in report["issues"] if issue.startswith(PREFIX) and "POST /note/" in issue]


def test_a_step_that_cannot_finish_is_recorded_as_not_checked(tmp_path):
    report = collect_constructibility_report(variant_app(tmp_path, "single_account"))

    [backend] = report["backends"]
    assert backend["boot"] == "ok", backend
    assert {entry["verdict"] for entry in backend["entities"].values()} == {"created"}
    assert "RuntimeError: one account only" in backend["account_writes_not_checked"], backend
