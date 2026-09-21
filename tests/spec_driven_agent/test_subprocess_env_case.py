"""The subprocess allowlist has to survive Windows case folding.

``os.environ`` looks names up case-insensitively on Windows but ITERATES them
upper-cased. ``_safe_subprocess_env`` iterates, so an allowlist entry spelled
``ProgramFiles`` was compared against ``PROGRAMFILES`` and never matched --
every LLM-invoked subprocess on Windows ran without it. ``SystemRoot`` was
listed twice, in both spellings, which is the fingerprint of someone hitting
this once and patching the symptom.

It is a quiet failure: npm's global prefix, JDK and MSVC wrappers, and several
installers expand %ProgramFiles%, so the breakage surfaces to the model as a
tool error inside the GENERATED app, which it then tries to fix in that app's
source.
"""
import os

import pytest

from besser.spec_driven_agent.execution.process import (
    _SAFE_ENV_ALLOWLIST,
    _SECRET_SUBSTRINGS,
    _safe_subprocess_env,
)


def _upper(names):
    return {n.upper() for n in names}


def test_every_allowlist_entry_can_actually_match(monkeypatch):
    """The regression. Each entry is exported in the casing Windows reports,
    and must survive -- matching has to be case-insensitive, not a list of
    spellings someone remembered to add."""
    monkeypatch.setattr(os, "environ", {n.upper(): "x" for n in _SAFE_ENV_ALLOWLIST})

    kept = _upper(_safe_subprocess_env())

    missing = sorted(_upper(_SAFE_ENV_ALLOWLIST) - kept)
    assert not missing, f"allowlisted but dropped when upper-cased: {missing}"


@pytest.mark.parametrize("spelling", ["ProgramFiles", "PROGRAMFILES", "programfiles"])
def test_a_variable_survives_whatever_casing_the_os_reports(monkeypatch, spelling):
    monkeypatch.setattr(os, "environ", {spelling: r"C:\Program Files"})

    assert _upper(_safe_subprocess_env()) >= {"PROGRAMFILES"}


def test_secrets_are_still_stripped_in_any_casing(monkeypatch):
    """Case-folding the allowlist must not case-fold the deny rule open."""
    monkeypatch.setattr(os, "environ", {
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "sk-secret",
        "aws_secret_access_key": "secret",
        "GitHub_Token": "ghp_x",
        "Session_Cookie": "c",
    })

    kept = _upper(_safe_subprocess_env())

    assert "PATH" in kept
    for banned in ("ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "SESSION_COOKIE"):
        assert banned not in kept, f"{banned} reached an LLM-invoked subprocess"


def test_nothing_outside_the_allowlist_gets_through(monkeypatch):
    """Deny-by-default is the real control; the substring list is a backstop."""
    monkeypatch.setattr(os, "environ", {
        "PATH": "/usr/bin",
        "DATABASE_URL": "postgres://u:p@h/db",
        "SMTP_PASSWORD_FILE": "/run/x",
        "KUBERNETES_SERVICE_HOST": "10.0.0.1",
    })

    kept = _upper(_safe_subprocess_env())

    assert kept - {"PATH", "PYTHONDONTWRITEBYTECODE"} == set()


def test_the_deny_list_is_not_dead_weight():
    """_SECRET_SUBSTRINGS only ever sees allowlisted names, so today it can
    never fire. Kept as a backstop for a future allowlist entry -- this pins
    that it would actually catch one rather than reading as false assurance."""
    assert not [n for n in _SAFE_ENV_ALLOWLIST
                if any(s in n.upper() for s in _SECRET_SUBSTRINGS)], (
        "an allowlist entry now looks secret-shaped; decide which rule wins"
    )
    assert any(s in "NPM_AUTH_TOKEN" for s in _SECRET_SUBSTRINGS)
