"""Provider-neutral authenticated principals for backend resources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Request

from besser.utilities.web_modeling_editor.backend.services.deployment.github_oauth import (
    github_session_candidates,
    get_user_session,
)


LOCAL_PRINCIPAL_SUBJECT = "local:anonymous"
_FALSE_VALUES = {"0", "false", "no", "off"}
_PRODUCTION_VALUES = {"prod", "production"}
_ENVIRONMENT_VARIABLES = ("BESSER_ENV", "APP_ENV", "ENVIRONMENT", "ENV")


@dataclass(frozen=True)
class Principal:
    """Authenticated identity used for authorization decisions.

    ``subject`` is the only value persisted with resources. Provider
    details remain at this boundary so SmartGen can support another
    identity provider without changing its run model.
    """

    subject: str
    provider: str
    display_name: Optional[str] = None


def smart_gen_auth_required() -> bool:
    """Return whether SmartGen requests must authenticate.

    Production can never opt out. The default is fail-closed; local and test
    environments can explicitly use the local principal with
    ``BESSER_SMART_GEN_AUTH_REQUIRED=false``.
    """
    is_production = any(
        os.environ.get(name, "").strip().lower() in _PRODUCTION_VALUES
        for name in _ENVIRONMENT_VARIABLES
    )
    if is_production:
        return True
    configured = os.environ.get("BESSER_SMART_GEN_AUTH_REQUIRED")
    if configured is None:
        # Fail closed when a deployment has not labelled its environment.
        # Local development and tests opt out explicitly.
        return True
    normalized = configured.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    # True values and invalid values both fail closed.
    return True


def principal_from_github_session(session_id: str) -> Optional[Principal]:
    """Resolve a GitHub-backed principal from the existing session store."""
    session = get_user_session(session_id)
    if not session:
        return None

    github_user_id = session.get("github_user_id")
    username = session.get("username")
    if (
        isinstance(github_user_id, int)
        and not isinstance(github_user_id, bool)
        and github_user_id > 0
    ):
        subject = f"github:{github_user_id}"
    elif isinstance(username, str) and username.strip():
        # Compatibility for sessions created before numeric IDs were stored.
        subject = f"github-login:{username.strip().casefold()}"
    else:
        return None

    return Principal(
        subject=subject,
        provider="github",
        display_name=username if isinstance(username, str) else None,
    )


async def get_current_principal(
    request: Request,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
) -> Principal:
    """FastAPI dependency returning the caller's authorization principal."""
    for session_id in github_session_candidates(
        request,
        header_session=github_session,
    ):
        principal = principal_from_github_session(session_id)
        if principal is not None:
            return principal

    if smart_gen_auth_required():
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in and try again.",
        )

    return Principal(subject=LOCAL_PRINCIPAL_SUBJECT, provider="local")
