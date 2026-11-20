"""Who may use BESSER Lab: the GitHub logins listed in ``BESSER_LAB_ALLOWED_LOGINS``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from besser.utilities.web_modeling_editor.backend.settings import BackendSettings


@dataclass(frozen=True)
class LabUser:
    """A signed-in GitHub user who is allowed to use the lab."""

    login: str
    session_id: str
    # Kept out of the repr so that logging a user never logs the token.
    access_token: str = field(repr=False)


def allowed_logins(settings: BackendSettings) -> frozenset[str]:
    """The allowed GitHub logins, lower-cased (GitHub logins ignore case)."""
    return frozenset(
        name.strip().lower() for name in settings.lab_allowed_logins.split(",") if name.strip())


def is_login_allowed(settings: BackendSettings, login: Optional[str]) -> bool:
    """True when the lab is on and ``login`` is on its list; False with the lab off or an empty list."""
    if not settings.lab_enabled or not login:
        return False
    return login.strip().lower() in allowed_logins(settings)
