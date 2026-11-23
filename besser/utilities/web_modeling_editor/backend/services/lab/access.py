"""Who may use BESSER Lab: the GitHub logins listed in ``BESSER_LAB_ALLOWED_LOGINS`` and the members of the
organization ``BESSER_LAB_ALLOWED_ORG``, and who among them is an admin (``BESSER_LAB_ADMIN_LOGINS``)."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional

from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabUser:
    """A signed-in GitHub user who is allowed to use the lab."""

    login: str
    session_id: str
    # Kept out of the repr so that logging a user never logs the token.
    access_token: str = field(repr=False)


def _logins(value: str) -> frozenset[str]:
    """A comma-separated list of logins, trimmed and lower-cased (GitHub logins ignore case), blanks skipped."""
    return frozenset(name.strip().lower() for name in value.split(",") if name.strip())


#: Key of a sign-in session: the organization (lower-cased) the user was a member of when signing in.
SESSION_ORG_KEY = "lab_org"


def allowed_org(settings: BackendSettings) -> str:
    """The organization whose members may use the lab, lower-cased; empty when only listed logins may."""
    return settings.lab_allowed_org.strip().lower()


def is_session_allowed(settings: BackendSettings, session: Optional[Mapping[str, object]]) -> bool:
    """True when the session's login is on the lab's list, or the session was verified at sign-in as a member
    of ``BESSER_LAB_ALLOWED_ORG`` (:data:`SESSION_ORG_KEY`). A session verified for another organization, kept
    from before the setting changed, does not pass."""
    if not settings.lab_enabled or not session:
        return False
    login = session.get("username")
    if is_login_allowed(settings, login if isinstance(login, str) else None):
        return True
    org = allowed_org(settings)
    return bool(org) and session.get(SESSION_ORG_KEY) == org


def allowed_logins(settings: BackendSettings) -> frozenset[str]:
    """The allowed GitHub logins, lower-cased (GitHub logins ignore case)."""
    return _logins(settings.lab_allowed_logins)


@functools.lru_cache(maxsize=8)
def _admin_logins(admins_value: str, allowed_value: str, org: str = "") -> frozenset[str]:
    """The admins who are also allowed; cached per configuration, so the warning is logged once. With an
    allowed organization an admin need not be on the list: its routes are reached only by a member."""
    admins, allowed = _logins(admins_value), _logins(allowed_value)
    if org:
        return admins
    ignored = sorted(admins - allowed)
    if ignored:
        logger.warning("BESSER_LAB_ADMIN_LOGINS lists %s, not on BESSER_LAB_ALLOWED_LOGINS; ignored as admins",
                       ", ".join(ignored))
    return admins & allowed


def listed_admin_logins(settings: BackendSettings) -> list[str]:
    """The logins of ``BESSER_LAB_ADMIN_LOGINS``, lower-cased and sorted; :func:`is_admin` says which count."""
    return sorted(_logins(settings.lab_admin_logins))


def is_admin(settings: BackendSettings, login: Optional[str]) -> bool:
    """True when ``login`` is allowed on this lab and listed in ``BESSER_LAB_ADMIN_LOGINS``.

    An admin may drop any change and stop any run. An admin not on the allowed list is ignored, with a warning
    logged once per configuration.
    """
    org = allowed_org(settings)
    if not settings.lab_enabled or not login or not (org or is_login_allowed(settings, login)):
        return False
    return login.strip().lower() in _admin_logins(settings.lab_admin_logins, settings.lab_allowed_logins, org)


def is_login_allowed(settings: BackendSettings, login: Optional[str]) -> bool:
    """True when the lab is on and ``login`` is on its list; False with the lab off or an empty list."""
    if not settings.lab_enabled or not login:
        return False
    return login.strip().lower() in allowed_logins(settings)
