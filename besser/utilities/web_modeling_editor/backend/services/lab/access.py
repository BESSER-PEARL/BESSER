"""Who may use BESSER Lab: the GitHub logins listed in ``BESSER_LAB_ALLOWED_LOGINS``, and who among them is an
admin (``BESSER_LAB_ADMIN_LOGINS``)."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Optional

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


def allowed_logins(settings: BackendSettings) -> frozenset[str]:
    """The allowed GitHub logins, lower-cased (GitHub logins ignore case)."""
    return _logins(settings.lab_allowed_logins)


@functools.lru_cache(maxsize=8)
def _admin_logins(admins_value: str, allowed_value: str) -> frozenset[str]:
    """The admins who are also allowed; cached per configuration, so the warning is logged once."""
    admins, allowed = _logins(admins_value), _logins(allowed_value)
    ignored = sorted(admins - allowed)
    if ignored:
        logger.warning("BESSER_LAB_ADMIN_LOGINS lists %s, not on BESSER_LAB_ALLOWED_LOGINS; ignored as admins",
                       ", ".join(ignored))
    return admins & allowed


def is_admin(settings: BackendSettings, login: Optional[str]) -> bool:
    """True when ``login`` is allowed on this lab and listed in ``BESSER_LAB_ADMIN_LOGINS``.

    An admin may drop any change and stop any run. An admin not on the allowed list is ignored, with a warning
    logged once per configuration.
    """
    if not is_login_allowed(settings, login):
        return False
    return login.strip().lower() in _admin_logins(settings.lab_admin_logins, settings.lab_allowed_logins)


def is_login_allowed(settings: BackendSettings, login: Optional[str]) -> bool:
    """True when the lab is on and ``login`` is on its list; False with the lab off or an empty list."""
    if not settings.lab_enabled or not login:
        return False
    return login.strip().lower() in allowed_logins(settings)
