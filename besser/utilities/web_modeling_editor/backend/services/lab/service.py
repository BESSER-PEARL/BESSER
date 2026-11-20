"""BESSER Lab's service, between phase 1 and phase 2.

Phase 1's single-working-copy service is gone; the phase 2 service (changes with their own working trees,
runs, ledgers, preview pool, memory) is rebuilt in this module by build package W4-A. Until then
:func:`create_lab_service` builds nothing and every lab route answers 501. :class:`LabRefusal` is final.
"""

from __future__ import annotations

import logging
from typing import Optional

from besser.utilities.web_modeling_editor.backend.settings import BackendSettings

logger = logging.getLogger(__name__)


class LabRefusal(Exception):
    """A lab action refused for a reason the user can act on.

    The route answers ``status`` with ``{"detail": {"code": code, "message": message}}``
    (``models.lab.LabRefusalResponse``).

    Args:
        code: One of ``models.lab.LabRefusalCode``.
        message: Written for the user; holds no secret.
        status: The HTTP status of the refusal.
    """

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class LabService:
    """Placeholder of the lab service of a lab instance; build package W4-A rebuilds it here."""


def create_lab_service(settings: BackendSettings) -> Optional[LabService]:
    """The lab service of a lab instance: None until the phase 2 service is built, so the lab routes answer 501."""
    logger.warning("BESSER Lab is being rebuilt; its routes answer 501")
    return None
