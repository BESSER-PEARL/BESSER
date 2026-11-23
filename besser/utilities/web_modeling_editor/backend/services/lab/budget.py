"""BESSER Lab's monthly caps, per login and for the whole lab.

Spent is this UTC month's ``run`` lines of the audit log (``audit.py``). A run starts by reserving its cost cap,
``min(per-run cap, what the user has left, what the lab has left)``; reservations count against both caps until
the run settles, so runs at once cannot overshoot a cap.
"""

from __future__ import annotations

import calendar
import logging
import threading

from besser.utilities.web_modeling_editor.backend.models.lab import LabMeter, LabMeters
from besser.utilities.web_modeling_editor.backend.services.lab.audit import AuditLog
from besser.utilities.web_modeling_editor.backend.services.lab.service import LabRefusal

logger = logging.getLogger(__name__)

#: A run with less to spend buys no useful turn, so it is refused instead of started.
MIN_RUN_BUDGET_USD = 0.25
_CAP_STATUS = 409


class LabBudget:
    """Reserves each run's cost cap against the monthly caps.

    Args:
        audit: The audit log whose ``run`` lines are this month's spending.
        per_run_usd: The cost cap of one run (``BESSER_LAB_MAX_COST_USD``).
        user_monthly_usd: What one login's runs may cost per UTC month.
        instance_monthly_usd: What all runs together may cost per UTC month.
    """

    def __init__(self, audit: AuditLog, *, per_run_usd: float, user_monthly_usd: float,
                 instance_monthly_usd: float):
        self._audit = audit
        self._per_run = per_run_usd
        self._user_cap = user_monthly_usd
        self._instance_cap = instance_monthly_usd
        self._lock = threading.Lock()
        self._reserved: dict[str, float] = {}

    @property
    def per_run_usd(self) -> float:
        """The cost cap of one run; a run on the user's own key gets it without a reservation."""
        return self._per_run

    def reserve(self, login: str) -> float:
        """Reserve the cost cap of a run of ``login`` and return it.

        Raises:
            LabRefusal: ``user_cap`` or ``instance_cap`` (409) when less than ``MIN_RUN_BUDGET_USD`` is left.
        """
        with self._lock:
            user_spent, user_held = self._audit.spent(login), self._reserved.get(login, 0.0)
            lab_spent, lab_held = self._audit.spent(), sum(self._reserved.values())
            user_left = self._user_cap - user_spent - user_held
            lab_left = self._instance_cap - lab_spent - lab_held
            if user_left < MIN_RUN_BUDGET_USD:
                raise LabRefusal("user_cap", self._user_cap_message(user_spent, user_held), _CAP_STATUS)
            if lab_left < MIN_RUN_BUDGET_USD:
                raise LabRefusal("instance_cap", self._instance_cap_message(lab_spent, lab_held), _CAP_STATUS)
            cap = min(self._per_run, user_left, lab_left)
            self._reserved[login] = user_held + cap
            return cap

    def settle(self, login: str, reserved: float, spent: float) -> None:
        """Release a finished run's reservation.

        Call it after the run's ``run`` line is in the audit log: between the two its cost counts twice (as spent
        and as held), never zero times.
        A run that cost more than it reserved (its last turn crossed the cap) is logged as a warning.
        """
        with self._lock:
            left = self._reserved.get(login, 0.0) - reserved
            if left > 1e-9:
                self._reserved[login] = left
            else:
                self._reserved.pop(login, None)
        if spent > reserved:
            logger.warning("A lab run of %s cost $%.2f, over its reserved $%.2f", login, spent, reserved)

    def meters(self, login: str) -> LabMeters:
        """This month's spending, reservations and caps of ``login`` and of the whole lab."""
        with self._lock:
            user_held, lab_held = self._reserved.get(login, 0.0), sum(self._reserved.values())
        return LabMeters(
            month=self._audit.current_month(),
            user=LabMeter(spentUsd=self._audit.spent(login), reservedUsd=user_held, capUsd=self._user_cap),
            instance=LabMeter(spentUsd=self._audit.spent(), reservedUsd=lab_held, capUsd=self._instance_cap),
        )

    def _user_cap_message(self, spent: float, held: float) -> str:
        held_note = f" (${held:.2f} more is held by your runs in progress)" if held else ""
        return (f"You have used ${spent:.2f} of your ${self._user_cap:.2f} this month{held_note}. "
                f"Runs start again on {self._next_month()}.")

    def _instance_cap_message(self, spent: float, held: float) -> str:
        held_note = f" (${held:.2f} more is held by runs in progress)" if held else ""
        return (f"The lab has used ${spent:.2f} of its ${self._instance_cap:.2f} for this month{held_note}. "
                f"Runs start again on {self._next_month()}.")

    def _next_month(self) -> str:
        """The first day of the next UTC month, e.g. ``1 December``."""
        month = int(self._audit.current_month().split("-")[1])
        return f"1 {calendar.month_name[month % 12 + 1]}"
