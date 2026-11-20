"""A privilege change counts only when a read of the record shows it.

Review of round 10: the step compared the PUT's own response, so a handler that
echoes the body it was sent, storing only the username, was flagged for a role
it never stored.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.constructibility import OPEN_ACCOUNT_PREFIX, collect_constructibility_report

from tests.spec_driven_agent.test_account_step_contained import variant_app

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_a_handler_that_echoes_the_body_is_not_flagged(tmp_path):
    report = collect_constructibility_report(variant_app(tmp_path, "echo"))

    [backend] = report["backends"]
    assert backend["boot"] == "ok", backend
    assert not [issue for issue in report["issues"] if issue.startswith(OPEN_ACCOUNT_PREFIX)], report["issues"]
