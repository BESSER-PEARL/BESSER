"""The second account of the account-write step is its own, later account.

Review of round 10: the step logged in with the main probe's username, email and
password, so in an app whose first registered account is the admin it acted as
that admin, and an admin changing another account's role was flagged.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.constructibility import OPEN_ACCOUNT_PREFIX, collect_constructibility_report

from tests.spec_driven_agent.test_account_step_contained import variant_app

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.mark.parametrize("variant", ["first_admin", "self_only"])
def test_writes_only_an_admin_or_the_account_itself_may_make_are_not_flagged(tmp_path, variant):
    report = collect_constructibility_report(variant_app(tmp_path, variant))

    [backend] = report["backends"]
    assert backend["boot"] == "ok", backend
    assert not [issue for issue in report["issues"] if issue.startswith(OPEN_ACCOUNT_PREFIX)], report["issues"]
