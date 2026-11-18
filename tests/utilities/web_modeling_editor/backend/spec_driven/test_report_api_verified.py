"""A requirement whose API half a workflow exercised, screens not run, shows that note.

r13-p21: every requirement also cites a page, so none can be fully verified; the report
must still say the backend half ran and the screens did not, not a bare "implemented".
"""

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import build_verification_report

NOTE = ("Found in the source; API: a passing workflow called GET /watchlist/1/prices/, the cited route; "
        "screens: not run (the frontend build check did not run)")


def test_an_api_verified_requirement_carries_its_note():
    verdict = {"id": 2, "text": "Show the prices of the last 7 days", "status": "implemented",
               "evidence": [{"path": "backend/routers/analytics.py", "line": "return rows"}],
               "verification": "api_verified", "verification_note": NOTE}
    assert build_verification_report({"requirements": [verdict]}).verified[0].how.endswith(NOTE)
