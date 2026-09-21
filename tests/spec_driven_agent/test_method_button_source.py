"""A method button must take its id from a table of the method's own entity.

Run 19h35 (2026-09-18): the GUI model put ``registerPayment`` (a Bill method)
on the Bill page, bound to the Bill table. Phase 2 then copied that button into
``Booking.tsx`` and rebound it to the Booking table, so the page posted
``/bill/{booking id}/methods/registerPayment/`` — booking 2, which has no
bill, produced ``/bill/2/...``. The generated ``TableBlock`` names its entity in
``dataBinding``, so the mismatch is a one-file static check.
"""

import pytest

from besser.spec_driven_agent.orchestrator import _classify_issue, _method_button_source_issues

# The offending lines from the live page, verbatim apart from the styles.
_BOOKING_PAGE = '''
        <TableBlock id="table-booking-4" title="Booking List" options={{"showHeader": true}} dataBinding={{"entity": "Booking", "endpoint": "/booking/", "row_key_fields": ["id"]}} />
        <MethodButton id="ixji0f" className="action-button-component" endpoint="/booking/{booking_id}/methods/cancel/" label="cancel" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
        <MethodButton id="idrapv" className="action-button-component" endpoint="/bill/{bill_id}/methods/registerPayment/" label="registerPayment" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
'''

_BILL_PAGE = '''
        <TableBlock id="table-bill-6" title="Bill List" dataBinding={{"entity": "Bill", "endpoint": "/bill/", "row_key_fields": ["id"]}} />
        <MethodButton id="idrapv" endpoint="/bill/{bill_id}/methods/registerPayment/" label="registerPayment" isInstanceMethod={true} instanceSourceTableId="table-bill-6" />
'''


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_the_live_mismatch_is_a_blocker_naming_both_entities(tmp_path):
    _write(tmp_path, "web_app/frontend/src/pages/Booking.tsx", _BOOKING_PAGE)
    issues = _method_button_source_issues(str(tmp_path))
    assert len(issues) == 1, issues
    issue = issues[0]
    assert issue.startswith("frontend contract: web_app/frontend/src/pages/Booking.tsx line 4:")
    assert "registerPayment" in issue and "/bill/" in issue
    assert "table-booking-4" in issue and "Booking" in issue
    assert _classify_issue(issue).severity == "blocker"


def test_the_correctly_bound_page_is_clean(tmp_path):
    # 8efe8fd4: the first entity belongs to a lookup column, not this table.
    page = _BILL_PAGE.replace(
        'dataBinding=',
        'columns={[{"lookup": {"entity": "Booking"}, "label": "Amount > 0"}]} dataBinding=',
    )
    _write(tmp_path, "web_app/frontend/src/pages/Bill.tsx", page)
    assert _method_button_source_issues(str(tmp_path)) == []


def test_dynamic_table_bindings_are_not_guessed_from_lookup_columns(tmp_path):
    page = _BILL_PAGE.replace(
        'dataBinding={{"entity": "Bill", "endpoint": "/bill/", "row_key_fields": ["id"]}}',
        'columns={[{"entity": "Booking"}]} dataBinding={binding}',
    )
    _write(tmp_path, "frontend/src/pages/Bill.tsx", page)
    assert _method_button_source_issues(str(tmp_path)) == []
    examples = (
        '/* <TableBlock id="example" dataBinding={{"entity": "Booking"}} /> */\n'
        'const example = \'<MethodButton endpoint="/bill/1/methods/pay/" instanceSourceTableId="example" />\';\n'
    )
    _write(tmp_path, "frontend/src/pages/Bill.tsx", examples + _BILL_PAGE)
    assert _method_button_source_issues(str(tmp_path)) == []


def test_a_button_whose_table_is_not_on_the_page_is_left_alone(tmp_path):
    """No table to compare against, no verdict: the check never guesses."""
    _write(tmp_path, "web_app/frontend/src/pages/Booking.tsx",
           '<MethodButton endpoint="/bill/{bill_id}/methods/registerPayment/" '
           'label="registerPayment" instanceSourceTableId="table-elsewhere" />\n')
    assert _method_button_source_issues(str(tmp_path)) == []
