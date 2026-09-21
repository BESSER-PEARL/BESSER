"""Tests for the TableBlock-aware create-path detection in the acceptance
matrix (besser/spec_driven_agent/acceptance.py).

Context: an adversarial review of 17 live Spec-Driven Agent runs found the
``create`` cell false-positiving on the SAME three entities (Employee, Guest,
Person) in every run that used the GUI-model web-app scaffold -- a 100% hit
rate with zero true positives. Root cause: ``_POST_RE`` only recognises a
literal ``.post(`` / ``method: 'POST'`` string. The scaffold's only generated
create pattern is a ``<TableBlock>`` whose POST is issued at runtime by a
shared component (``TableComponent.tsx``, ``axios.post`` gated on
``modalMode === 'add'``), never by a literal call in the page file -- so the
old check was structurally blind to the only pattern this generator produces.

``PERSON_TSX`` and ``BOOKING_TSX`` below are copied verbatim from a real run,
not read from ``verification/`` at test time -- source:
verification/spec-iterations/gpt-5.6-terra-dp3trml9/app/web_app/frontend/src/pages/Person.tsx
verification/spec-iterations/gpt-5.6-terra-dp3trml9/app/web_app/frontend/src/pages/Booking.tsx
"""

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.acceptance import build_acceptance_matrix, matrix_issues

StringType = PrimitiveDataType("str")


def _model(*names) -> DomainModel:
    classes = set()
    for n in names:
        c = Class(name=n)
        c.attributes = {Property(name="name", type=StringType)}
        classes.add(c)
    return DomainModel(name="Test", types=classes)


def _workspace(tmp_path, files: dict):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# Verbatim: verification/spec-iterations/gpt-5.6-terra-dp3trml9/app/web_app/
# frontend/src/pages/Person.tsx. A real run's fully-wired create page: a
# <TableBlock> whose dataBinding names "Person"/"/person/" and whose
# options.formColumns lists five editable fields plus a booking lookup.
# There is no literal ".post(" / "method: 'POST'" anywhere in this file --
# the create POST is issued by TableComponent.tsx, a shared component this
# page only references through props.
PERSON_TSX = '''import React from "react";
import { TableBlock } from "../components/runtime/TableBlock";

const Person: React.FC = () => {
  return (
    <div id="page-person-0">
    <div id="iu0di" style={{"height": "100vh", "fontFamily": "Arial, sans-serif", "display": "flex", "--chart-color-palette": "default"}}>
      <nav id="ik079" style={{"width": "250px", "padding": "20px", "display": "flex", "overflowY": "auto", "background": "linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", "color": "white", "--chart-color-palette": "default", "flexDirection": "column"}}>
        <h2 id="ivcks" style={{"fontSize": "24px", "fontWeight": "bold", "marginTop": "0", "marginBottom": "30px", "--chart-color-palette": "default"}}>{"BESSER"}</h2>
        <div id="ihync" style={{"display": "flex", "--chart-color-palette": "default", "flexDirection": "column", "flex": "1"}}>
          <a id="iw3jr" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "rgba(255,255,255,0.2)", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/person">{"Person"}</a>
          <a id="il8z8" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/employee">{"Employee"}</a>
          <a id="i7fw2" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/guest">{"Guest"}</a>
          <a id="ibf98" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/room">{"Room"}</a>
          <a id="i22p1" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/booking">{"Booking"}</a>
          <a id="isbhd" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/reservedroom">{"ReservedRoom"}</a>
          <a id="im38j" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/bill">{"Bill"}</a>
        </div>
        <p id="iyfpq" style={{"fontSize": "11px", "paddingTop": "20px", "marginTop": "auto", "textAlign": "center", "opacity": "0.8", "borderTop": "1px solid rgba(255,255,255,0.2)", "--chart-color-palette": "default"}}>{"© 2026 BESSER. All rights reserved."}</p>
      </nav>
      <main id="im0qi" style={{"padding": "40px", "overflowY": "auto", "background": "#f5f5f5", "--chart-color-palette": "default", "flex": "1"}}>
        <h1 id="i130e" style={{"fontSize": "32px", "marginTop": "0", "marginBottom": "10px", "color": "#333", "--chart-color-palette": "default"}}>{"Person"}</h1>
        <p id="iqf8e" style={{"marginBottom": "30px", "color": "#666", "--chart-color-palette": "default"}}>{"Manage Person data"}</p>
        <TableBlock id="table-person-0" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} title="Person List" options={{"showHeader": true, "stripedRows": false, "showPagination": true, "rowsPerPage": 5, "actionButtons": true, "columns": [{"label": "Id", "column_type": "field", "field": "id", "type": "int", "required": true}, {"label": "IdentifyingNumber", "column_type": "field", "field": "identifyingNumber", "type": "str", "required": true}, {"label": "FirstName", "column_type": "field", "field": "firstName", "type": "str", "required": true}, {"label": "FamilyName", "column_type": "field", "field": "familyName", "type": "str", "required": true}, {"label": "Phone", "column_type": "field", "field": "phone", "type": "str", "required": true}, {"label": "Email", "column_type": "field", "field": "email", "type": "str", "required": true}], "formColumns": [{"column_type": "field", "field": "identifyingNumber", "label": "identifyingNumber", "type": "str", "required": true, "defaultValue": null}, {"column_type": "field", "field": "firstName", "label": "firstName", "type": "str", "required": true, "defaultValue": null}, {"column_type": "field", "field": "familyName", "label": "familyName", "type": "str", "required": true, "defaultValue": null}, {"column_type": "field", "field": "phone", "label": "phone", "type": "str", "required": true, "defaultValue": null}, {"column_type": "field", "field": "email", "label": "email", "type": "str", "required": true, "defaultValue": null}, {"column_type": "lookup", "path": "booking", "field": "booking", "lookup_field": "bookingNumber", "target_field": "id", "target_type": "int", "entity": "Booking", "type": "list", "required": false}]}} dataBinding={{"entity": "Person", "endpoint": "/person/", "row_key_fields": ["id"]}} />
      </main>
    </div>    </div>
  );
};

export default Person;
'''

# Verbatim: same run, .../pages/Booking.tsx. Booking's OWN dataBinding is
# {"entity": "Booking", ...}; "Guest"/"Person"/"Room"/"Employee" appear
# only as formColumns LOOKUP targets (e.g. {"entity": "Guest", ...} for the
# booking's guest list) and as unrelated sidebar nav links -- never as this
# file's own dataBinding. Used to prove the fix attributes a TableBlock's
# create-wiring to its own bound entity, not to every entity name the file
# happens to mention.
BOOKING_TSX = '''import React from "react";
import { TableBlock } from "../components/runtime/TableBlock";
import { MethodButton } from "../components/MethodButton";

const Booking: React.FC = () => {
  return (
    <div id="page-booking-4">
    <div id="iah0lx" style={{"height": "100vh", "fontFamily": "Arial, sans-serif", "display": "flex", "--chart-color-palette": "default"}}>
      <nav id="i8ng5w" style={{"width": "250px", "padding": "20px", "display": "flex", "overflowY": "auto", "background": "linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", "color": "white", "--chart-color-palette": "default", "flexDirection": "column"}}>
        <h2 id="ixclul" style={{"fontSize": "24px", "fontWeight": "bold", "marginTop": "0", "marginBottom": "30px", "--chart-color-palette": "default"}}>{"BESSER"}</h2>
        <div id="i9s1cc" style={{"display": "flex", "--chart-color-palette": "default", "flexDirection": "column", "flex": "1"}}>
          <a id="im9lyr" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/person">{"Person"}</a>
          <a id="i7tr4l" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/employee">{"Employee"}</a>
          <a id="i5idit" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/guest">{"Guest"}</a>
          <a id="ivkk0l" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/room">{"Room"}</a>
          <a id="iyy7qi" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "rgba(255,255,255,0.2)", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/booking">{"Booking"}</a>
          <a id="i0wxcf" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/reservedroom">{"ReservedRoom"}</a>
          <a id="io3b4m" style={{"padding": "10px 15px", "textDecoration": "none", "marginBottom": "5px", "display": "block", "background": "transparent", "color": "white", "borderRadius": "4px", "--chart-color-palette": "default"}} href="/bill">{"Bill"}</a>
        </div>
        <p id="i04rqg" style={{"fontSize": "11px", "paddingTop": "20px", "marginTop": "auto", "textAlign": "center", "opacity": "0.8", "borderTop": "1px solid rgba(255,255,255,0.2)", "--chart-color-palette": "default"}}>{"© 2026 BESSER. All rights reserved."}</p>
      </nav>
      <main id="iw8rta" style={{"padding": "40px", "overflowY": "auto", "background": "#f5f5f5", "--chart-color-palette": "default", "flex": "1"}}>
        <h1 id="i1f7lo" style={{"fontSize": "32px", "marginTop": "0", "marginBottom": "10px", "color": "#333", "--chart-color-palette": "default"}}>{"Booking"}</h1>
        <p id="iyqpzn" style={{"marginBottom": "30px", "color": "#666", "--chart-color-palette": "default"}}>{"Create and manage bookings, billing, payments, and guest stay status."}</p>
        <TableBlock id="table-booking-4" styles={{"width": "100%", "minHeight": "400px", "--chart-color-palette": "default"}} title="Booking List" options={{"showHeader": true, "stripedRows": false, "showPagination": true, "rowsPerPage": 5, "actionButtons": true, "columns": [{"label": "Id", "column_type": "field", "field": "id", "type": "int", "required": true}, {"label": "BookingNumber", "column_type": "field", "field": "bookingNumber", "type": "str", "required": true}, {"label": "ArrivalDate", "column_type": "field", "field": "arrivalDate", "type": "date", "required": true}, {"label": "DepartureDate", "column_type": "field", "field": "departureDate", "type": "date", "required": true}, {"label": "CommercialStatus", "column_type": "field", "field": "commercialStatus", "type": "enum", "options": ["AWAITING_PAYMENT", "CANCELLED", "CONFIRMED"], "required": true}, {"label": "PhysicalStatus", "column_type": "field", "field": "physicalStatus", "type": "enum", "options": ["CHECKED_IN", "CHECKED_OUT", "NOT_YET_ARRIVED"], "required": true}, {"label": "TotalPrice", "column_type": "field", "field": "totalPrice", "type": "float", "required": true}, {"label": "Contact", "column_type": "lookup", "path": "contact", "entity": "Person", "field": "identifyingNumber", "type": "str", "required": true}, {"label": "Rooms", "column_type": "lookup", "path": "rooms", "entity": "Room", "field": "roomNumber", "type": "list", "required": true}, {"label": "Bill", "column_type": "lookup", "path": "bill", "entity": "Bill", "field": "billNumber", "type": "str", "required": false}], "formColumns": [{"column_type": "field", "field": "bookingNumber", "label": "bookingNumber", "type": "str", "required": true, "defaultValue": null}, {"column_type": "field", "field": "arrivalDate", "label": "arrivalDate", "type": "date", "required": true, "defaultValue": null}, {"column_type": "field", "field": "departureDate", "label": "departureDate", "type": "date", "required": true, "defaultValue": null}, {"column_type": "lookup", "path": "rooms", "field": "rooms", "lookup_field": "roomNumber", "target_field": "id", "target_type": "int", "entity": "Room", "type": "list", "required": true, "association_class": {"entity": "ReservedRoom", "fields": [{"name": "agreedPrice", "type": "float", "required": true}, {"name": "extraCharges", "type": "float", "required": true}]}}, {"column_type": "lookup", "path": "contact", "field": "contact", "lookup_field": "identifyingNumber", "target_field": "id", "target_type": "int", "entity": "Person", "type": "str", "required": true}, {"column_type": "lookup", "path": "guest", "field": "guest", "lookup_field": "identifyingNumber", "target_field": "id", "target_type": "int", "entity": "Guest", "type": "list", "required": true}, {"column_type": "lookup", "path": "employee", "field": "employee", "lookup_field": "identifyingNumber", "target_field": "id", "target_type": "int", "entity": "Employee", "type": "str", "required": true}]}} dataBinding={{"entity": "Booking", "endpoint": "/booking/", "row_key_fields": ["id"]}} />
        <div id="ijha21" style={{"marginTop": "20px", "display": "flex", "--chart-color-palette": "default", "flexWrap": "wrap", "gap": "10px"}}>
          <MethodButton id="irbko8" className="action-button-component" style={{"padding": "6px 14px", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none", "letterSpacing": "0.01em", "display": "flex", "cursor": "pointer", "transition": "background 0.2s", "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)", "color": "#fff", "borderRadius": "4px", "border": "none", "boxShadow": "0 1px 4px rgba(37,99,235,0.10)", "--chart-color-palette": "default", "alignItems": "center"}} endpoint="/booking/{booking_id}/methods/produceBill/" label="produceBill" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
          <MethodButton id="i3z5n8" className="action-button-component" style={{"padding": "6px 14px", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none", "letterSpacing": "0.01em", "display": "flex", "cursor": "pointer", "transition": "background 0.2s", "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)", "color": "#fff", "borderRadius": "4px", "border": "none", "boxShadow": "0 1px 4px rgba(37,99,235,0.10)", "--chart-color-palette": "default", "alignItems": "center"}} endpoint="/booking/{booking_id}/methods/registerArrival/" label="registerArrival" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
          <MethodButton id="inx16j" className="action-button-component" style={{"padding": "6px 14px", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none", "letterSpacing": "0.01em", "display": "flex", "cursor": "pointer", "transition": "background 0.2s", "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)", "color": "#fff", "borderRadius": "4px", "border": "none", "boxShadow": "0 1px 4px rgba(37,99,235,0.10)", "--chart-color-palette": "default", "alignItems": "center"}} endpoint="/booking/{booking_id}/methods/registerDeparture/" label="registerDeparture" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
          <MethodButton id="iux1og" className="action-button-component" style={{"padding": "6px 14px", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none", "letterSpacing": "0.01em", "display": "flex", "cursor": "pointer", "transition": "background 0.2s", "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)", "color": "#fff", "borderRadius": "4px", "border": "none", "boxShadow": "0 1px 4px rgba(37,99,235,0.10)", "--chart-color-palette": "default", "alignItems": "center"}} endpoint="/booking/{booking_id}/methods/cancel/" label="cancel" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
          <MethodButton id="ihy0da" className="action-button-component" style={{"padding": "6px 14px", "fontSize": "13px", "fontWeight": "600", "textDecoration": "none", "letterSpacing": "0.01em", "display": "flex", "cursor": "pointer", "transition": "background 0.2s", "background": "linear-gradient(90deg, #2563eb 0%, #1e40af 100%)", "color": "#fff", "borderRadius": "4px", "border": "none", "boxShadow": "0 1px 4px rgba(37,99,235,0.10)", "--chart-color-palette": "default", "alignItems": "center"}} endpoint="/booking/{booking_id}/methods/computeAmountOwed/" label="computeAmountOwed" isInstanceMethod={true} instanceSourceTableId="table-booking-4" />
        </div>
      </main>
    </div>    </div>
  );
};

export default Booking;
'''


def _router(entity_path: str) -> str:
    return f'''from fastapi import APIRouter
router = APIRouter(prefix="/{entity_path}", tags=["{entity_path}"])
@router.get("/")
def list_{entity_path}():
    return []
'''


def test_real_person_tsx_table_block_create_is_recognized(tmp_path):
    """The exact false positive from the adversarial review: a real run's
    Person.tsx has a fully wired TableBlock (dataBinding + non-empty
    formColumns) and zero literal POST calls anywhere in the file. Against
    the pre-fix acceptance.py this produced create=False (a false "create
    form not wired" finding); this test fails on that version.
    """
    assert ".post(" not in PERSON_TSX
    assert "'POST'" not in PERSON_TSX and '"POST"' not in PERSON_TSX

    _workspace(tmp_path, {
        "backend/routers/person.py": _router("person"),
        "frontend/src/pages/Person.tsx": PERSON_TSX,
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Person"))
    assert matrix == {"Person": {"route": True, "page": True, "create": True}}
    assert matrix_issues(matrix) == []


def test_table_block_for_a_different_entity_does_not_count(tmp_path):
    """Booking.tsx mentions "Guest" (a sidebar nav link, and a formColumns
    lookup target for the booking's guest list) but its own <TableBlock> is
    bound to Booking, not Guest. This must not satisfy "create" for Guest --
    proves the check is scoped to each TableBlock's own dataBinding, not a
    whole-file keyword search.
    """
    _workspace(tmp_path, {
        "backend/routers/guest.py": _router("guest"),
        "frontend/src/pages/Booking.tsx": BOOKING_TSX,
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("Guest"))
    assert matrix["Guest"] == {"route": True, "page": True, "create": False}
    issues = matrix_issues(matrix)
    assert len(issues) == 1
    assert "no frontend create path found" in issues[0]


# Constructed: no run in the verification corpus (22 runs surveyed) has an
# empty formColumns array -- every generated TableBlock carries at least one
# editable field. Built by hand to prove a genuinely list-only page (a
# TableBlock with a real dataBinding but nothing to submit) is still
# reported missing -- the fix recognises the create pattern, it does not
# rubber-stamp every TableBlock as a create path.
READONLY_TABLE_TSX = '''import React from "react";
import { TableBlock } from "../components/runtime/TableBlock";

const AuditLog: React.FC = () => (
  <TableBlock id="table-auditlog-0" title="AuditLog List" options={{"showHeader": true, "actionButtons": false, "columns": [{"label": "Id", "column_type": "field", "field": "id", "type": "int", "required": true}], "formColumns": []}} dataBinding={{"entity": "AuditLog", "endpoint": "/auditlog/", "row_key_fields": ["id"]}} />
);

export default AuditLog;
'''


def test_table_block_with_empty_form_columns_is_still_flagged_missing(tmp_path):
    _workspace(tmp_path, {
        "backend/routers/auditlog.py": _router("auditlog"),
        "frontend/src/pages/AuditLog.tsx": READONLY_TABLE_TSX,
    })
    matrix = build_acceptance_matrix(str(tmp_path), _model("AuditLog"))
    assert matrix["AuditLog"] == {"route": True, "page": True, "create": False}
    issues = matrix_issues(matrix)
    assert len(issues) == 1
    assert "no frontend create path found" in issues[0]
