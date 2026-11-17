"""A real decorator is located but proves nothing; it no longer sinks the citations after it.

Live evidence (2026-11-14, claude-sonnet-5-5, run r9-p21, R2): citation 1 was
the route decorator ``@router.get("/watchlist/...", response_model=None,
tags=["Analytics"])``, which ``ruff format`` had split over five lines. Any
quote starting with ``@`` and no ``def`` failed in the location phase, and
verification stops at the first citation that does not locate, so citations
2-6 (four of which prove the requirement on their own) were never weighed. A
real comment line was treated differently: located, merely non-proving.
"""

from __future__ import annotations

import pytest

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.planning.requirements_ledger import verify_evidence

# backend/routers/analytics.py of run r9-p21, the cited parts verbatim.
ANALYTICS = '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from sql_alchemy import Company, StockPrice, Watchlist

router = APIRouter()

DEFAULT_WINDOW_DAYS = 7  # "last 7 days"


def _price_row(p: StockPrice) -> dict:
    return {
        "id": p.id,
        "date": p.date.isoformat(),
        "open": p.open,
        "high": p.high,
        "low": p.low,
        "close": p.close,
        "volume": p.volume,
    }


@router.get(
    "/watchlist/{watchlist_id}/prices-last-7-days/",
    response_model=None,
    tags=["Analytics"],
)
def get_watchlist_prices_last_7_days(
    watchlist_id: int,
    days: int = DEFAULT_WINDOW_DAYS,
    database: Session = Depends(get_db),
) -> dict:
    """Last 7 days of prices for every company in a watchlist."""
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")
    watchlist = _get_watchlist(database, watchlist_id)
    return {
        "watchlist": {"id": watchlist.id, "name": watchlist.name},
        "days": days,
        "companies": [
            {
                "company": _company_info(c),
                "prices": [
                    _price_row(p) for p in _last_days_prices(database, c.id, days)
                ],
            }
            for c in watchlist.companies
        ],
    }
'''
COMPANY = '''from fastapi import APIRouter

router = APIRouter()


@router.get("/company/{company_id}/", response_model=None, tags=["Company"])
def get_company(company_id: int):
    return {"id": company_id}
'''
DASHBOARD = '''export default function Dashboard() {
  async function load() {
    try {
      const [p, v, f] = await Promise.all([
        watchlistPrices(selected),
        watchlistVisualization(selected, days),
        watchlistForecast(selected),
      ]);
      setPrices(p);
    } catch (e) {
      setError(e.message);
    }
  }
}
'''
API = '''// Analytics
export const watchlistPrices = (id) => api.get(`/watchlist/${id}/prices-last-7-days/`);
'''

# r9-p21 R2's evidence, verbatim from the recipe.
R2_EVIDENCE = [
    {"path": "backend/routers/analytics.py",
     "line": "@router.get(\"/watchlist/{watchlist_id}/prices-last-7-days/\", response_model=None, "
             "tags=[\"Analytics\"])"},
    {"path": "backend/routers/analytics.py",
     "line": "def get_watchlist_prices_last_7_days(watchlist_id: int, days: int = DEFAULT_WINDOW_DAYS, "
             "database: Session = Depends(get_db)) -> dict:"},
    {"path": "backend/routers/analytics.py", "line": "DEFAULT_WINDOW_DAYS = 7  # \"last 7 days\""},
    {"path": "backend/routers/analytics.py",
     "line": "def _price_row(p: StockPrice) -> dict: return { \"id\": p.id, \"date\": p.date.isoformat(), "
             "\"open\": p.open, \"high\": p.high, \"low\": p.low, \"close\": p.close, \"volume\": p.volume, }"},
    {"path": "frontend/src/pages/Dashboard.jsx",
     "line": "const [p, v, f] = await Promise.all([watchlistPrices(selected), watchlistVisualization(selected, days), "
             "watchlistForecast(selected)]);"},
    {"path": "frontend/src/api.js",
     "line": "export const watchlistPrices = (id) => api.get(`/watchlist/${id}/prices-last-7-days/`);"},
]
DECORATOR = R2_EVIDENCE[0]
GOOD_LINE = {"path": "backend/routers/analytics.py",
             "line": "_price_row(p) for p in _last_days_prices(database, c.id, days)"}
INVENTED_LINE = {"path": "backend/routers/analytics.py", "line": "prices = fetch_last_7_days(company_id)"}

VALIDATORS = '''from pydantic import BaseModel, model_validator


class BookingCreate(BaseModel):
    @model_validator(mode="after")
    def validate_capacity(self):
        return self
'''


def _check(tmp_path, evidence, kind="action",
           text="Display stock prices in numbers for the last 7 days for each company in the list"):
    for rel, content in {"backend/routers/analytics.py": ANALYTICS, "backend/routers/company.py": COMPANY,
                         "frontend/src/pages/Dashboard.jsx": DASHBOARD, "frontend/src/api.js": API,
                         "backend/schemas.py": VALIDATORS}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(content, encoding="utf-8")
    verdict = {"id": 2, "kind": kind, "text": text, "status": "implemented", "evidence": evidence}
    return verify_evidence([verdict], str(tmp_path))[0]


def test_the_r9_p21_verdict_verifies(tmp_path):
    verdict = _check(tmp_path, R2_EVIDENCE)
    assert verdict["status"] == "implemented", verdict.get("note")


def test_a_decorator_and_its_def_collapsed_onto_one_line_verify(tmp_path):
    line = DECORATOR["line"] + " " + R2_EVIDENCE[1]["line"]
    verdict = _check(tmp_path, [{"path": DECORATOR["path"], "line": line}])
    assert verdict["status"] == "implemented", verdict.get("note")


@pytest.mark.parametrize("evidence", [
    pytest.param([DECORATOR], id="route-decorator-alone"),
    pytest.param([{"path": "backend/schemas.py", "line": '@model_validator(mode="after")'}], id="bare-validator"),
    pytest.param([DECORATOR, INVENTED_LINE], id="decorator-then-invented-line"),
    pytest.param([{"path": DECORATOR["path"], "line": DECORATOR["line"].replace("7-days", "30-days")}, GOOD_LINE],
                 id="invented-decorator-then-good-line"),
    pytest.param([{"path": "backend/routers/company.py", "line": DECORATOR["line"]}, GOOD_LINE],
                 id="misfiled-decorator-then-good-line"),
    pytest.param([{"path": DECORATOR["path"], "line": DECORATOR["line"].replace("7-days", "30-days")
                   + " " + R2_EVIDENCE[1]["line"]}], id="invented-decorator-collapsed-with-real-def"),
    pytest.param([{"path": DECORATOR["path"], "line": DECORATOR["line"] + "\nprices = fetch_last_7_days(company_id)"}],
                 id="decorator-and-invented-line-in-one-quote"),
])
def test_a_decorator_never_proves_or_rescues_a_citation(tmp_path, evidence):
    verdict = _check(tmp_path, evidence)
    assert verdict["status"] == "unverified", verdict


def test_a_decorator_alone_says_why_it_proves_nothing(tmp_path):
    verdict = _check(tmp_path, [DECORATOR])
    assert "not only a decorator" in verdict["note"], verdict["note"]


def test_an_invented_decorator_fails_like_an_invented_line(tmp_path):
    verdict = _check(tmp_path, [{"path": DECORATOR["path"], "line": DECORATOR["line"].replace("7-days", "30-days")},
                                GOOD_LINE])
    assert "citation 1" in verdict["note"] and "not found in the cited source" in verdict["note"], verdict["note"]


# --- review of round 7 -------------------------------------------------------------------------

BOOKINGS = '''from dataclasses import dataclass


@dataclass
class Booking:
    id: int


@router.delete("/bookings/{booking_id}")
def remove_booking(booking_id: int):
    return delete_it(booking_id)


@router.post("/bookings")
def cancel_booking(booking_id: int):
    booking = get(booking_id)
    booking.status = "cancelled"
    return booking
'''
CONTROLLER = '''package app;

@RestController
public class BookingController {
    @PostMapping("/bookings")
    public ResponseEntity<Booking> create(@RequestBody Booking b) {
        return ResponseEntity.ok(repo.save(b));
    }

    @DeleteMapping("/bookings/{id}")
    public void remove(@PathVariable long id) {
        repo.deleteById(id);
    }
}
'''


def _check_bookings(tmp_path, path, line, kind="action", text="A user can cancel a booking"):
    for rel, content in {"backend/bookings.py": BOOKINGS,
                         "src/main/java/app/BookingController.java": CONTROLLER}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(content, encoding="utf-8")
    verdict = {"id": 1, "kind": kind, "text": text, "status": "implemented",
               "evidence": [{"path": path, "line": line}]}
    return verify_evidence([verdict], str(tmp_path))[0]


@pytest.mark.parametrize("path,line", [
    pytest.param("backend/bookings.py",
                 '@router.delete("/bookings/{booking_id}") def cancel_booking(booking_id: int):', id="python"),
    pytest.param("src/main/java/app/BookingController.java",
                 '@DeleteMapping("/bookings/{id}")\npublic ResponseEntity<Booking> create(@RequestBody Booking b) {',
                 id="java"),
])
def test_a_decorator_must_be_on_the_function_it_is_quoted_with(tmp_path, path, line):
    verdict = _check_bookings(tmp_path, path, line)
    assert verdict["status"] == "unverified", verdict


@pytest.mark.parametrize("path,line", [
    pytest.param("backend/bookings.py", "@dataclass\nclass Booking:", id="python-class"),
    pytest.param("src/main/java/app/BookingController.java",
                 '@PostMapping("/bookings")\npublic ResponseEntity<Booking> create(@RequestBody Booking b) {',
                 id="java-signature"),
    pytest.param("backend/bookings.py",
                 '@router.post("/bookings") def cancel_booking(booking_id: int):', id="python-collapsed-def"),
])
def test_an_annotation_and_its_signature_in_one_quote_locate(tmp_path, path, line):
    verdict = _check_bookings(tmp_path, path, line)
    assert "not found in the cited source" not in verdict.get("note", ""), verdict["note"]
    assert ledger.EVIDENCE_UNLOCATED not in verdict
