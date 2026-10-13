"""A/B comparison: the Spec-Driven Agent with and without the scaffold map.

Runs the same requests through the real pipeline (Phase 1 generator, planner,
Phase 2 LLM loop, Phase 3 validation) twice per repeat: once with the map
(the "Where to add code" overview and the ``locate_code`` tool) and once with
``BESSER_LLM_SCAFFOLD_MAP=0`` (neither, as before the map existed). Arms
alternate so provider drift hits both alike.

Each scenario asks for changes that land exactly where the map is meant to
help: a static route that the ``/{id}/`` route captures if placed after it, a
model method whose generated handler is an HTTP 501 stub, a field that must
reach the ORM, the schema and the create handler, a cross-record rule. Every
run is then judged on the delivered app, not on what the agent says:

* functional checks: the generated backend is booted with FastAPI's
  ``TestClient`` in a subprocess (fresh SQLite) and each requested behaviour
  is exercised over HTTP;
* process metrics from the run's recipe: turns, tool calls, failed edits,
  ``locate_code`` calls, cost, blockers left by Phase 3;
* placement pitfalls the map's own FastAPI extractor detects in the final
  code (dead ``@app`` routes, captured static routes, unmounted routers).

Real LLM calls, so slow and paid (about 0.5-2 USD per run on a frontier model)
and noisy: use several repeats before reading anything into a difference.
The API key is read from the provider's usual variable (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ...)::

    python tests/live/scaffold_map_ab.py --provider anthropic --model <model id> \\
        --repeats 3 --out ab_results

``--dry-run`` replaces the LLM with one that ends its turn at once: no network,
no cost; it checks the harness end to end on the bare scaffold (whose checks
must fail, since nothing was implemented).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    IntegerType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)

ARMS = ("map", "baseline")
_EDIT_TOOLS = ("modify_file", "replace_file_lines", "write_file")
_PITFALL_WORDS = ("dead", "captured", "never mounted", "shadow", "not mounted")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    model: Callable[[], DomainModel]
    instructions: str
    # Python source defining ``probe(client) -> dict[str, bool]``; runs in the
    # generated backend's directory against a fresh database.
    probe: str
    checks: list[str] = field(default_factory=list)


def _library_model() -> DomainModel:
    library = Class(name="Library", attributes={
        Property(name="name", type=StringType), Property(name="address", type=StringType),
    })
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType), Property(name="pages", type=IntegerType),
        Property(name="price", type=FloatType),
    })
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    book.add_method(Method(name="applyDiscount", parameters=[Parameter(name="percent", type=FloatType)],
                           type=FloatType))
    return DomainModel(
        name="Library",
        types={library, book, author},
        associations={
            BinaryAssociation(name="lib_book", ends={
                Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1)),
                Property(name="has", type=book, multiplicity=Multiplicity(0, "*")),
            }),
            BinaryAssociation(name="book_author", ends={
                Property(name="writtenBy", type=author, multiplicity=Multiplicity(0, "*")),
                Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*")),
            }),
        },
    )


_LIBRARY_PROBE = '''
def rec(response, name):
    """The record in a response: generated handlers wrap it as {"<entity>": {...}, ...}."""
    try:
        body = response.json()
    except Exception:
        return {}
    if isinstance(body, dict) and isinstance(body.get(name), dict):
        return body[name]
    return body if isinstance(body, dict) else {}


def numbers(response):
    """Numbers in a JSON body, two levels deep; numeric strings count (the generated
    instance-method handler returns ``"result": str(result)``)."""
    try:
        body = response.json()
    except Exception:
        return []
    values = [body] if not isinstance(body, dict) else list(body.values())
    values += [v for d in values if isinstance(d, dict) for v in d.values()]
    out = []
    for v in values:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def probe(c):
    out = {}
    lib = c.post("/library/", json={"name": "Central", "address": "Main St"})
    lib_id = rec(lib, "library").get("id") if lib.status_code < 300 else None
    book = {"title": "T", "pages": 100, "price": 100.0, "isbn": "978-3-16-148410-0",
            "locatedIn": lib_id, "writtenBy": []}
    created = c.post("/book/", json=book)
    book_id = rec(created, "book").get("id") if created.status_code < 300 else None
    # price 50 (not 100): no sum or average of prices can equal the expected 200 average pages
    c.post("/book/", json={**book, "title": "U", "pages": 300, "price": 50.0, "isbn": "978-0-00-000000-2"})

    stats = c.get("/book/stats/")
    body = rec(stats, "stats") if stats.status_code == 200 else {}
    values = list(body.values()) + [v for d in body.values() if isinstance(d, dict) for v in d.values()]
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out["stats_route"] = stats.status_code == 200 and 2 in nums and 200 in [round(n) for n in nums]

    if book_id is not None:
        r = c.post(f"/book/{book_id}/methods/applyDiscount/", json={"params": {"percent": 10}})
        after = c.get(f"/book/{book_id}/")
        price = rec(after, "book").get("price") if after.status_code == 200 else None
        returned = r.status_code < 300 and any(abs(n - 90.0) < 0.01 for n in numbers(r))
        out["discount_method"] = returned and price is not None and abs(price - 90.0) < 0.01
        isbn = rec(after, "book").get("isbn") if after.status_code == 200 else None
        out["isbn_field"] = isinstance(isbn, str) and isbn.replace("-", "").replace(" ", "") == "9783161484100"
    else:
        out["discount_method"] = out["isbn_field"] = False
    # Only meaningful when a complete book can be created at all.
    no_isbn = {k: v for k, v in book.items() if k != "isbn"}
    out["isbn_required"] = book_id is not None and c.post("/book/", json=no_isbn).status_code == 422
    return out
'''


def _booking_model() -> DomainModel:
    room_status = Enumeration(name="RoomStatus", literals={
        EnumerationLiteral(name="AVAILABLE"), EnumerationLiteral(name="MAINTENANCE"),
    })
    booking_status = Enumeration(name="BookingStatus", literals={
        EnumerationLiteral(name="CONFIRMED"), EnumerationLiteral(name="CANCELLED"),
    })
    room = Class(name="Room", attributes={
        Property(name="number", type=IntegerType), Property(name="status", type=room_status),
    })
    booking = Class(name="Booking", attributes={
        Property(name="date", type=DateType), Property(name="status", type=booking_status),
    })
    return DomainModel(
        name="Hotel",
        types={room_status, booking_status, room, booking},
        associations={BinaryAssociation(name="room_booking", ends={
            Property(name="room", type=room, multiplicity=Multiplicity(1, 1)),
            Property(name="bookings", type=booking, multiplicity=Multiplicity(0, "*")),
        })},
    )


_BOOKING_PROBE = '''
def rec(response, name):
    """The record in a response: generated handlers wrap it as {"<entity>": {...}, ...}."""
    try:
        body = response.json()
    except Exception:
        return {}
    if isinstance(body, dict) and isinstance(body.get(name), dict):
        return body[name]
    return body if isinstance(body, dict) else {}


def probe(c):
    out = {}
    free = c.post("/room/", json={"number": 101, "status": "AVAILABLE"})
    busy = c.post("/room/", json={"number": 102, "status": "MAINTENANCE"})
    free_id = rec(free, "room").get("id") if free.status_code < 300 else None
    busy_id = rec(busy, "room").get("id") if busy.status_code < 300 else None
    first = c.post("/booking/", json={"date": "2030-05-01", "status": "CONFIRMED", "room": free_id})
    first_id = rec(first, "booking").get("id") if first.status_code < 300 else None
    clash = c.post("/booking/", json={"date": "2030-05-01", "status": "CONFIRMED", "room": free_id})
    other_day = c.post("/booking/", json={"date": "2030-05-02", "status": "CONFIRMED", "room": free_id})
    # Only a CONFIRMED booking blocks the date: a cancelled one must not.
    cancelled = c.post("/booking/", json={"date": "2030-05-03", "status": "CANCELLED", "room": free_id})
    after_cancelled = c.post("/booking/", json={"date": "2030-05-03", "status": "CONFIRMED", "room": free_id})
    out["overlap_rejected"] = (first.status_code < 300 and clash.status_code == 409 and other_day.status_code < 300
                               and cancelled.status_code < 300 and after_cancelled.status_code < 300)

    if first_id is not None:
        cancel = c.post(f"/booking/{first_id}/cancel/")
        returned = str(rec(cancel, "booking").get("status")).upper().endswith("CANCELLED")
        after = c.get(f"/booking/{first_id}/")
        status = rec(after, "booking").get("status") if after.status_code == 200 else None
        out["cancel_route"] = cancel.status_code < 300 and returned and str(status).upper().endswith("CANCELLED")
    else:
        out["cancel_route"] = False

    avail = c.get("/room/available/")
    items = avail.json() if avail.status_code == 200 else []
    if isinstance(items, dict):  # a wrapper such as {"rooms": [...], "count": 1}
        items = next((v for v in items.values() if isinstance(v, list)), [])
    items = items if isinstance(items, list) else []
    ids = [(item.get("room") if isinstance(item.get("room"), dict) else item).get("id")
           for item in items if isinstance(item, dict)]
    out["available_route"] = avail.status_code == 200 and free_id in ids and busy_id not in ids
    return out
'''

SCENARIOS = {
    "library": Scenario(
        name="library",
        model=_library_model,
        instructions=(
            "Add an endpoint GET /book/stats/ that returns the number of books and their average "
            "number of pages as JSON. Implement Book.applyDiscount(percent): it lowers the book's "
            "price by that percentage, saves it and returns the new price. Add a required `isbn` "
            "string field to Book: creating a book without it must fail with HTTP 422, and the "
            "isbn must be returned when a book is read."
        ),
        probe=_LIBRARY_PROBE,
        checks=["stats_route", "discount_method", "isbn_field", "isbn_required"],
    ),
    "booking": Scenario(
        name="booking",
        model=_booking_model,
        instructions=(
            "Add POST /booking/{booking_id}/cancel/ that sets the booking's status to CANCELLED and "
            "returns the booking. Creating a booking for a room that already has a CONFIRMED "
            "booking on the same date must fail with HTTP 409. Add GET /room/available/ that lists "
            "only the rooms whose status is AVAILABLE."
        ),
        probe=_BOOKING_PROBE,
        checks=["overlap_rejected", "cancel_route", "available_route"],
    ),
}


# ---------------------------------------------------------------------------
# Running one generation
# ---------------------------------------------------------------------------


class _EndTurnClient:
    """--dry-run LLM: answers every call by ending the turn."""

    model = "dry-run"
    max_tokens = 4096

    class _Usage:
        api_calls = input_tokens = output_tokens = total_tokens = 0
        cache_creation_tokens = cache_read_tokens = 0
        estimated_cost = 0.0
        billed_to_user = False
        model = "dry-run"

        def summary(self):
            return {"api_calls": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "model": "dry-run"}

    usage = _Usage()

    def chat(self, system=None, messages=None, tools=None, **kwargs):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}]}


@contextlib.contextmanager
def _map_switch(arm: str):
    old = os.environ.get("BESSER_LLM_SCAFFOLD_MAP")
    os.environ["BESSER_LLM_SCAFFOLD_MAP"] = "1" if arm == "map" else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("BESSER_LLM_SCAFFOLD_MAP", None)
        else:
            os.environ["BESSER_LLM_SCAFFOLD_MAP"] = old


def _client(args):
    if args.dry_run:
        return _EndTurnClient()
    from besser.spec_driven_agent.providers.llm_client import create_llm_client
    return create_llm_client(provider=args.provider, api_key=None, model=args.model)


def _generate(scenario: Scenario, arm: str, workspace: Path, args) -> Optional[str]:
    """Run the pipeline into ``workspace``; returns an error string or None."""
    from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

    with _map_switch(arm):
        try:
            orchestrator = LLMOrchestrator(
                llm_client=_client(args),
                domain_model=scenario.model(),
                output_dir=str(workspace),
                max_cost_usd=args.max_cost,
                max_runtime_seconds=args.max_runtime,
                enable_toolchain_validation=False,
                allow_shell_tools=args.shell,
            )
            orchestrator.run(scenario.instructions)
        except Exception as exc:  # a crashed run is a result, not a harness failure
            return f"{type(exc).__name__}: {exc}"[:500]
    return None


# ---------------------------------------------------------------------------
# Judging one run
# ---------------------------------------------------------------------------

_PROBE_RUNNER = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, ".")
    result = {"boots": False}
    try:
        from fastapi.testclient import TestClient
        import main_api
        with TestClient(main_api.app) as client:
            result["boots"] = client.get("/").status_code < 500
            ns = {}
            exec(PROBE, ns)
            for key, value in ns["probe"](client).items():
                result[key] = bool(value)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print("PROBE_RESULT " + json.dumps(result))
''')


def _backend_dir(workspace: Path) -> Optional[Path]:
    hits = sorted(p.parent for p in workspace.rglob("main_api.py")
                  if "node_modules" not in p.parts and ".besser_snapshot" not in p.parts)
    return hits[0] if hits else None


def run_checks(scenario: Scenario, workspace: Path, timeout: int = 180) -> dict:
    """Boot a copy of the generated backend and run the scenario's probe."""
    backend = _backend_dir(workspace)
    if backend is None:
        return {"boots": False, "error": "no main_api.py"}
    with tempfile.TemporaryDirectory(prefix="besser_ab_probe_") as tmp:
        copy = Path(tmp) / "backend"
        shutil.copytree(backend, copy, ignore=shutil.ignore_patterns("*.db", "__pycache__"))
        script = f"PROBE = {scenario.probe!r}\n{_PROBE_RUNNER}"
        try:
            proc = subprocess.run([sys.executable, "-c", script], cwd=copy, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"boots": False, "error": "probe timed out"}
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_RESULT ")), None)
    if line is None:
        return {"boots": False, "error": (proc.stderr or proc.stdout)[-300:]}
    return json.loads(line[len("PROBE_RESULT "):])


def run_metrics(workspace: Path) -> dict:
    """Process metrics from the run's recipe, and placement pitfalls left in the code."""
    from besser.spec_driven_agent.planning.scaffold_map import build_workspace_map

    recipe_path = workspace / ".besser_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.is_file() else {}
    calls = recipe.get("tool_calls") or []
    edits = [c for c in calls if c.get("tool") in _EDIT_TOOLS]
    issues = recipe.get("validation_issues") or []
    usage = recipe.get("usage") or {}
    pitfalls = [w for m in build_workspace_map(str(workspace)).maps if m.family == "fastapi"
                for w in m.warnings if any(word in w.lower() for word in _PITFALL_WORDS)]
    requirements = recipe.get("requirements") or []
    return {
        "turns": recipe.get("turns"),
        "tool_calls": len(calls),
        "edits": len(edits),
        "failed_edits": sum(1 for c in edits if c.get("success") is False),
        "failed_tool_calls": sum(1 for c in calls if c.get("success") is False),
        "read_file_calls": sum(1 for c in calls if c.get("tool") == "read_file"),
        "search_calls": sum(1 for c in calls if c.get("tool") == "search_in_files"),
        "locate_code_calls": sum(1 for c in calls if c.get("tool") == "locate_code"),
        "blockers": sum(1 for i in issues if i.get("severity") == "blocker"),
        "warnings": sum(1 for i in issues if i.get("severity") == "warning"),
        "requirements_met": sum(1 for r in requirements if isinstance(r, dict) and r.get("verdict") == "met"),
        "requirements_total": len(requirements),
        "cost_usd": usage.get("estimated_cost_usd"),
        "tokens": usage.get("total_tokens"),
        "elapsed_s": recipe.get("elapsed_seconds"),
        "placement_pitfalls": pitfalls,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_one(scenario: Scenario, arm: str, repeat: int, args) -> dict:
    out_dir = Path(args.out)
    workspace = out_dir / "workspaces" / f"{scenario.name}-{arm}-{repeat}"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    started = time.monotonic()
    error = _generate(scenario, arm, workspace, args)
    record = {
        "scenario": scenario.name, "arm": arm, "repeat": repeat,
        "error": error, "wall_s": round(time.monotonic() - started, 1),
        "checks": run_checks(scenario, workspace),
        "metrics": run_metrics(workspace),
    }
    passed = [record["checks"].get(name) is True for name in scenario.checks]
    record["checks_passed"] = sum(passed)
    record["checks_total"] = len(passed)
    record["all_passed"] = all(passed) and record["checks"].get("boots") is True
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs" / f"{scenario.name}-{arm}-{repeat}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(values), 3) if values else None


def summarize(records: list[dict]) -> str:
    """A markdown table per scenario: one column per arm."""
    lines = ["# Scaffold map A/B", ""]
    for scenario in sorted({r["scenario"] for r in records}):
        rows = [r for r in records if r["scenario"] == scenario]
        by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
        checks = SCENARIOS[scenario].checks if scenario in SCENARIOS else []
        lines += [f"## {scenario}", "", "| | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]

        def row(label, fn):
            lines.append(f"| {label} | " + " | ".join(str(fn(by_arm[a])) for a in ARMS) + " |")

        row("runs", len)
        row("all checks pass", lambda rs: f"{sum(r['all_passed'] for r in rs)}/{len(rs)}")
        row("app boots", lambda rs: f"{sum(r['checks'].get('boots') is True for r in rs)}/{len(rs)}")
        for name in checks:
            row(f"check: {name}", lambda rs, n=name: f"{sum(r['checks'].get(n) is True for r in rs)}/{len(rs)}")
        for key in ("turns", "tool_calls", "edits", "failed_edits", "failed_tool_calls", "read_file_calls",
                    "search_calls", "locate_code_calls", "blockers", "cost_usd", "elapsed_s"):
            row(f"mean {key}", lambda rs, k=key: _mean([r["metrics"].get(k) for r in rs]))
        row("runs with placement pitfalls",
            lambda rs: f"{sum(bool(r['metrics'].get('placement_pitfalls')) for r in rs)}/{len(rs)}")
        row("crashed runs", lambda rs: sum(bool(r["error"]) for r in rs))
        lines.append("")
    lines.append("Small samples: treat differences of one or two runs as noise.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None, help="model id (provider default when omitted)")
    parser.add_argument("--scenarios", nargs="*", default=sorted(SCENARIOS), choices=sorted(SCENARIOS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=2.0, help="USD cap per run")
    parser.add_argument("--max-runtime", type=int, default=1200, help="seconds cap per run")
    parser.add_argument("--shell", action="store_true", help="give the agent run_command (trusted host only)")
    parser.add_argument("--out", default="scaffold_map_ab_results")
    parser.add_argument("--dry-run", action="store_true", help="no LLM: end every turn at once")
    args = parser.parse_args(argv)

    records = []
    for repeat in range(1, args.repeats + 1):
        for scenario_name in args.scenarios:
            # Alternate which arm goes first, so slow drift does not favour one.
            arms = ARMS if repeat % 2 else tuple(reversed(ARMS))
            for arm in arms:
                record = run_one(SCENARIOS[scenario_name], arm, repeat, args)
                records.append(record)
                print(f"[{scenario_name} #{repeat} {arm}] checks {record['checks_passed']}/"
                      f"{record['checks_total']} boots={record['checks'].get('boots')} "
                      f"cost={record['metrics'].get('cost_usd')} error={record['error']}", flush=True)
    summary = summarize(records)
    Path(args.out, "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
