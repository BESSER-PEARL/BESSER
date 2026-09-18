"""LIVE end-to-end tests for the keyless FREE tier of the Spec-Driven ("vibe")
Agent, driven over the DEPLOYED backend SSE endpoint (POST /besser_api/smart-generate
with ``provider="free"``).

These prove the WHOLE generation pipeline runs against the real stack + real
model (Cloudflare-tunnelled qwen3-coder) and PRODUCES an artifact. They do NOT
assert the generated app boots/runs — that fidelity check (the Phase-3 boot
check) is a separate, deliberately-deferred piece. So a green run here means
"vibe generation completed and produced the expected kind of output", not
"the produced app works".

Two scenarios:
  1. full app   — a class model -> a generated backend app (FastAPI .py files);
  2. rust       — a class model -> generated Rust (a .rs file with structs).

SLOW + non-deterministic (real LLM on a shared GPU; ~1-3 min each, first call
adds ~60s of model VRAM reload). Skipped by default. Enable::

    RUN_LIVE_FREE_E2E=1 python -m pytest tests/live/test_vibe_free_e2e.py -s

Point at another host with BACKEND_URL (default: the experimental deploy). The
server must have the free tier configured (BESSER_FREE_LLM_* env).
"""
import io
import json
import os
import time
import zipfile

import pytest

# When true (set by the standalone runner), stream each SSE event to stdout so
# a human can WATCH the generation happen. Pytest leaves it off (quiet).
_LIVE_LOG = False

BACKEND_URL = os.environ.get(
    "BACKEND_URL", "https://experimental.besser-pearl.org/besser_api"
).rstrip("/")
_HOST = BACKEND_URL.rsplit("/besser_api", 1)[0]

# Cloudflare's bot filter 403s a bare requests/urllib UA on the hosted deploy.
_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126 Safari/537.36",
}

# Generous — first call reloads the model into VRAM (~60s) then generates.
_RUN_TIMEOUT = int(os.environ.get("FREE_E2E_TIMEOUT", "600"))

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_FREE_E2E"),
    reason="live free-tier e2e — set RUN_LIVE_FREE_E2E=1 to run",
)

# A small but non-trivial domain: Author 1..* --- 0..* Book, each with an attr.
_MODEL = {
    "elements": {
        "class-1": {"id": "class-1", "type": "Class", "name": "Author",
                     "attributes": ["attr-1"], "methods": []},
        "attr-1": {"id": "attr-1", "type": "ClassAttribute", "name": "name: str"},
        "class-2": {"id": "class-2", "type": "Class", "name": "Book",
                     "attributes": ["attr-2"], "methods": []},
        "attr-2": {"id": "attr-2", "type": "ClassAttribute", "name": "title: str"},
    },
    "relationships": {
        "rel-1": {
            "id": "rel-1", "type": "ClassBidirectional",
            "source": {"element": "class-1", "multiplicity": "1..*", "role": "authors"},
            "target": {"element": "class-2", "multiplicity": "0..*", "role": "books"},
        },
    },
}


def _project():
    return {
        "id": "free-e2e", "type": "BesserProject", "name": "FreeE2E",
        "createdAt": "2026-07-20T00:00:00Z",
        "diagrams": {"ClassDiagram": [{"id": "cd1", "title": "Library", "model": _MODEL}]},
        "currentDiagramIndices": {"ClassDiagram": 0},
    }


def _generate_free(instructions):
    """Run one keyless free generation to completion; return the ``done`` event.

    Fails loudly on an error event, a non-200, or a stream that ends without a
    ``done`` — the three ways the pipeline can be broken.
    """
    requests = pytest.importorskip("requests")
    body = {"project": _project(), "instructions": instructions, "provider": "free"}
    done = None
    t0 = time.monotonic()

    def _log(msg):
        if _LIVE_LOG:
            print(f"  [{int(time.monotonic() - t0):>4}s] {msg}", flush=True)

    _log(f"POST /smart-generate  provider=free  - {instructions[:60]}...")
    with requests.post(
        f"{BACKEND_URL}/smart-generate", json=body, headers=_HEADERS,
        stream=True, timeout=_RUN_TIMEOUT,
    ) as r:
        assert r.status_code == 200, f"smart-generate -> {r.status_code}: {r.text[:300]}"
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            try:
                evt = json.loads(raw[5:].strip())
            except json.JSONDecodeError:
                continue
            et = evt.get("event")
            if et == "start":
                # Sanity: the free tier really is what ran (not a silent fallback).
                assert evt.get("provider") == "free", evt
                _log(f"start  provider={evt.get('provider')} model={evt.get('llmModel')}")
            elif et == "phase":
                _log(f"phase  {evt.get('message')}")
            elif et == "error":
                pytest.fail(f"free run errored: {evt.get('code')} - {evt.get('message')}")
            elif et == "done":
                _log(f"done   file={evt.get('fileName')}")
                done = evt
                break
    assert done is not None, "SSE stream ended without a 'done' event"
    assert done.get("downloadUrl"), f"done event has no downloadUrl: {done}"
    return done


def _download_zip(done):
    """Download the produced artifact; return (namelist, ZipFile)."""
    requests = pytest.importorskip("requests")
    url = done["downloadUrl"]
    full = url if url.startswith("http") else _HOST + url
    r = requests.get(full, headers={"User-Agent": _HEADERS["User-Agent"]}, timeout=120)
    assert r.status_code == 200, f"download -> {r.status_code}"
    assert done.get("isZip"), f"expected a zip artifact: {done}"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    return zf.namelist(), zf


# ---------------------------------------------------------------------
# Scenario 1 — full vibe app: model -> generated backend app
# ---------------------------------------------------------------------

def test_free_tier_generates_a_backend_app():
    done = _generate_free(
        "Build a FastAPI backend application for this model, with CRUD endpoints "
        "for Author and Book."
    )
    names, _ = _download_zip(done)
    # It produced a real backend scaffold (not an empty / stub zip).
    assert any(n.endswith("main_api.py") for n in names), names
    assert any(n.endswith("pydantic_classes.py") for n in names), names
    assert len([n for n in names if n.endswith(".py")]) >= 3, names


# ---------------------------------------------------------------------
# Scenario 2 — model -> generate Rust classes
# ---------------------------------------------------------------------

def test_free_tier_generates_rust_classes():
    done = _generate_free(
        "Generate a single small Rust file (src/main.rs) with a struct for each "
        "class in this model (Author, Book) and their relationship. Idiomatic Rust."
    )
    names, zf = _download_zip(done)
    rs_files = [n for n in names if n.endswith(".rs")]
    if rs_files:
        body = zf.read(rs_files[0]).decode("utf-8", "replace").lower()
        assert "struct" in body, f"{rs_files[0]} has no struct: {body[:200]}"
    else:
        # Lenient fallback: the model is free/non-deterministic — accept Rust
        # emitted into any file, but surface the tree if nothing rust-ish exists.
        rustish = any(
            "struct " in zf.read(n).decode("utf-8", "replace").lower()
            for n in names if n.endswith((".rs", ".txt", ".md"))
        )
        assert rustish, f"no .rs file and no Rust structs produced; tree: {names}"


if __name__ == "__main__":
    # Standalone runner (demo-friendly: prints a summary, exits non-zero on
    # failure) — mirrors modeling-agent/tests/live/test_nl_generation_scenarios.py.
    os.environ.setdefault("RUN_LIVE_FREE_E2E", "1")
    _LIVE_LOG = True
    import traceback

    print(f"\nFree-tier vibe E2E  ->  {BACKEND_URL}")
    print("(real qwen3-coder on the shared GPU; first call adds ~60s of reload)\n")
    results = []
    for label, fn in (("full-app", test_free_tier_generates_a_backend_app),
                       ("rust-classes", test_free_tier_generates_rust_classes)):
        print(f"--- {label} ---")
        try:
            fn()
            results.append((label, "PASS", ""))
            print(f"  => PASS\n")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
            results.append((label, "FAIL", str(exc).splitlines()[0][:120]))
            traceback.print_exc()
            print(f"  => FAIL\n")
    print("===== summary =====")
    for label, verdict, note in results:
        print(f"  {verdict:4}  {label}  {note}")
    raise SystemExit(0 if all(v == "PASS" for _, v, _ in results) else 1)
