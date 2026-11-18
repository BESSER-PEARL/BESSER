"""The judge's digest shows the code the agent wrote before code it barely touched.

Live evidence, r11-p81 (risk awareness): the agent added an auth dependency to
seven generated CRUD routers (20-44k characters each) and wrote the detection
logic in ``backend/risk_service.py`` (18k, every line its own). Every router the
agent touched ranked before every other module, so the routers filled the digest
and risk_service.py, the file the verdicts cite, was left out.
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger
from besser.spec_driven_agent.state.generator_output import store_generator_output

_DETECT = "    unwanted = [m for m in messages if m.isUnwanted]\n"


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _risk_app(root):
    """Generated routers the agent touched by one line, and a module it wrote whole."""
    _write(root, "backend/pydantic_classes.py", "class MessageCreate(BaseModel):\n    body: str\n" * 10)
    _write(root, "backend/sql_alchemy.py", "class Message(Base):\n    body = Column(String)\n" * 10)
    generated = {}
    for i in range(10):
        rel = f"backend/routers/entity{i:02}.py"
        body = "".join(f"@router.get('/entity{i}/{n}/')\ndef read_{i}_{n}(): return []\n" for n in range(40))
        generated[rel] = body.encode()
        _write(root, rel, "router = APIRouter(dependencies=[Depends(require_user)])\n" + body)
    _write(root, "backend/risk_service.py", f"def detect_signals(messages):\n{_DETECT}    return unwanted\n" * 25)
    store_generator_output(str(root), "backend", generated, "8.0.0")


def test_the_module_the_agent_wrote_is_shown_before_routers_it_touched(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", 20_000)
    _risk_app(tmp_path)

    digest, gaps = ledger.build_app_digest_and_gaps(str(tmp_path))

    assert "### backend/risk_service.py" in digest and _DETECT.strip() in digest
    assert "backend/risk_service.py" not in gaps
    # The schema files still come first.
    assert digest.index("### backend/sql_alchemy.py") < digest.index("### backend/risk_service.py")
    # Each touched router is shown as its changed line with context, the rest marked;
    # it counts as a gap, so a verdict naming it is judged again with the whole file.
    assert "Omitted files:" not in digest
    router = digest.split("### backend/routers/entity00.py\n", 1)[1].split("\n\n", 1)[0]
    assert router.startswith("router = APIRouter(dependencies=[Depends(require_user)])")
    assert "[77 unchanged generator lines left out]" in router
    assert "backend/routers/entity00.py" in gaps and "shown as their edits only" in digest


def test_without_the_region_view_the_written_module_still_wins(tmp_path, monkeypatch):
    """Ranking alone: with every router shown whole, they are what is left out."""
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", 20_000)
    monkeypatch.setattr(ledger, "_REGION_VIEW_MAX_SHARE", 0)
    _risk_app(tmp_path)
    digest = ledger.build_app_digest(str(tmp_path))
    assert "### backend/risk_service.py" in digest
    omitted = digest.split("Omitted files: ", 1)[1].split("\n", 1)[0].split(", ")
    assert omitted and all(rel.startswith("backend/routers/entity") for rel in omitted)


def test_a_router_the_agent_rewrote_ranks_before_one_it_touched(tmp_path):
    _risk_app(tmp_path)
    _write(tmp_path, "backend/routers/entity07.py", "def custom():\n    return compute()\n" * 30)
    files = ledger._source_files(str(tmp_path))
    shares = ledger._written_shares(str(tmp_path), files)

    assert shares["backend/routers/entity07.py"] == 1.0
    assert 0 < shares["backend/routers/entity00.py"] < 0.05
    assert "backend/risk_service.py" not in shares  # not generator output: wholly written
    ranked = sorted(files, key=lambda rel: ledger._digest_priority(rel, shares))
    assert ranked.index("backend/routers/entity07.py") < ranked.index("backend/routers/entity00.py")


def test_without_a_stored_generator_output_the_order_is_unchanged(tmp_path, monkeypatch):
    """No store (a from-scratch app): nothing to compare, so ranking stays by kind and path."""
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", 20_000)
    _write(tmp_path, "backend/routers/a.py", "x = 1\n")
    _write(tmp_path, "backend/b_service.py", "y = 2\n" * 3)
    files = ledger._source_files(str(tmp_path))

    assert ledger._written_shares(str(tmp_path), files) == {}
    digest = ledger.build_app_digest(str(tmp_path))
    assert digest.index("### backend/routers/a.py") < digest.index("### backend/b_service.py")


def test_a_removed_generator_route_is_marked_where_it_was():
    """Review: a deleted generated route showed only as unchanged lines left out."""
    generated = "".join(f"def route_{n}():\n    return {n}\n" for n in range(20))
    current = generated.replace("def route_10():\n    return 10\n", "")
    shown = ledger._changed_regions(ledger._compact(current), generated)
    assert "[2 generator lines removed here]" in shown
    before, after = shown.split("[2 generator lines removed here]")
    assert "def route_9():" in before and "def route_11():" in after


def test_a_very_long_repetitive_file_is_not_diffed(tmp_path, monkeypatch):
    """Review: a 4,900-line repetitive generated file took 54 s to diff on every cache-key
    computation. Over the bound its share is unknown: ranked as edited, shown whole."""
    import time

    generated = "".join(f"x_{n % 3} = {n % 5}\n" for n in range(4_900))
    current = generated + "y = 1\n"
    started = time.monotonic()
    assert ledger._written_share(current, generated) is None
    assert time.monotonic() - started < 1.0
    router = "".join(f"@router.get('/item/{n}/')\ndef item_{n}(): return fetch({n}, 'item')\n" for n in range(300))
    _write(tmp_path, "backend/routers/big.py", current)
    _write(tmp_path, "backend/routers/item.py", "router = APIRouter(dependencies=[Depends(user)])\n" + router)
    _write(tmp_path, "backend/pydantic_classes.py", "class A(BaseModel):\n    x: int\n")
    store_generator_output(str(tmp_path), "backend", {"backend/routers/big.py": generated.encode(),
                                                      "backend/routers/item.py": router.encode()}, "8.0.0")
    monkeypatch.setattr(ledger, "_DIGEST_MAX_TOTAL_CHARS", 40_000)
    digest = ledger.build_app_digest(str(tmp_path))
    # The whole app does not fit, so the router edited by one line is shown as its edit;
    # the file with an unknown share is shown whole (here cut at the per-file cap).
    big = digest.split("### backend/routers/big.py\n", 1)[1].split("\n\n", 1)[0]
    item = digest.split("### backend/routers/item.py\n", 1)[1].split("\n\n", 1)[0]
    assert "generator lines left out]" in item and "generator lines left out]" not in big
