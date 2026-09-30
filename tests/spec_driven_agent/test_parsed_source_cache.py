"""One parse per distinct source text, across all the validators.

Thirteen call sites in validation/ and planning/ each read and parse the
generated tree independently. Measured on fixture run_7f918e11/web_app with
only four of them exercised: 108 ast.parse calls for 25 distinct texts, 77%
redundant, one text parsed 14 times. The full static pass measured 198 parses
of 22 texts.

That pass runs once per Phase 3 fix attempt, and max_attempts is the remaining
turn budget -- so a 60-file app over 40 attempts spent minutes re-parsing
source that had not changed between attempts.
"""
import ast
from pathlib import Path

import pytest

from besser.spec_driven_agent.parsed_source import (
    cache_info,
    clear_cache,
    parse_source,
)

FIXTURE = Path(__file__).parent / "fixtures" / "run_7f918e11" / "web_app"


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


def test_identical_text_is_parsed_once():
    """The regression: the same source reaching two validators costs one parse."""
    source = (FIXTURE / "backend" / "sql_alchemy.py").read_text(encoding="utf-8-sig")

    first = parse_source(source)
    second = parse_source(source)

    assert first is second, "a second reader re-parsed identical text"
    assert cache_info().hits == 1


def test_a_changed_file_is_reparsed():
    """Keyed on text, not path -- a file repaired between fix attempts must
    not serve the pre-repair tree."""
    before = parse_source("x = 1\n")
    after = parse_source("x = 2\n")

    assert before is not after
    assert cache_info().misses == 2


def test_mode_is_part_of_the_key():
    """eval mode yields ast.Expression where exec yields ast.Module. Sharing
    across modes would hand a caller the wrong node type -- a correctness bug,
    not a performance one. Annotation readers use eval mode."""
    module = parse_source("List[int]", mode="exec")
    expression = parse_source("List[int]", mode="eval")

    assert isinstance(module, ast.Module)
    assert isinstance(expression, ast.Expression)
    assert module is not expression


def test_a_broken_file_still_names_itself():
    """ast.parse fills in "<unknown>" when given no filename, so an emptiness
    check never fires and the report would lose which file failed."""
    with pytest.raises(SyntaxError) as caught:
        parse_source("def broken(:\n", filename="routers/booking.py")

    assert caught.value.filename == "routers/booking.py"


def test_a_failed_parse_is_not_cached():
    """Otherwise a file repaired mid-run would keep raising the old error."""
    for _ in range(2):
        with pytest.raises(SyntaxError):
            parse_source("def broken(:\n")

    assert cache_info().currsize == 0

    repaired = parse_source("def fixed():\n    pass\n")
    assert isinstance(repaired, ast.Module)


def test_the_validators_no_longer_re_parse_the_same_file():
    """End to end on a real generated app, counting actual ast.parse calls.

    Pre-fix this was 108 calls for 25 texts. Asserting equality rather than a
    ratio, so a new unconverted call site shows up as a failure rather than as
    a slightly worse number nobody reads.
    """
    from besser.spec_driven_agent.planning.action_inventory import collect_action_endpoints
    from besser.spec_driven_agent.validation.python_source import (
        _create_schema_router_mismatches,
    )

    calls = {"n": 0}
    real = ast.parse

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    ast.parse = counting
    try:
        _create_schema_router_mismatches(str(FIXTURE))
        collect_action_endpoints(str(FIXTURE))
    finally:
        ast.parse = real

    info = cache_info()
    assert calls["n"] == info.currsize, (
        f"{calls['n']} parses produced only {info.currsize} distinct trees -- "
        f"a call site is bypassing parse_source"
    )
    assert info.hits > 0, "no sharing happened; the cache is not being reached"
