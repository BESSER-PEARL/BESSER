"""Every cited file reaches the citation-repair digest, however large the first one is.

Each focused file was capped at the WHOLE source budget, so one large cited
file filled the digest and a second cited file was listed as omitted: its
citation could not be repaired from a digest that did not show it.
"""

from besser.spec_driven_agent.planning import requirements_ledger as ledger


def _big_module(name: str, lines: int) -> str:
    return "\n".join(f"{name}_{i} = compute_{name}({i}, factor={i * 3})" for i in range(lines)) + "\n"


def test_two_large_cited_files_are_both_shown(tmp_path):
    for name in ("rooms", "guests"):
        path = tmp_path / "backend" / "services" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_big_module(name, 4000), encoding="utf-8")  # ~150k characters each

    digest = ledger.build_app_digest(
        str(tmp_path), focus_paths=["backend/services/rooms.py", "backend/services/guests.py"])
    source, coverage = digest.split("## Digest coverage")

    assert "### backend/services/rooms.py" in source
    assert "### backend/services/guests.py" in source, "the second cited file was omitted"
    assert "Omitted files:" not in coverage
    assert len(digest) <= ledger._DIGEST_MAX_TOTAL_CHARS


def test_one_cited_file_still_gets_the_whole_budget(tmp_path):
    path = tmp_path / "backend" / "services" / "rooms.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_big_module("rooms", 4000), encoding="utf-8")

    digest = ledger.build_app_digest(str(tmp_path), focus_paths=["backend/services/rooms.py"])

    assert "DIGEST TRUNCATED" not in digest
