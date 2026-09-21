"""The inventory lists the code the model must edit, not the junk beside it.

Measured on a healthy 60-file web-app scaffold (2026-09-18): the alphabetical
30-path cap listed .besser_* records and __pycache__/*.pyc in 5 slots and
listed 0 of the 33 frontend/src files ("... and 35 more files"); on a
delivered app 16 slots were .pyc/.db and three routers fell off the end. A
symbol map for every code file costs about the same as that list did.
"""
from besser.spec_driven_agent.prompt_builder import build_inventory


def _listed(inv):
    return [ln[4:].split(" (")[0] for ln in inv.splitlines() if ln.startswith("  - ")]


def test_junk_is_skipped_and_every_code_file_is_listed(tmp_path):
    (tmp_path / "pkg").mkdir()
    for i in range(40):
        (tmp_path / "pkg" / f"m{i:02d}.py").write_text(f"def f{i}():\n    pass\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "m00.cpython-311.pyc").write_bytes(b"\x00")
    (tmp_path / ".besser_trace.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "app.db").write_bytes(b"\x00")
    (tmp_path / "bundle.zip").write_bytes(b"\x00")

    inv = build_inventory(str(tmp_path), None, "x")
    listed = _listed(inv)
    assert len([p for p in listed if p.startswith("pkg/m")]) == 40
    assert not any(
        ".pyc" in p or ".besser_" in p or "node_modules" in p or p.endswith((".db", ".zip"))
        for p in listed
    ), listed
    assert "more files" not in inv


def test_a_very_large_tree_is_still_bounded_with_code_first(tmp_path):
    for i in range(200):
        (tmp_path / f"note{i:03d}.txt").write_text("n", encoding="utf-8")
    (tmp_path / "zzz_last.py").write_text("def z():\n    pass\n", encoding="utf-8")
    inv = build_inventory(str(tmp_path), None, "x")
    listed = _listed(inv)
    assert "zzz_last.py" in listed, "code files are listed before the cap applies"
    assert "more files" in inv
    assert len(listed) <= 150


def test_the_symbol_map_survives(tmp_path):
    (tmp_path / "a.py").write_text("class A:\n    pass\n\ndef b():\n    pass\n", encoding="utf-8")
    line = next(ln for ln in build_inventory(str(tmp_path), None, "x").splitlines() if "a.py" in ln)
    assert "A@1" in line and "b@4" in line
