"""Every static walker prunes the run snapshot through ``walk_app_tree``.

The contract and frontend-contract walkers each carried their own copy of the
directory pruning and skipped the snapshot by a path-prefix test on each file,
which only catches it at the workspace root.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.state.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.validation.contract_checks import _walk_python
from besser.spec_driven_agent.validation.frontend_contract import (
    _method_button_source_issues,
    collect_frontend_contract_issues,
)
from besser.spec_driven_agent.validation.python_source import walk_app_tree

_DEAD_FORM = "export default () => <form onSubmit={() => {}}><input /></form>;\n"
_WRONG_TABLE_BUTTON = (
    'export default () => <>\n'
    '  <TableBlock id="t1" dataBinding={{"entity": "Booking"}} />\n'
    '  <MethodButton endpoint="/bill/{id}/methods/pay/" instanceSourceTableId="t1" label="Pay" />\n'
    '</>;\n')

_SNAPSHOTS = [_SNAPSHOT_DIR, f"{_SNAPSHOT_DIR}_staging", f"frontend/{_SNAPSHOT_DIR}"]


def _write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("snapshot", _SNAPSHOTS)
def test_walk_app_tree_prunes_a_snapshot(tmp_path, snapshot):
    _write(tmp_path, f"{snapshot}/x.py", "")
    _write(tmp_path, "node_modules/y.py", "")
    _write(tmp_path, "app/z.py", "")
    walked = {name for _, files in walk_app_tree(str(tmp_path)) for name in files}
    assert walked == {"z.py"}


@pytest.mark.parametrize("snapshot", _SNAPSHOTS)
def test_the_contract_walk_skips_a_snapshot(tmp_path, snapshot):
    _write(tmp_path, f"{snapshot}/old.py", "x = 1\n")
    _write(tmp_path, ".besser_probe.py", "x = 1\n")
    _write(tmp_path, "backend/main.py", "x = 1\n")
    parsed, _ = _walk_python(str(tmp_path))
    assert [rel for rel, _, _ in parsed] == ["backend/main.py"]


@pytest.mark.parametrize("snapshot", _SNAPSHOTS)
def test_the_frontend_contract_skips_a_snapshot(tmp_path, snapshot):
    _write(tmp_path, f"{snapshot}/src/Form.jsx", _DEAD_FORM)
    _write(tmp_path, f"{snapshot}/src/Booking.jsx", _WRONG_TABLE_BUTTON)
    assert collect_frontend_contract_issues(str(tmp_path)) == []
    assert _method_button_source_issues(str(tmp_path)) == []


def test_the_frontend_contract_still_reads_the_app(tmp_path):
    _write(tmp_path, "frontend/src/Form.jsx", _DEAD_FORM)
    _write(tmp_path, "frontend/src/Booking.jsx", _WRONG_TABLE_BUTTON)
    assert collect_frontend_contract_issues(str(tmp_path))
    assert _method_button_source_issues(str(tmp_path))
