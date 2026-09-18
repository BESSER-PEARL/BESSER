"""Phase 2 drops sub-generator tools the primary already bundles, and every
run ships a .gitignore.

Empirical run (library model, clean deterministic path) still emitted stray
`pydantic/`, `sqlalchemy/`, `rest_api/` dirs next to the assembled `backend/`,
because the Phase-2 agent was handed the standalone sub-generator tools that
`generate_fastapi_backend` already bundles. It also shipped no `.gitignore`.
"""
import os

from besser.generators.llm.orchestrator import _REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY
from tests.generators.llm.test_phase3_toolchain import _build_orchestrator


def _names(tools):
    return {t["name"] for t in tools if isinstance(t, dict) and "name" in t}


def test_fastapi_primary_drops_bundled_subgenerator_tools(tmp_path):
    orch = _build_orchestrator(tmp_path)
    orch._generator_used = "generate_fastapi_backend"
    orch.tools = [
        {"name": "generate_fastapi_backend"},
        {"name": "generate_pydantic"},
        {"name": "generate_sqlalchemy"},
        {"name": "generate_rest_api"},
        {"name": "generate_react"},   # not bundled -> keep
        {"name": "write_file"},        # non-generator -> keep
    ]
    orch._drop_redundant_generator_tools()
    kept = _names(orch.tools)
    assert "generate_pydantic" not in kept
    assert "generate_sqlalchemy" not in kept
    assert "generate_rest_api" not in kept
    assert {"generate_fastapi_backend", "generate_react", "write_file"} <= kept


def test_non_bundling_primary_keeps_all(tmp_path):
    orch = _build_orchestrator(tmp_path)
    orch._generator_used = "generate_react"  # not in the map
    orch.tools = [{"name": "generate_pydantic"}, {"name": "generate_react"}]
    orch._drop_redundant_generator_tools()
    assert _names(orch.tools) == {"generate_pydantic", "generate_react"}


def test_no_generator_used_is_noop(tmp_path):
    orch = _build_orchestrator(tmp_path)
    orch._generator_used = None
    orch.tools = [{"name": "generate_pydantic"}]
    orch._drop_redundant_generator_tools()
    assert _names(orch.tools) == {"generate_pydantic"}


def test_gitignore_written(tmp_path):
    orch = _build_orchestrator(tmp_path)
    orch._ensure_gitignore()
    gi = os.path.join(orch.output_dir, ".gitignore")
    assert os.path.isfile(gi)
    content = open(gi, encoding="utf-8").read()
    for pat in ("__pycache__/", "*.db", ".env", "node_modules/"):
        assert pat in content


def test_gitignore_not_overwritten(tmp_path):
    orch = _build_orchestrator(tmp_path)
    os.makedirs(orch.output_dir, exist_ok=True)
    gi = os.path.join(orch.output_dir, ".gitignore")
    with open(gi, "w", encoding="utf-8") as fh:
        fh.write("custom\n")
    orch._ensure_gitignore()
    assert open(gi, encoding="utf-8").read() == "custom\n"


def test_map_documents_fastapi_bundle():
    fa = _REDUNDANT_GENERATOR_TOOLS_BY_PRIMARY["generate_fastapi_backend"]
    assert {"generate_pydantic", "generate_sqlalchemy", "generate_rest_api"} <= fa
