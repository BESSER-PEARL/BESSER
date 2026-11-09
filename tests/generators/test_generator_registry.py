"""``besser.generators.registry`` is the one list of generators.

The web editor's ``SUPPORTED_GENERATORS`` and the Spec-Driven Agent's generator
tools are derived from it. The snapshots under ``snapshots/`` were captured from
the two hand-written lists the registry replaced; an intended change to either
surface updates them.
"""
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from besser.generators import GeneratorInterface
from besser.generators import registry
from besser.generators.registry import GENERATORS, AGENT_TOOL_ORDER, agent_tools
from besser.spec_driven_agent.agent import tools
from besser.utilities.web_modeling_editor.backend.backend import get_api_root
from besser.utilities.web_modeling_editor.backend.config import SUPPORTED_GENERATORS
from besser.utilities.web_modeling_editor.backend.config import generators as backend_config
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_OUTPUT_FILENAME,
    BPMN_PROJECT_DIAGRAM_KEY,
    DEPLOYMENT_DIAGRAM_TYPE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = Path(__file__).parent / "snapshots"

WEB_EDITOR_IDS = [
    "python", "java", "pydantic", "test_case", "django", "spring", "backend", "web_app",
    "sqlalchemy", "sql", "supabase", "jsonschema", "jsonobject", "agent", "qiskit", "rdf",
    "rest_api", "react", "flutter", "terraform", "bpmn", "docker_compose", "pytorch",
    "tensorflow", "forecaster",
]
AGENT_TOOLS = {
    "pydantic": "generate_pydantic",
    "sqlalchemy": "generate_sqlalchemy",
    "backend": "generate_fastapi_backend",
    "django": "generate_django",
    "python": "generate_python_classes",
    "java": "generate_java_classes",
    "sql": "generate_sql",
    "jsonschema": "generate_json_schema",
    "rest_api": "generate_rest_api",
    "react": "generate_react",
    "web_app": "generate_web_app",
    "rdf": "generate_rdf",
    "qiskit": "generate_qiskit",
    "supabase": "generate_supabase",
    "jsonobject": "generate_json_object",
    "agent": "generate_baf",
    "bpmn": "generate_bpmn",
    "pytorch": "generate_pytorch",
    "tensorflow": "generate_tensorflow",
}
# The surfaces disagree here on purpose; offering one of these to the agent is
# a product decision, not a registry fix.
WEB_EDITOR_ONLY = {"spring", "flutter", "terraform", "docker_compose", "test_case", "forecaster"}


def _snapshot(name):
    return json.loads((SNAPSHOTS / name).read_text(encoding="utf-8"))


def _class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


# -- Registry integrity ------------------------------------------------------

def test_ids_and_tool_names_are_unique():
    ids = [spec.id for spec in GENERATORS]
    assert len(ids) == len(set(ids))
    names = [tool.name for tool in agent_tools()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("spec", GENERATORS, ids=lambda spec: spec.id)
def test_every_class_loads_and_is_a_generator(spec):
    assert re.fullmatch(r"[\w.]+:\w+", spec.class_path)
    cls = spec.load_class()
    assert cls.__name__ == spec.class_path.rpartition(":")[2]
    assert issubclass(cls, GeneratorInterface)
    assert spec.output_type in {"file", "zip"}
    assert spec.file_extension.startswith(".")


def test_importing_the_registry_imports_no_generator():
    code = (
        "import sys, besser.generators.registry\n"
        "loaded = [m for m in sys.modules if m.startswith('besser.generators.')\n"
        "          and m not in ('besser.generators.registry', 'besser.generators.generator_interface')]\n"
        "print(loaded)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    assert out.strip() == "[]"


def test_values_shared_with_the_backend_constants():
    by_id = {spec.id: spec for spec in GENERATORS}
    assert by_id["agent"].download_filename == AGENT_OUTPUT_FILENAME
    assert by_id["bpmn"].required_diagram_type == BPMN_PROJECT_DIAGRAM_KEY
    assert by_id["docker_compose"].required_diagram_type == DEPLOYMENT_DIAGRAM_TYPE


# -- The two derived surfaces --------------------------------------------------

def test_web_editor_and_agent_sets_are_pinned():
    assert list(SUPPORTED_GENERATORS) == WEB_EDITOR_IDS
    agent = {spec.id: spec.agent_tool.name for spec in GENERATORS if spec.agent_tool}
    assert agent == AGENT_TOOLS
    assert [tool["name"] for tool in tools.GENERATOR_TOOLS] == [AGENT_TOOLS[i] for i in AGENT_TOOL_ORDER]
    assert set(WEB_EDITOR_IDS) - set(AGENT_TOOLS) == WEB_EDITOR_ONLY
    assert set(AGENT_TOOLS) <= set(WEB_EDITOR_IDS)


def test_supported_generators_come_from_the_registry():
    for spec in GENERATORS:
        info = SUPPORTED_GENERATORS[spec.id]
        assert info.generator_class is spec.load_class()
        assert info._replace(generator_class=None) == backend_config.GeneratorInfo(
            generator_class=None,
            output_type=spec.output_type,
            file_extension=spec.file_extension,
            category=spec.category,
            requires_class_diagram=spec.requires_class_diagram,
            required_diagram_type=spec.required_diagram_type,
            download_filename=spec.download_filename,
            keep_generated_filename=spec.keep_generated_filename,
            needs_beyond_one_diagram=spec.needs_beyond_one_diagram,
        )


def test_agent_tools_and_requirements_come_from_the_registry():
    for tool in agent_tools():
        assert tool.schema() in tools.GENERATOR_TOOLS
        assert tools._TOOL_MODEL_REQUIREMENTS[tool.name] == tool.requires


def test_web_editor_payload_matches_the_snapshot():
    snapshot = _snapshot("web_editor_generators.json")
    assert get_api_root()["supported_generators"] == snapshot["supported_generators"]
    current = {
        key: {"generator_class": _class_name(info.generator_class),
              **{k: v for k, v in info._asdict().items() if k != "generator_class"}}
        for key, info in SUPPORTED_GENERATORS.items()
    }
    assert current == snapshot["generators"]


def test_agent_tool_schemas_match_the_snapshot():
    snapshot = _snapshot("agent_generator_tools.json")
    assert tools.GENERATOR_TOOLS == snapshot["generator_tools"]
    assert {k: sorted(v) for k, v in tools._TOOL_MODEL_REQUIREMENTS.items()} == snapshot["model_requirements"]
    assert [t["name"] for t in tools.get_all_tools_including_generators()] == snapshot["all_tool_names"]
    assert tools.get_available_generator_names(
        True, True, True, True, True, True, True, True) == snapshot["available_all_models"]
    assert tools.get_available_generator_names() == snapshot["available_domain_only"]


# -- Derivation rules ------------------------------------------------------------

def test_an_agent_tool_missing_from_the_order_raises(monkeypatch):
    extra = replace(GENERATORS[0], id="extra",
                    agent_tool=replace(GENERATORS[0].agent_tool, name="generate_extra"))
    monkeypatch.setattr(registry, "GENERATORS", (extra,) + GENERATORS)
    with pytest.raises(ValueError, match="extra"):
        registry.agent_tools()


# -- Docs --------------------------------------------------------------------------

def test_agent_tools_doc_lists_every_generator_tool():
    doc = (REPO_ROOT / "docs/source/spec_driven_agent/tools.rst").read_text(encoding="utf-8")
    section = doc.split("Generator tools\n---------------", 1)[1].split("\n\n\n", 1)[0]
    listed = set(re.findall(r"^   \* - ``(generate_\w+)``", section, re.MULTILINE))
    assert listed == set(AGENT_TOOLS.values())
