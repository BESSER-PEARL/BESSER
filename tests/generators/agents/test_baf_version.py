"""One BAF release is used wherever BESSER installs the Agentic Framework."""
import pathlib
import re

from besser.generators.agents.baf_version import BAF_VERSION
from besser.generators.docker_compose import docker_compose_generator

ROOT = pathlib.Path(__file__).resolve().parents[3]
# `besser-agentic-framework[extras,llms]==4.6.1`, with or without extras
PIN = re.compile(r"besser-agentic-framework(?:\[[^\]]*\])?==([0-9][0-9A-Za-z.]*)")
# frontend is a separate repo (a submodule); releases/ are history
SKIPPED_PARTS = {"frontend", "releases", "node_modules", "__pycache__", ".git"}
SOURCE_SUFFIXES = {".py", ".j2", ".txt", ".rst", ".md", ".yml", ".yaml", ".toml", ".cfg"}


def test_docker_compose_uses_the_shared_version():
    assert docker_compose_generator.BAF_VERSION == BAF_VERSION


def test_every_pin_in_besser_is_the_shared_version():
    found = {}
    for base in (ROOT / "besser", ROOT / "docs" / "source"):
        for path in base.rglob("*"):
            if not path.is_file() or SKIPPED_PARTS & set(path.parts):
                continue
            if path.suffix not in SOURCE_SUFFIXES and path.name != "Dockerfile":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for version in PIN.findall(text):
                found.setdefault(version, []).append(str(path.relative_to(ROOT)))
    stale = {version: files for version, files in found.items() if version != BAF_VERSION}
    assert not stale, f"pins that differ from BAF_VERSION {BAF_VERSION}: {stale}"
    # the simulator image's requirements are a plain file, so they are the one place the pin is written out
    simulator = "besser/utilities/web_modeling_editor/agent_simulator/requirements.txt".replace("/", __import__("os").sep)
    assert any(simulator in f for f in found.get(BAF_VERSION, []))
