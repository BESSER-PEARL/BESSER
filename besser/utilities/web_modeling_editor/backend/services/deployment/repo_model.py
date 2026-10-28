"""The BESSER files inside a project's GitHub repository.

One layout for every path that writes or reads a repo (the GitHub sidebar
save, the Spec-Driven push, continue-from-GitHub, the repo browser):

- ``buml/diagrams.json``: the project's model, a V2 project-export envelope.
  It is the only model file; ``besser-project.json`` is the legacy name the
  sidebar used and is still read.
- ``buml/*.py``: the same model as B-UML Python source, for people reading the
  repo. Never read back.
- ``.besser/project.json``: which BESSER project the repo belongs to, so any
  browser that opens the repo finds its project again.

Commits BESSER makes carry a ``BESSER-Origin`` trailer, which is how the
history tells the agent's and the editor's commits from a person's.
"""

import json
import logging
import os
from typing import Optional

from besser.generators.web_app.web_app_generator import agent_slug
from besser.spec_driven_agent.planning.model_change import describe_model_change, model_summary
from besser.utilities.buml_code_builder import (
    agent_model_to_code,
    domain_model_to_code,
    gui_model_to_code,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import BUML_EMIT_IDS
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.spec_driven.model_assembly import (
    assemble_models_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import redact_credentials

logger = logging.getLogger(__name__)

MODEL_DIR = "buml"
MODEL_FILE = f"{MODEL_DIR}/diagrams.json"
LEGACY_PROJECT_FILE = "besser-project.json"
LINK_FILE = ".besser/project.json"
LINK_SCHEMA_VERSION = 1
# GitHub topic set on every repo BESSER pushes, so the repo browser can list
# a user's BESSER repos without opening each one.
BESSER_TOPIC = "besser"
ORIGIN_TRAILER = "BESSER-Origin"
ORIGIN_AGENT = "spec-driven-agent"
ORIGIN_EDITOR = "editor"
# The agent diagram's config.yaml text (frontend ``ProjectDiagram``, shared/types/project.ts).
AGENT_CONFIG_YAML_FIELDS = ("configYaml", "agentConfigCustomYaml")


def with_origin(message: str, origin: str) -> str:
    """``message`` with the ``BESSER-Origin`` trailer that marks a BESSER commit."""
    return f"{message.rstrip()}\n\n{ORIGIN_TRAILER}: {origin}"


def commit_origin(message: str) -> Optional[str]:
    """The ``BESSER-Origin`` of a commit message, or None for a person's commit."""
    prefix = f"{ORIGIN_TRAILER}:"
    for line in reversed(message.splitlines()):
        if line.startswith(prefix):
            return line[len(prefix):].strip() or None
    return None


def project_of(export: dict) -> dict:
    """The project inside a V2 export envelope, or ``export`` itself when it is a bare project."""
    inner = export.get("project")
    return inner if isinstance(inner, dict) else export


def link_record(export: dict) -> dict:
    """The ``.besser/project.json`` content for ``export``."""
    project = project_of(export)
    return {
        "schemaVersion": LINK_SCHEMA_VERSION,
        "projectId": project.get("id"),
        "projectName": project.get("name"),
        "model": MODEL_FILE,
    }


def model_files(project_export: Optional[dict]) -> dict[str, bytes]:
    """Every BESSER model file for ``project_export``, as repo path -> content.

    The B-UML ``.py`` exports are best effort: a model that fails to build
    still gets its ``diagrams.json`` and link file, with a logged warning.
    """
    if not isinstance(project_export, dict) or not project_export:
        logger.warning("No project export given; the repo gets no BESSER model files")
        return {}

    # Agent diagrams carry the agent's runtime config (LLM keys, bot tokens, DB passwords).
    project_export, findings = redact_credentials(project_export, yaml_fields=AGENT_CONFIG_YAML_FIELDS)
    if findings:
        logger.warning("Redacted %d credential(s) from the model before writing it to the repo", findings)

    files: dict[str, bytes] = {
        MODEL_FILE: json.dumps(project_export, indent=2, default=str).encode("utf-8"),
        LINK_FILE: json.dumps(link_record(project_export), indent=2).encode("utf-8"),
    }
    try:
        files.update(_buml_sources(project_export))
    except Exception:
        logger.warning("Could not export the B-UML model files; continuing without them", exc_info=True)
    return files


def _buml_sources(project_export: dict) -> dict[str, bytes]:
    import tempfile

    assembled = assemble_models_from_project(ProjectInput.model_validate(project_of(project_export)))
    writers = []
    if assembled.domain_model is not None:
        writers.append(("domain_model.py", lambda path: domain_model_to_code(
            model=assembled.domain_model, file_path=path, emit_ids=BUML_EMIT_IDS)))
    if assembled.gui_model is not None:
        writers.append(("gui_model.py", lambda path: gui_model_to_code(
            model=assembled.gui_model, file_path=path, domain_model=assembled.domain_model)))
    if assembled.agent_model is not None:
        writers.append((f"agent_model_{agent_slug(assembled.agent_model.name)}.py",
                        lambda path: agent_model_to_code(assembled.agent_model, path)))

    sources: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="besser_") as scratch:
        for name, write in writers:
            path = os.path.join(scratch, name)
            write(path)
            with open(path, "rb") as fh:
                sources[f"{MODEL_DIR}/{name}"] = fh.read()
    return sources


def write_model_files(workdir: str, project_export: Optional[dict]) -> None:
    """Write :func:`model_files` into ``workdir`` (a tree about to be pushed)."""
    for repo_path, content in model_files(project_export).items():
        target = os.path.join(workdir, *repo_path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(content)


def summarize(project_export: dict) -> dict:
    """What a repo card shows about a model: name, diagrams per type, class count."""
    project = project_of(project_export)
    diagrams = project.get("diagrams") or {}
    counts = {
        kind: len(value) if isinstance(value, list) else 1
        for kind, value in diagrams.items()
        if value
    }
    classes = _domain_summary(project_export).get("classes", {})
    return {
        "project_name": project.get("name") or "",
        "project_id": project.get("id"),
        "diagram_counts": counts,
        "class_count": len(classes),
        "class_names": sorted(classes)[:12],
    }


def _domain_summary(project_export: dict) -> dict:
    """``model_summary`` of the export's class diagram, or ``{}`` when it has none."""
    try:
        assembled = assemble_models_from_project(ProjectInput.model_validate(project_of(project_export)))
    except Exception:
        logger.warning("Could not build the domain model of a repo's diagrams.json", exc_info=True)
        return {}
    if assembled.domain_model is None:
        return {}
    return model_summary(assembled.domain_model)


def describe_change(old_export: Optional[dict], new_export: dict) -> list[str]:
    """One line per class-diagram change between two versions of ``diagrams.json``."""
    new = _domain_summary(new_export)
    if old_export is None:
        return [f"class {name} added" for name in sorted(new.get("classes", {}))]
    return describe_model_change(_domain_summary(old_export), new)
