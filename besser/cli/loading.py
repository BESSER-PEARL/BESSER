"""Load the B-UML models a model file holds.

A model file is a B-UML ``.py`` script or an editor JSON export (a project or
one diagram). A ``.py`` file is Python code and is **executed**; a JSON file
is parsed as data and converted with the editor backend's converters.
"""

from __future__ import annotations

import json
import runpy
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from besser.BUML.metamodel.bpmn import BPMNModel
from besser.BUML.metamodel.deployment import DeploymentModel
from besser.BUML.metamodel.feature_model import FeatureModel
from besser.BUML.metamodel.gui import GUIModel
from besser.BUML.metamodel.kg import KnowledgeGraph
from besser.BUML.metamodel.nn import NN
from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.quantum import QuantumCircuit
from besser.BUML.metamodel.state_machine import StateMachine
from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.metamodel.uml_component import ComponentModel
from besser.BUML.metamodel.uml_deployment import DeploymentModel as UMLDeploymentModel
from besser.cli.output import stdout_to_stderr

# ``__name__`` inside a model file while it loads: not "__main__", so the part
# of a script guarded by ``if __name__ == "__main__":`` does not run.
MODEL_RUN_NAME = "__besser_model__"

# Model kind -> root class, most specific first (an Agent is a StateMachine).
# The kinds a generator reads are those of ``GeneratorSpec.models``.
MODEL_KINDS: tuple[tuple[str, type], ...] = (
    ("domain", DomainModel),
    ("object", ObjectModel),
    ("gui", GUIModel),
    ("agent", Agent),
    ("state_machine", StateMachine),
    ("quantum", QuantumCircuit),
    ("nn", NN),
    ("bpmn", BPMNModel),
    ("deployment", DeploymentModel),
    ("uml_deployment", UMLDeploymentModel),
    ("component", ComponentModel),
    ("kg", KnowledgeGraph),
    ("feature", FeatureModel),
)

# Editor project key of a single-diagram export whose type differs from it.
_PROJECT_DIAGRAM_KEYS = {"BPMNDiagram": "BPMN"}
_MAX_SCHEMA_ERRORS_SHOWN = 5


class ModelLoadError(Exception):
    """The model file could not be turned into B-UML models."""


class MissingDependencyError(ModelLoadError):
    """The editor backend's converters need packages that are not installed."""


@dataclass(frozen=True)
class LoadedModel:
    """One B-UML model of a model file."""

    # The variable holding it in a ``.py`` file, else the model's own name.
    name: str
    kind: str
    model: Any

    @property
    def label(self) -> str:
        return f"{self.name} ({type(self.model).__name__})"


@dataclass
class LoadedModels:
    """The B-UML models of one model file."""

    path: Path
    models: list[LoadedModel]
    # The file's own ``Project``, when it defines one.
    project: Project | None = None
    # Dangling references between the diagrams of an editor project.
    errors: list[str] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[LoadedModel]:
        return [loaded for loaded in self.models if loaded.kind == kind]

    def as_project(self) -> Project:
        """The file's ``Project``, or its models wrapped in one named after the file."""
        if self.project is not None:
            return self.project
        return Project(name=self.path.stem, models=[loaded.model for loaded in self.models])


def model_kind(value: Any) -> str | None:
    """The kind of B-UML model ``value`` is, or ``None`` when it is not a model."""
    for kind, model_class in MODEL_KINDS:
        if isinstance(value, model_class):
            return kind
    return None


def load_models(path: str | Path) -> LoadedModels:
    """Load the models of the ``.py`` or ``.json`` model file at ``path``.

    Raises:
        ModelLoadError: The file is missing, is neither kind, fails to load or holds no model.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"{path}: no such file")
    suffix = path.suffix.lower()
    if suffix == ".py":
        loaded = _load_python(path)
    elif suffix == ".json":
        loaded = _load_json(path)
    else:
        raise ModelLoadError(f"{path}: expected a B-UML .py file or an editor .json export")
    if not loaded.models:
        raise ModelLoadError(f"{path}: no B-UML model found")
    return loaded


def _load_python(path: Path) -> LoadedModels:
    """Execute a B-UML script and collect the models it defines.

    A ``Project`` in the file says which models it holds; without one, every
    model bound to a top-level name counts.
    """
    namespace = _execute(path)
    names = {id(value): name for name, value in reversed(list(namespace.items()))}

    projects = [value for value in namespace.values() if isinstance(value, Project)]
    if projects:
        candidates = [model for project in projects for model in project.models]
    else:
        candidates = list(namespace.values())
        nested = {id(sub) for value in candidates if isinstance(value, NN) for sub in value.sub_nns}
        candidates = [value for value in candidates if id(value) not in nested]

    models, seen = [], set()
    for value in candidates:
        kind = model_kind(value)
        if kind is None or id(value) in seen:
            continue
        seen.add(id(value))
        name = names.get(id(value)) or getattr(value, "name", None) or kind
        models.append(LoadedModel(name=name, kind=kind, model=value))
    return LoadedModels(path=path, models=models, project=projects[0] if len(projects) == 1 else None)


def _execute(path: Path) -> dict[str, Any]:
    """Run the script at ``path`` and return its top-level names.

    The file's directory is not added to ``sys.path``: the script imports
    installed packages, never a module that happens to sit beside it.
    """
    try:
        with stdout_to_stderr():
            return runpy.run_path(str(path), run_name=MODEL_RUN_NAME)
    except SystemExit as exc:
        raise ModelLoadError(f"{path}: the file exited (status {exc.code}) while loading") from exc
    except SyntaxError as exc:
        raise ModelLoadError(f"{path}: line {exc.lineno}: SyntaxError: {exc.msg}") from exc
    except Exception as exc:
        # The deepest frame in the model file itself: where the user can fix it.
        lines = [frame.lineno for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == str(path)]
        where = f"line {lines[-1]}: " if lines else ""
        raise ModelLoadError(f"{path}: {where}{type(exc).__name__}: {exc}") from exc


def _load_json(path: Path) -> LoadedModels:
    """Convert an editor export to B-UML with the editor backend's converters."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"{path}: not a JSON file: {exc}") from exc
    payload = _project_payload(data, path)

    try:
        import pydantic

        from besser.utilities.web_modeling_editor.backend.models import ProjectInput
        from besser.utilities.web_modeling_editor.backend.services.converters import json_to_buml_project
    except ModuleNotFoundError as exc:
        # A module missing from besser itself is a packaging bug: keep the traceback for that.
        if (exc.name or "").split(".")[0] in ("", "besser"):
            raise
        raise MissingDependencyError(missing_backend_dependency_message(exc)) from exc

    try:
        project_input = ProjectInput.model_validate(payload)
    except pydantic.ValidationError as exc:
        shown = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:_MAX_SCHEMA_ERRORS_SHOWN]
        ]
        raise ModelLoadError(f"{path}: not an editor project or diagram: {'; '.join(shown)}") from exc
    try:
        with stdout_to_stderr():
            project = json_to_buml_project(project_input)
    except Exception as exc:
        # The converters raise what the metamodel setters raise (ValueError), ConversionError,
        # and KeyError / TypeError / AttributeError on a malformed payload.
        raise ModelLoadError(f"{path}: cannot convert to B-UML: {type(exc).__name__}: {exc}") from exc

    models = [
        LoadedModel(name=getattr(model, "name", None) or model_kind(model), kind=model_kind(model), model=model)
        for model in project.models
    ]
    return LoadedModels(path=path, models=models, project=project, errors=list(project._cross_diagram_errors["errors"]))


def _project_payload(data: Any, path: Path) -> dict[str, Any]:
    """The editor project in ``data``: an export, a bare project, or one diagram wrapped in a project."""
    if isinstance(data, dict) and isinstance(data.get("project"), dict):
        data = data["project"]
    if isinstance(data, dict) and isinstance(data.get("diagrams"), dict):
        project = dict(data)
    elif isinstance(data, dict) and isinstance(data.get("model"), dict):
        diagram_type = data["model"].get("type") or data.get("diagramType")
        if not isinstance(diagram_type, str):
            raise ModelLoadError(f"{path}: the diagram has no type (model.type)")
        project = {"diagrams": {_PROJECT_DIAGRAM_KEYS.get(diagram_type, diagram_type): [data]}}
    else:
        raise ModelLoadError(
            f"{path}: not an editor export: expected a project (an object with 'diagrams') "
            "or a diagram (an object with 'model')"
        )
    # What the editor writes and the converters do not read.
    project.setdefault("id", path.stem)
    project.setdefault("type", "Project")
    project.setdefault("name", path.stem)
    project.setdefault("createdAt", "")
    return project


def missing_backend_dependency_message(exc: ModuleNotFoundError) -> str:
    """How to install what the editor's JSON converters need, when ``exc`` reports it missing."""
    from besser.utilities.web_modeling_editor.backend.cli import REQUIREMENTS_FILE, REQUIREMENTS_REPO_PATH

    requirements = REQUIREMENTS_FILE if REQUIREMENTS_FILE.is_file() else REQUIREMENTS_REPO_PATH
    return (
        f"editor JSON needs the Web Modeling Editor backend's packages (missing module: {exc.name!r}); "
        f"install them with: pip install -r {requirements}"
    )
