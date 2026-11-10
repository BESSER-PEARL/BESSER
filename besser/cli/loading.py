"""Load the B-UML models a model file holds.

A model file is a B-UML ``.py`` script or an editor JSON export (a project or
one diagram). A ``.py`` file is Python code and is **executed**; a JSON file
is parsed as data and converted with the editor backend's converters. ``-``
reads either from standard input.
"""

from __future__ import annotations

import json
import runpy
import sys
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

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
from besser.cli.output import plural, stdout_to_stderr

# ``__name__`` inside a model file while it loads: not "__main__", so the part
# of a script guarded by ``if __name__ == "__main__":`` does not run.
MODEL_RUN_NAME = "__besser_model__"

# The model path that reads the model file from standard input, and how messages name it.
STDIN = "-"
_STDIN_PATH = Path("<stdin>")
# What a project is called when its file gives no name.
_STDIN_PROJECT_NAME = "model"

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
# Editor diagram type -> the kind of model it converts to, in the order ``json_to_buml_project``
# converts them (its models come out in this order, one per non-empty diagram).
_CONVERTED_DIAGRAM_KINDS: tuple[tuple[str, str], ...] = (
    ("ClassDiagram", "domain"),
    ("ObjectDiagram", "object"),
    ("UserDiagram", "object"),
    ("AgentDiagram", "agent"),
    ("GUINoCodeDiagram", "gui"),
    ("QuantumCircuitDiagram", "quantum"),
    ("KnowledgeGraphDiagram", "kg"),
    ("StateMachineDiagram", "state_machine"),
    ("NNDiagram", "nn"),
    ("BPMN", "bpmn"),
    ("ComponentDiagram", "component"),
    ("DeploymentDiagram", "uml_deployment"),
)
# Diagram types the converter skips when the project has no class diagram to resolve them against.
_NEED_CLASS_DIAGRAM = frozenset({"ObjectDiagram", "GUINoCodeDiagram"})
_MAX_SCHEMA_ERRORS_SHOWN = 5


class ModelLoadError(Exception):
    """The model file could not be turned into B-UML models."""


class MissingDependencyError(ModelLoadError):
    """The editor backend's converters need packages that are not installed."""


class NoModelError(ModelLoadError):
    """The model file loads and holds no B-UML model."""


@dataclass(frozen=True)
class LoadedModel:
    """One B-UML model of a model file."""

    # The variable holding it in a ``.py`` file, else the model's own name.
    name: str
    kind: str
    model: Any
    # The editor diagram it was converted from, e.g. "ClassDiagram 1 (id 'd2')"; empty for a ``.py`` model.
    diagram: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({type(self.model).__name__})"

    @property
    def names(self) -> frozenset[str]:
        """What ``--use`` can call it: ``name`` and the model's own name."""
        return frozenset(name for name in (self.name, getattr(self.model, "name", None)) if name)


@dataclass(frozen=True)
class CurrentModel:
    """The model of the diagram an editor project marks as current for its type."""

    model: LoadedModel
    # The editor diagram type the marker is for, e.g. "ClassDiagram".
    diagram_type: str


@dataclass
class LoadedModels:
    """The B-UML models of one model file."""

    path: Path
    models: list[LoadedModel]
    # ".py" or ".json": what the file holds, whatever it is called (standard input has no name).
    suffix: str
    # The file's own ``Project``, when it defines one.
    project: Project | None = None
    # Dangling references between the diagrams of an editor project.
    errors: list[str] = field(default_factory=list)
    # Kind -> the model of the diagram the editor project marks as current (``currentDiagramIndices``).
    current: dict[str, CurrentModel] = field(default_factory=dict)
    # Kind -> why the editor project names no current model for it, for each kind it holds several of
    # and has no current one for.
    current_unresolved: dict[str, str] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list[LoadedModel]:
        return [loaded for loaded in self.models if loaded.kind == kind]

    def as_project(self) -> Project:
        """The file's ``Project``, or its models wrapped in one named after the file."""
        if self.project is not None:
            return self.project
        return Project(name=_project_name(self.path), models=[loaded.model for loaded in self.models])


def _project_name(path: Path) -> str:
    return _STDIN_PROJECT_NAME if path == _STDIN_PATH else path.stem


def model_kind(value: Any) -> str | None:
    """The kind of B-UML model ``value`` is, or ``None`` when it is not a model."""
    for kind, model_class in MODEL_KINDS:
        if isinstance(value, model_class):
            return kind
    return None


def load_models(path: str | Path) -> LoadedModels:
    """Load the models of the ``.py`` or ``.json`` model file at ``path``, or of standard input for ``-``.

    Raises:
        NoModelError: The file loads and holds no model.
        ModelLoadError: The file is missing, is neither kind or fails to load.
    """
    if str(path) == STDIN:
        path = _STDIN_PATH
        text = _read_text(path, sys.stdin.buffer.read)
        # An editor export is one JSON object; no B-UML script starts with a brace.
        suffix = ".json" if text.lstrip().startswith("{") else ".py"
        loaded = _load_json(path, text) if suffix == ".json" else _load_python(path, _execute(path, text))
    else:
        path = Path(path)
        if not path.is_file():
            raise ModelLoadError(f"{path}: no such file")
        suffix = path.suffix.lower()
        if suffix == ".py":
            loaded = _load_python(path, _execute(path))
        elif suffix == ".json":
            loaded = _load_json(path, _read_text(path, path.read_bytes))
        else:
            raise ModelLoadError(f"{path}: expected a B-UML .py file or an editor .json export")
    if not loaded.models:
        raise NoModelError(f"{path}: no B-UML model found")
    return loaded


def _read_text(path: Path, read: Callable[[], bytes]) -> str:
    try:
        return read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelLoadError(f"{path}: cannot be read as UTF-8 text: {exc}") from exc


def _load_python(path: Path, namespace: dict[str, Any]) -> LoadedModels:
    """Collect the models among ``namespace``, the top-level names of an executed B-UML script.

    A ``Project`` in the file says which models it holds; without one, every
    model bound to a top-level name counts.
    """
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
    return LoadedModels(
        path=path, models=models, suffix=".py", project=projects[0] if len(projects) == 1 else None,
    )


def _execute(path: Path, source: str | None = None) -> dict[str, Any]:
    """Run the script at ``path``, or ``source`` read from standard input, and return its top-level names.

    The file's directory is not added to ``sys.path``: the script imports
    installed packages, never a module that happens to sit beside it.
    """
    try:
        with stdout_to_stderr():
            if source is None:
                return runpy.run_path(str(path), run_name=MODEL_RUN_NAME)
            namespace: dict[str, Any] = {"__name__": MODEL_RUN_NAME}
            exec(compile(source, str(path), "exec"), namespace)  # as run_path does with a file
            return namespace
    except SystemExit as exc:
        raise ModelLoadError(f"{path}: the file exited (status {exc.code}) while loading") from exc
    except SyntaxError as exc:
        raise ModelLoadError(f"{path}: line {exc.lineno}: SyntaxError: {exc.msg}") from exc
    except Exception as exc:
        # The deepest frame in the model file itself: where the user can fix it.
        lines = [frame.lineno for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == str(path)]
        where = f"line {lines[-1]}: " if lines else ""
        raise ModelLoadError(f"{path}: {where}{type(exc).__name__}: {exc}") from exc


def _load_json(path: Path, text: str) -> LoadedModels:
    """Convert an editor export to B-UML with the editor backend's converters."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
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
    positions, problems = _diagram_positions(project_input, models)
    for (diagram_type, index), position in positions.items():
        diagram_id = project_input.diagrams[diagram_type][index].id
        models[position] = replace(models[position], diagram=f"{diagram_type} {index} (id {diagram_id!r})")
    current, unresolved = _current_models(project_input, models, positions, problems)
    return LoadedModels(
        path=path, models=models, suffix=".json", project=project,
        errors=list(project._cross_diagram_errors["errors"]), current=current, current_unresolved=unresolved,
    )


def _diagram_positions(
    project_input: Any, models: list[LoadedModel],
) -> tuple[dict[tuple[str, int], int], dict[str, str]]:
    """(diagram type, index in the project) -> the position in ``models`` of the model converted from it,
    and kind -> why the diagrams of that kind cannot be told apart by their model.

    ``json_to_buml_project`` converts the non-empty diagrams type by type, in the order of
    ``_CONVERTED_DIAGRAM_KINDS``, so its models line up with them. When they do not, no
    position is known and every kind gets a reason. The converter caches a class diagram's
    model by the diagram's id, so two diagrams sharing an id give one model twice: no
    model of such a kind is taken for a given diagram's.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.project_converter import (
        _is_valid_diagram,
    )

    def valid(diagram_type: str) -> list[tuple[int, Any]]:
        return [(index, diagram) for index, diagram in enumerate(project_input.diagrams.get(diagram_type, []))
                if _is_valid_diagram(diagram, diagram_type)]

    kinds = dict.fromkeys(kind for _, kind in _CONVERTED_DIAGRAM_KINDS)
    has_class_diagram = bool(valid("ClassDiagram"))
    produced = iter(enumerate(models))
    positions: dict[tuple[str, int], int] = {}
    for diagram_type, kind in _CONVERTED_DIAGRAM_KINDS:
        if diagram_type in _NEED_CLASS_DIAGRAM and not has_class_diagram:
            continue
        for index, _ in valid(diagram_type):
            position, loaded = next(produced, (None, None))
            if loaded is None or loaded.kind != kind:
                return {}, dict.fromkeys(kinds, "its models do not line up with its diagrams")
            positions[diagram_type, index] = position

    problems: dict[str, str] = {}
    for diagram_type, kind in _CONVERTED_DIAGRAM_KINDS:
        ids = [diagram.id for diagram in project_input.diagrams.get(diagram_type, []) if diagram.id]
        shared = next((diagram_id for diagram_id in ids if ids.count(diagram_id) > 1), None)
        if shared is not None:
            problems.setdefault(kind, f"two of its {diagram_type} diagrams share the id {shared!r}")
    seen: set[int] = set()
    for position in positions.values():
        loaded = models[position]
        if id(loaded.model) in seen:
            problems.setdefault(loaded.kind, f"two of its diagrams convert to one {loaded.kind} model")
        seen.add(id(loaded.model))
    return positions, problems


def _current_models(
    project_input: Any, models: list[LoadedModel], positions: dict[tuple[str, int], int], problems: dict[str, str],
) -> tuple[dict[str, CurrentModel], dict[str, str]]:
    """The model of each kind's current diagram, and why the project names none for some kinds.

    Only kinds with several models need one, so only they get a reason. When the diagrams
    of two types make models of one kind (object and user diagrams), the project's
    ``currentDiagramType`` says which type's current diagram counts.
    """
    indices = project_input.currentDiagramIndices or {}
    current: dict[str, CurrentModel] = {}
    unresolved: dict[str, str] = {}
    for kind in dict.fromkeys(kind for _, kind in _CONVERTED_DIAGRAM_KINDS):
        if sum(model.kind == kind for model in models) < 2:
            continue
        if kind in problems:
            unresolved[kind] = problems[kind]
            continue
        types = [diagram_type for diagram_type, of in _CONVERTED_DIAGRAM_KINDS
                 if of == kind and project_input.diagrams.get(diagram_type)]
        marked, reasons = [], []
        for diagram_type in (diagram_type for diagram_type in types if diagram_type in indices):
            index = indices[diagram_type]
            diagrams = project_input.diagrams[diagram_type]
            marker = f"its currentDiagramIndices.{diagram_type} is {index}"
            if not 0 <= index < len(diagrams):
                reasons.append(f"{marker} and it holds {plural(len(diagrams), diagram_type + ' diagram')}")
            elif (diagram_type, index) not in positions:
                title = diagrams[index].title or diagrams[index].id
                reasons.append(f"{marker}, an empty diagram ({title!r})")
            else:
                marked.append(CurrentModel(models[positions[diagram_type, index]], diagram_type))
        if len(marked) > 1:
            chosen = [candidate for candidate in marked if candidate.diagram_type == project_input.currentDiagramType]
            if len(chosen) != 1:
                reasons = [f"it marks a current {' and '.join(c.diagram_type for c in marked)} "
                           "and its currentDiagramType is neither"]
            marked = chosen
        if len(marked) == 1:
            current[kind] = marked[0]
        else:
            unresolved[kind] = "; ".join(reasons) or f"its currentDiagramIndices has no {' or '.join(types)} entry"
    return current, unresolved


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
    project.setdefault("id", _project_name(path))
    project.setdefault("type", "Project")
    project.setdefault("name", _project_name(path))
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
