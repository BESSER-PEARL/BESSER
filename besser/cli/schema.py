"""What ``besser schema`` prints: the formats the command reads and writes, for whoever writes them.

A model kind is described from the metamodel itself (its classes and their constructor
arguments, read with ``inspect``), so the description cannot drift from the code.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

from besser.cli.loading import _CONVERTED_DIAGRAM_KINDS, MODEL_KINDS
from besser.cli.output import EXIT_FAILED, EXIT_OK, EXIT_USAGE
from besser.generators.registry import GENERATORS

# Model kind -> the package its classes are imported from.
KIND_PACKAGES: dict[str, str] = {
    "domain": "besser.BUML.metamodel.structural",
    "object": "besser.BUML.metamodel.object",
    "gui": "besser.BUML.metamodel.gui",
    "agent": "besser.BUML.metamodel.state_machine.agent",
    "state_machine": "besser.BUML.metamodel.state_machine",
    "quantum": "besser.BUML.metamodel.quantum",
    "nn": "besser.BUML.metamodel.nn",
    "bpmn": "besser.BUML.metamodel.bpmn",
    "deployment": "besser.BUML.metamodel.deployment",
    "uml_deployment": "besser.BUML.metamodel.uml_deployment",
    "component": "besser.BUML.metamodel.uml_component",
    "kg": "besser.BUML.metamodel.kg",
    "feature": "besser.BUML.metamodel.feature_model",
}
PROJECT_PACKAGE = "besser.BUML.metamodel.project"

# A dotted path in a signature ("besser.BUML.metamodel.structural.structural.Property"): its last name is enough.
_DOTTED_NAME = re.compile(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)")


def editor_diagram_type(kind: str) -> str | None:
    """The editor diagram a model of ``kind`` is drawn as, or ``None`` when the editor has none."""
    return next((diagram_type for diagram_type, of in _CONVERTED_DIAGRAM_KINDS if of == kind), None)


def _short(annotation: str) -> str:
    return _DOTTED_NAME.sub(r"\1", annotation)


def _parameters(cls: type) -> list[dict[str, Any]]:
    """The constructor's arguments, in order: name, type and default as written in the metamodel."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):  # no Python signature (a class without its own constructor)
        return []
    parameters = []
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        entry: dict[str, Any] = {"name": parameter.name, "required": parameter.default is parameter.empty}
        if parameter.annotation is not parameter.empty:
            annotation = parameter.annotation
            entry["type"] = _short(annotation if isinstance(annotation, str) else inspect.formatannotation(annotation))
        if parameter.default is not parameter.empty:
            entry["default"] = _short(repr(parameter.default))
        parameters.append(entry)
    return parameters


def _signature(name: str, parameters: list[dict[str, Any]]) -> str:
    arguments = []
    for parameter in parameters:
        text = parameter["name"]
        if "type" in parameter:
            text += f": {parameter['type']}"
        if "default" in parameter:
            text += f" = {parameter['default']}" if "type" in parameter else f"={parameter['default']}"
        arguments.append(text)
    return f"{name}({', '.join(arguments)})"


def _first_sentence(obj: Any) -> str:
    """The first paragraph of the docstring, on one line."""
    doc = inspect.getdoc(obj) or ""
    return " ".join(re.split(r"\n\s*\n|\n\s*Args:", doc, maxsplit=1)[0].split())


def describe_package(package: str) -> dict[str, Any]:
    """The public classes a metamodel package defines and the values it exports (``StringType``, ...)."""
    module = importlib.import_module(package)
    # A package that exports nothing at its top level (the agent metamodel) is described from its own module.
    root = package.rsplit(".", 1)[0] if not hasattr(module, "__path__") else package
    classes, values = [], []
    for name, value in vars(module).items():
        if name.startswith("_") or inspect.ismodule(value):
            continue
        defined_in = getattr(value, "__module__", None) or ""
        if inspect.isclass(value):
            if not defined_in.startswith(root):
                continue
            parameters = _parameters(value)
            classes.append({
                "name": name,
                "signature": _signature(name, parameters),
                "doc": _first_sentence(value),
                "bases": [base.__name__ for base in value.__bases__ if base is not object],
                "abstract": inspect.isabstract(value),
                "parameters": parameters,
            })
        elif type(value).__module__.startswith("besser.") or (name.isupper() and name != "TYPE_CHECKING"):
            if not callable(value):
                values.append({"name": name, "type": type(value).__name__})
    return {"import": package, "classes": classes, "values": values}


def describe_kind(kind: str) -> dict[str, Any]:
    """A model kind: its root class, where its classes come from, and the generators that read it."""
    root = dict(MODEL_KINDS)[kind]
    return {
        "kind": kind,
        "root_class": root.__name__,
        "doc": _first_sentence(root),
        "editor_diagram": editor_diagram_type(kind),
        "generators": [spec.id for spec in GENERATORS if kind in spec.models],
        **describe_package(KIND_PACKAGES[kind]),
    }


def kind_text(entry: dict[str, Any]) -> list[str]:
    """The text report of ``describe_kind`` (or of the project's description)."""
    lines = [f"{entry['kind']}: a {entry['root_class']}, from {entry['import']}"]
    if entry["doc"]:
        lines.append(f"  {entry['doc']}")
    lines.append(f"  a starter to edit: besser new model.py --kind {entry['kind']}")
    if entry.get("editor_diagram"):
        lines.append(f"  in editor JSON: a {entry['editor_diagram']} of the project ('besser schema project'); "
                     f"besser new model.json --kind {entry['kind']} writes one")
    elif "editor_diagram" in entry:
        lines.append("  in editor JSON: none; the editor has no diagram for it, so it is a .py file only")
    if "generators" in entry:
        lines.append(f"  generators that read it: {', '.join(entry['generators']) or 'none'}")
    lines.extend(["", f"Classes (from {entry['import']} import ...), with their constructor arguments:"])
    for cls in entry["classes"]:
        lines.append(f"  {cls['signature']}")
        if cls["doc"]:
            lines.append(f"      {cls['doc']}")
    if entry["values"]:
        lines.extend(["", "Values: " + ", ".join(value["name"] for value in entry["values"])])
    return lines


# ── the JSON output of the commands ─────────────────────────────────────────

_STRINGS = {"type": "array", "items": {"type": "string"}}
_WOULD_WRITE = {"type": "object", "additionalProperties": {"type": "string"},
                "description": "with --dry-run: path -> what the run would do to it"}
_MODELS_READ = {"type": "object", "additionalProperties": {"type": "string"},
                "description": "model kind -> the name of the model read"}
_REPORT_LISTS = ("written", "merged", "conflicted", "deleted", "kept_removed_from_model", "not_restored",
                 "not_merged", "differs_from_model")

# Command -> the keys it adds to the object every command prints.
COMMAND_KEYS: dict[str, dict[str, Any]] = {
    "version": {"version": {"type": "string"}},
    "new": {
        "model": {"type": "string", "description": "the file written"},
        "kind": {"type": "string"},
        "kinds": {"type": "array", "description": "with --list: per kind its kind, summary, models and generator"},
    },
    "schema": {
        "topic": {"type": "string"},
        "topics": {"type": "array", "description": "without a topic: the topics there are"},
        "schema": {"type": "object", "description": "topics project and result: a JSON Schema"},
        "description": {"type": "object", "description": "a model kind: its classes, values and generators"},
        "example": {"type": "string", "description": "topic config: an example file"},
    },
    "validate": {
        "model_files": {**_STRINGS, "description": "the files checked, directories expanded"},
        "models": {"type": "array", "description": "per model: file, name, kind, type, validated, errors, warnings"},
        "results": {"type": "array", "description": "per file: file, ok, exit_code, errors, warnings, models"},
        "skipped": {"type": "array", "description": "per file a directory search left out: file, reason"},
        "diagnostics": {
            "type": "array",
            "description": "every error and warning with where it is",
            "items": {
                "type": "object",
                "required": ["severity", "file", "message"],
                "properties": {
                    "severity": {"enum": ["error", "warning"]},
                    "file": {"type": "string"},
                    "model": {"type": "string", "description": "the model's name; absent for a file that "
                                                               "does not load and for a project's own errors"},
                    "message": {"type": "string"},
                    "line": {"type": "integer", "description": "the line of a .py file where the element the "
                                                               "message names is defined, when it is found"},
                },
            },
        },
    },
    "info": {
        "model": {"type": "string"},
        "models": {"type": "array", "description": "per model: name, kind, type, model_name, current, counts, "
                                                   "elements; a domain model's classes, enumerations, associations"},
        "generators": {**_STRINGS, "description": "the ids of the generators that can run on the file"},
    },
    "generators": {
        "generators": {"type": "array", "description": "per generator: id, summary, category, models, options "
                                                       "(JSON Schema properties), supports_regenerate"},
    },
    "generate": {
        "generator": {"type": "string"}, "model": {"type": "string"}, "output_dir": {"type": "string"},
        "dry_run": {"type": "boolean"},
        "recorded": {"type": "boolean", "description": "whether .besser_generated.json was (or would be) written"},
        "used": _MODELS_READ, "would_write": _WOULD_WRITE,
    },
    "regenerate": {
        "generator": {"type": "string"}, "model": {"type": "string"}, "output_dir": {"type": "string"},
        "dry_run": {"type": "boolean"}, "adopted": {"type": "boolean"},
        "used": _MODELS_READ, "would_write": _WOULD_WRITE,
        **{name: _STRINGS for name in _REPORT_LISTS},
    },
    "build": {
        "config": {"type": "string", "description": "the configuration file read"},
        "dry_run": {"type": "boolean"},
        "outputs": {"type": "array", "description": "per output built: its name and the object of the generate "
                                                    "or regenerate run that built it"},
    },
    "convert": {
        "input": {"type": "string"}, "inputs": _STRINGS, "output": {"type": "string"},
        "dry_run": {"type": "boolean"}, "would_write": _WOULD_WRITE,
        "models": {"type": "array", "description": "per model written: name, kind, type"},
    },
    "completion": {"shell": {"type": "string"}, "script": {"type": "string"}},
}


def result_schema() -> dict[str, Any]:
    """The JSON Schema of the object a command prints with ``--json``."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "besser --json output",
        "description": "One object on standard output per run. The first seven keys are on every command; "
                       "each command adds its own.",
        "type": "object",
        "required": ["ok", "command", "exit_code", "errors", "warnings", "notes", "files"],
        "properties": {
            "ok": {"type": "boolean", "description": "true when exit_code is 0"},
            "command": {"type": "string", "description": "the command that ran ('version' for --version); empty "
                                                         "when the arguments name none"},
            "exit_code": {
                "enum": [EXIT_OK, EXIT_FAILED, EXIT_USAGE],
                "description": "0 success; 1 the command ran and failed (validation errors, a generator that "
                               "raised, conflicts); 2 it could not run (bad arguments, a model that does not load, "
                               "a refused output directory)",
            },
            "errors": {**_STRINGS, "description": "what to fix"},
            "warnings": {**_STRINGS, "description": "never make ok false, except with --strict"},
            "notes": {**_STRINGS, "description": "what the command chose on its own"},
            "files": {**_STRINGS, "description": "the files written; empty with --dry-run"},
        },
        "allOf": [
            {"if": {"properties": {"command": {"const": command}}}, "then": {"properties": keys}}
            for command, keys in COMMAND_KEYS.items()
        ],
    }


def project_schema() -> dict[str, Any]:
    """The JSON Schema of an editor project, from the model the editor backend validates it with.

    Raises:
        ModuleNotFoundError: The editor backend's packages are not installed.
    """
    from besser.utilities.web_modeling_editor.backend.models import ProjectInput

    schema = ProjectInput.model_json_schema()
    schema["title"] = "BESSER Web Modeling Editor project"
    schema["description"] = (
        "A project as the editor exports it. A model file is this object, or an export wrapping it as "
        "{\"project\": {...}, \"exportedAt\": ..., \"version\": ...}, or one diagram alone ({\"title\", "
        "\"model\"}). 'diagrams' maps a diagram type to its diagrams; a diagram's 'model' holds 'type', "
        "'elements' and 'relationships', by id. 'besser new model.json --kind KIND' writes an example of each."
    )
    return schema
