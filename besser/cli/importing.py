"""Read what ``besser convert`` imports: files that are not B-UML and that BESSER turns into a model.

Each format is read as data by the importer BESSER already has for it: nothing here is
executed, and no importer calls a language model or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from besser.cli.loading import LoadedModel, LoadedModels, MissingDependencyError, ModelLoadError
from besser.cli.loading import missing_backend_dependency_message

# What an importer returns: the model and what it could not take from the source.
_Imported = tuple[Any, list[str]]


@dataclass(frozen=True)
class ImportFormat:
    """One format ``convert`` reads besides B-UML ``.py`` and editor JSON."""

    # Its name for ``--from``.
    name: str
    # The file suffixes that mean this format without ``--from`` (none: it must be named).
    suffixes: tuple[str, ...]
    # The kind of model it becomes (see ``loading.MODEL_KINDS``).
    kind: str
    summary: str
    read: Callable[[Sequence[Path]], _Imported]
    # It reads a directory, not a file.
    directory: bool = False
    # It reads several files into one model.
    several: bool = False


def _syntax_errors(path: Path, parse: Callable[[Any], Any]) -> Any:
    """``parse(listener)`` with an ANTLR error listener, failing on the file's syntax errors.

    ANTLR's default listener prints them and goes on with what it could recover.
    """
    from antlr4.error.ErrorListener import ErrorListener

    errors: list[str] = []

    class _Collect(ErrorListener):
        def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):  # noqa: N802 (ANTLR's name)
            errors.append(f"line {line}, column {column + 1}: {msg}")

    result = parse(_Collect())
    if errors:
        shown = "; ".join(errors[:_MAX_SYNTAX_ERRORS_SHOWN])
        more = len(errors) - _MAX_SYNTAX_ERRORS_SHOWN
        raise ModelLoadError(f"{path}: {shown}{f' (and {more} more)' if more > 0 else ''}")
    return result


# A file with one mistake often reports many: the first ones are where to look.
_MAX_SYNTAX_ERRORS_SHOWN = 5


def _read_plantuml(paths: Sequence[Path]) -> _Imported:
    from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker

    from besser.BUML.notations.structuralPlantUML.plantUML_buml_listener import BUMLGenerationListener
    from besser.BUML.notations.structuralPlantUML.PlantUMLLexer import PlantUMLLexer
    from besser.BUML.notations.structuralPlantUML.PlantUMLParser import PlantUMLParser

    path = paths[0]

    def parse(error_listener: Any) -> Any:
        lexer = PlantUMLLexer(FileStream(str(path), encoding="utf-8"))
        parser = PlantUMLParser(CommonTokenStream(lexer))
        for recognizer in (lexer, parser):
            recognizer.removeErrorListeners()
            recognizer.addErrorListener(error_listener)
        return parser.domainModel()

    tree = _syntax_errors(path, parse)
    listener = BUMLGenerationListener()
    ParseTreeWalker().walk(listener, tree)
    return listener.get_buml_model(), []


def _read_drawio(paths: Sequence[Path]) -> _Imported:
    from besser.BUML.notations.structuralDrawIO.structural_drawio_to_buml import structural_drawio_to_buml

    return structural_drawio_to_buml(str(paths[0])), []


def _read_csv(paths: Sequence[Path]) -> _Imported:
    try:
        from besser.utilities.web_modeling_editor.backend.services.reverse_engineering.csv_reverse import (
            csv_to_domain_model,
        )
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("", "besser"):
            raise
        raise MissingDependencyError(missing_backend_dependency_message(exc)) from exc

    return csv_to_domain_model([str(path) for path in paths], model_name=paths[0].stem), []


def _read_rdf(paths: Sequence[Path]) -> _Imported:
    from besser.BUML.notations.kg_rdf import owl_file_to_knowledge_graph

    skipped: list[dict[str, str]] = []
    graph = owl_file_to_knowledge_graph(str(paths[0]), warnings=skipped)
    return graph, [entry["message"] for entry in skipped]


def _read_sdmx(paths: Sequence[Path]) -> _Imported:
    from besser.utilities.sdmx_to_buml import sdmx_xml_to_buml

    return sdmx_xml_to_buml(paths[0].read_bytes()), []


def _read_code(paths: Sequence[Path]) -> _Imported:
    try:
        from besser.utilities.web_modeling_editor.backend.services.reverse_engineering.code_reverse import (
            analyze_repository,
        )
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in ("", "besser"):
            raise
        raise MissingDependencyError(missing_backend_dependency_message(exc)) from exc

    analysis, model = analyze_repository(str(paths[0]))
    if model is None:
        raise ModelLoadError(
            f"{paths[0]}: no data classes found: the reader takes SQLAlchemy models, Django models, "
            "Pydantic models and dataclasses from Python sources"
        )
    return model, [f"relationship left out: {skipped}" for skipped in analysis.skipped_relationships]


IMPORT_FORMATS: tuple[ImportFormat, ...] = (
    ImportFormat("plantuml", (".plantuml", ".puml", ".pu"), "domain",
                 "a PlantUML class diagram", _read_plantuml),
    ImportFormat("drawio", (".drawio",), "domain",
                 "a draw.io (diagrams.net) UML class diagram", _read_drawio),
    ImportFormat("csv", (".csv",), "domain",
                 "CSV files with a header row: a class per file, an association per id column", _read_csv,
                 several=True),
    ImportFormat("rdf", (".ttl", ".owl", ".rdf", ".nt", ".n3"), "kg",
                 "an OWL / RDF / SHACL file, as a knowledge graph", _read_rdf),
    ImportFormat("sdmx", (), "domain",
                 "an SDMX-ML 2.1 structure message (a downloaded .xml file)", _read_sdmx),
    ImportFormat("code", (), "domain",
                 "a directory of Python code: its SQLAlchemy, Django or Pydantic models", _read_code,
                 directory=True),
)


def import_format(name: str) -> ImportFormat | None:
    """The import format called ``name``, or ``None``."""
    return next((fmt for fmt in IMPORT_FORMATS if fmt.name == name), None)


def format_of(path: Path) -> ImportFormat | None:
    """The import format the suffix of ``path`` means, or ``None``."""
    suffix = path.suffix.lower()
    return next((fmt for fmt in IMPORT_FORMATS if suffix in fmt.suffixes), None)


def import_models(fmt: ImportFormat, inputs: Sequence[str]) -> tuple[LoadedModels, list[str]]:
    """Read ``inputs`` as ``fmt`` and return the model, as one loaded model file, and the import's warnings.

    Raises:
        ModelLoadError: An input is missing or of the wrong kind, does not parse, or holds no model.
    """
    paths = [Path(value) for value in inputs]
    if len(paths) > 1 and not fmt.several:
        raise ModelLoadError(f"--from {fmt.name} reads one {'directory' if fmt.directory else 'file'}, "
                             f"got {len(paths)}")
    for path in paths:
        if fmt.directory and not path.is_dir():
            raise ModelLoadError(f"{path}: --from {fmt.name} reads a directory")
        if not fmt.directory and not path.is_file():
            raise ModelLoadError(f"{path}: no such file")
    try:
        model, warnings = fmt.read(paths)
    except ModelLoadError:
        raise
    except Exception as exc:
        # The importers raise what their parsers and the metamodel setters raise.
        raise ModelLoadError(f"{paths[0]}: cannot be read as {fmt.name}: {type(exc).__name__}: {exc}") from exc
    if model is None or (fmt.kind == "domain" and not (model.get_classes() or model.get_enumerations())):
        raise ModelLoadError(f"{paths[0]}: no model found: it holds nothing the {fmt.name} import reads")
    name = getattr(model, "name", None) or fmt.kind
    loaded = LoadedModels(path=paths[0], models=[LoadedModel(name=name, kind=fmt.kind, model=model)],
                          suffix=fmt.name)
    return loaded, warnings
