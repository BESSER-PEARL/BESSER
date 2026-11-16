"""Trace links from models and their elements to the code a generator renders from them.

Every generator yields file-level links for free: :func:`collect_generated_files` reads
the files a run wrote, and :func:`build_generation_trace` links each input model to each
of them. A generator that is *instrumented* records, on top of that, which lines come
from which element.

A template marks the block it renders for an element::

    {% trace class, "class" %}
    class {{ class.name }}(Base):
        __tablename__ = "{{ class.name.lower() }}"
    {% endtrace %}

:class:`TraceExtension` (in the environment :func:`traced_environment` makes) wraps the
rendered block in invisible markers that carry the element's persistent id.
:func:`extract_trace_regions` removes them and returns the line range of each block, so
the generated file is byte-identical to an untraced render. A file generated as a whole
from one element (a class's router) names that element instead (:func:`trace_source`).

Every generator has ``traced_files`` (``GeneratorInterface``): for each file it wrote
(by path, relative to its output directory), a :class:`FileTrace`. A generator records
it by writing its renders with ``GeneratorInterface._write_traced``; one built from
others merges theirs (``GeneratorInterface._merge_traced_files``).
:func:`build_generation_trace` turns the files of one generation run into a
:class:`~besser.BUML.metamodel.traceability.TraceModel` fragment.

An element without an ``id`` (a model built in Python rather than in the editor) cannot be
referenced from a trace, so its block is rendered untraced.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from jinja2 import BaseLoader, Environment, FileSystemLoader, nodes
from jinja2.ext import Extension
from markupsafe import Markup

from besser.BUML.metamodel.structural import Constraint, Model, NamedElement
from besser.BUML.metamodel.traceability import (
    CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ExecutionStatus, ModelArtifact, Person,
    RuleEvidence, ToolCall, TraceLink, TraceModel,
)
from besser.BUML.metamodel.traceability.traceability import model_elements

logger = logging.getLogger(__name__)

# Control characters never emitted by a template; json.dumps escapes them inside the payload.
_START, _END = "\x02", "\x03"
_CLOSE = f"{_START}/trace{_END}"
_MARKER = re.compile(f"{_START}(/trace|trace:[^{_END}]*){_END}")


@dataclass(frozen=True)
class ElementRef:
    """A model element as a trace records it: its persistent id, metaclass and name."""
    element_id: str
    element_type: str
    element_name: str


# Specialisations recorded under the metaclass the editor names them by: an OCL constraint parsed
# from the editor is an ``OCLConstraint``, but the editor records the constraint box as a ``Constraint``.
_TRACED_AS = (Constraint,)


def element_ref(element) -> Optional[ElementRef]:
    """The reference of a B-UML element, or None when it has no ``id`` (it cannot be traced)."""
    element_id = getattr(element, "id", None)
    if not element_id:
        return None
    name = element.name if isinstance(element, NamedElement) else element_id
    metaclass = next((base for base in _TRACED_AS if isinstance(element, base)), type(element))
    return ElementRef(element_id, metaclass.__name__, name)


@dataclass(frozen=True)
class TraceRegion:
    """The lines of a generated file rendered from one model element by one template rule.

    Lines are 1-based and inclusive. ``owner`` is the element's owner when it has one (a
    method's class), so a trace can nest the element under it.
    """
    element_id: str
    element_type: str
    element_name: str
    rule: str
    start_line: int
    end_line: int
    owner: Optional[ElementRef] = None


@dataclass(frozen=True)
class TraceSource:
    """A model element a whole file was generated from (a class's router), by a rule."""
    element: ElementRef
    rule: str


def trace_source(element, rule: str) -> Optional[TraceSource]:
    """The source of a whole file, or None when the element has no ``id``."""
    ref = element_ref(element)
    return TraceSource(ref, rule) if ref else None


@dataclass(frozen=True)
class FileTrace:
    """What one generated file was rendered from: the line ranges of its traced blocks, and the
    elements it was generated from as a whole."""
    regions: Tuple[TraceRegion, ...] = ()
    sources: Tuple[TraceSource, ...] = ()


def shift_file_trace(trace: FileTrace, lines: int) -> FileTrace:
    """``trace`` with its regions moved down by ``lines``, for a header added above the file."""
    return FileTrace(tuple(shift_regions(trace.regions, lines)), trace.sources) if lines else trace


def prefixed_traces(traced_files: Mapping[str, FileTrace], prefix: str) -> Dict[str, FileTrace]:
    """``traced_files`` with every path under ``prefix`` (a folder, relative, ``/``-separated)."""
    prefix = prefix.replace(os.sep, "/").strip("/")
    return {f"{prefix}/{path}" if prefix else path: trace for path, trace in traced_files.items()}


@dataclass(frozen=True)
class GeneratedFile:
    """A text file of a generation run as delivered: its path in the delivery, content and trace."""
    path: str
    content: str
    trace: FileTrace = field(default_factory=FileTrace)

    @property
    def content_hash(self) -> str:
        """str: The SHA-256 of the content (UTF-8)."""
        return _sha256(self.content)


def generated_text_files(entries: Iterable[Tuple[str, bytes]],
                         traced_files: Optional[Mapping[str, FileTrace]] = None) -> List[GeneratedFile]:
    """The text files among ``entries`` (path, bytes), each with its trace from ``traced_files``.

    A file that is not UTF-8 (model weights, an image) has no lines to trace: it is left out,
    and logged. Files are sorted by path, so the trace of a run does not depend on the order
    the file system lists them in.
    """
    traced_files = traced_files or {}
    files = []
    for path, raw in entries:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            logger.info("Generated file %s is not text; it is left out of the trace.", path)
            continue
        files.append(GeneratedFile(path, content, traced_files.get(path, FileTrace())))
    return sorted(files, key=lambda f: f.path)


def collect_generated_files(output_dir: str,
                            traced_files: Optional[Mapping[str, FileTrace]] = None) -> List[GeneratedFile]:
    """Every text file under ``output_dir``, by path relative to it (``/``-separated), with the
    trace a generator recorded for it in ``traced_files`` (none for an uninstrumented generator).

    This is what gives every generator, instrumented or not, file-level links: pass the
    result to :func:`build_generation_trace`.
    """
    def entries():
        for root, _, names in os.walk(output_dir):
            for name in names:
                full_path = os.path.join(root, name)
                with open(full_path, "rb") as f:
                    yield os.path.relpath(full_path, output_dir).replace(os.sep, "/"), f.read()
    return generated_text_files(entries(), traced_files)


@dataclass(frozen=True)
class InputModel:
    """A model a generation run reads, as its trace records it.

    ``ref`` is the model itself: its persistent id (the editor's diagram id), metaclass
    (``DomainModel``, ``GUIModel``, ``Agent``, ...) and name. ``element_ids`` are the ids of
    its elements, so an element a template traced is placed under the model it belongs to.
    """
    ref: ElementRef
    element_ids: FrozenSet[str] = frozenset()


def input_model(model: Model, model_id: Optional[str] = None, name: Optional[str] = None) -> InputModel:
    """The :class:`InputModel` of a B-UML model.

    Args:
        model: The model the generator read.
        model_id: Its persistent id (the editor's diagram id); defaults to ``model.id``.
        name: Its display name (the diagram title); defaults to ``model.name``.

    Raises:
        ValueError: If the model has no id: a trace refers to a model by it.
    """
    model_id = model_id or getattr(model, "id", None)
    if not model_id:
        raise ValueError(f"Model {model.name!r} has no id: a trace refers to a model by its id.")
    ids = frozenset(e.id for e in model_elements(model) if getattr(e, "id", None))
    return InputModel(ElementRef(model_id, type(model).__name__, name or model.name), ids)



class TraceExtension(Extension):
    """Jinja tag ``{% trace element[, "label"] %}`` (block) ``{% endtrace %}``.

    The rule recorded for the block is ``<template name>#<label>`` (the label
    defaults to the element's metaclass name).
    """
    tags = {"trace"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        element = parser.parse_expression()
        label = parser.parse_expression() if parser.stream.skip_if("comma") else nodes.Const(None)
        body = parser.parse_statements(("name:endtrace",), drop_needle=True)
        template = nodes.Const(parser.name or "<template>")
        return nodes.CallBlock(self.call_method("_traced", [element, label, template]), [], [], body).set_lineno(lineno)

    @staticmethod
    def _traced(element, label, template, caller):
        body = caller()
        ref = element_ref(element)
        if ref is None:
            return body
        owner = element_ref(getattr(element, "owner", None))
        payload = json.dumps({
            "id": ref.element_id,
            "type": ref.element_type,
            "name": ref.element_name,
            "rule": f"{template}#{label or ref.element_type}",
            **({"owner": [owner.element_id, owner.element_type, owner.element_name]} if owner else {}),
        })
        start = f"{_START}trace:{payload}{_END}"
        if isinstance(body, Markup):
            return Markup(start) + body + Markup(_CLOSE)
        return f"{start}{body}{_CLOSE}"


def traced_environment(templates: Union[str, BaseLoader], **options) -> Environment:
    """A Jinja environment whose templates can use ``{% trace %}``.

    Args:
        templates: The templates folder, or a Jinja loader.
        **options: Any other ``Environment`` option (``trim_blocks``, ``extensions``, ...);
            :class:`TraceExtension` is added to the extensions given.
    """
    loader = FileSystemLoader(templates) if isinstance(templates, str) else templates
    extensions = [*options.pop("extensions", ()), TraceExtension]
    return Environment(loader=loader, extensions=extensions, **options)


def extract_trace_regions(rendered: str) -> Tuple[str, List[TraceRegion]]:
    """Remove the trace markers from ``rendered``.

    Returns:
        The text without markers, and the regions it holds sorted by position. A
        region spans the lines of its block's non-whitespace content; a block that
        rendered only whitespace gives no region.
    """
    pieces: List[str] = []
    length = 0
    open_regions: list = []  # (payload, start offset in the clean text)
    spans: list = []
    last = 0
    for match in _MARKER.finditer(rendered):
        pieces.append(rendered[last:match.start()])
        length += match.start() - last
        last = match.end()
        token = match.group(1)
        if token == "/trace":
            if not open_regions:
                raise ValueError("Unbalanced trace markers: an end marker has no start.")
            payload, start = open_regions.pop()
            spans.append((payload, start, length))
        else:
            open_regions.append((json.loads(token[len("trace:"):]), length))
    if open_regions:
        raise ValueError("Unbalanced trace markers: a start marker has no end.")
    pieces.append(rendered[last:])
    clean = "".join(pieces)

    regions = []
    for payload, start, end in spans:
        block = clean[start:end]
        stripped = block.strip()
        if not stripped:
            continue
        first = start + (len(block) - len(block.lstrip()))
        last_char = first + len(stripped) - 1
        regions.append(TraceRegion(
            element_id=payload["id"], element_type=payload["type"], element_name=payload["name"],
            rule=payload["rule"],
            start_line=clean.count("\n", 0, first) + 1,
            end_line=clean.count("\n", 0, last_char) + 1,
            owner=ElementRef(*payload["owner"]) if "owner" in payload else None,
        ))
    regions.sort(key=lambda r: (r.start_line, -r.end_line))
    return clean, regions


def shift_regions(regions: Sequence[TraceRegion], lines: int) -> List[TraceRegion]:
    """``regions`` moved down by ``lines``, for a header added above the rendered text."""
    return [replace(r, start_line=r.start_line + lines, end_line=r.end_line + lines) for r in regions]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_generation_trace(
    files: Sequence[GeneratedFile],
    *,
    models: Sequence[InputModel],
    generator: str,
    generator_version: Optional[str],
    actor_name: str,
    max_stored_content_bytes: int,
    started_at: datetime,
    ended_at: datetime,
) -> TraceModel:
    """The trace fragment of one generation run.

    It holds a CodeGeneration operation performed by ``actor_name`` with the generator,
    reading the input models and writing every file and one fragment per region. Each
    input model is a root ModelArtifact (its id as ``element_id``, its metaclass as
    ``element_type``). Trace links go from every input model to every file (rule
    ``generator:<generator>``), from each whole-file source to its file, and from each
    element to the fragments rendered from it, with the template rule as evidence and
    confidence 1 (they are made by construction). An element is placed under the input
    model that holds it (the first one when none does).

    A file's content is kept when it is at most ``max_stored_content_bytes`` (UTF-8), so the
    views can show it and its fragments' lines; its hash always is. The fragment's
    ``project_id`` is None: the caller merges it into the project trace.

    Raises:
        ValueError: If no input model is given.
    """
    if not models:
        raise ValueError("A generation trace needs at least one input model.")
    actor = Person(actor_name)
    tool = CodeGenerator(generator, generator_version)
    roots = [ModelArtifact(m.ref.element_name, m.ref.element_id, m.ref.element_type) for m in models]
    artifacts = list(roots)
    reads = list(roots)
    writes: list = []
    links: list = []
    elements: Dict[str, ModelArtifact] = {}

    def root_of(element_id: str) -> ModelArtifact:
        return next((root for root, m in zip(roots, models) if element_id in m.element_ids), roots[0])

    def element(ref: ElementRef, owner: Optional[ElementRef] = None) -> ModelArtifact:
        artifact = elements.get(ref.element_id)
        if artifact is None:
            parent = element(owner) if owner else root_of(ref.element_id)
            artifact = ModelArtifact(ref.element_name, ref.element_id, ref.element_type, parent=parent)
            elements[ref.element_id] = artifact
            artifacts.append(artifact)
            reads.append(artifact)
        return artifact

    def link(source, target, rule: str):
        evidence = RuleEvidence(rule, timestamp=ended_at)
        links.append((source, target, evidence))

    for generated in files:
        stored = None
        if len(generated.content.encode("utf-8")) <= max_stored_content_bytes:
            stored = generated.content
        else:
            logger.warning("Generated file %s is over %d bytes; the trace keeps only its hash.",
                           generated.path, max_stored_content_bytes)
        file_artifact = CodeArtifact(generated.path, generated.path, generated.content_hash, content=stored)
        artifacts.append(file_artifact)
        writes.append(file_artifact)
        for root in roots:
            link(root, file_artifact, f"generator:{generator}")
        for source in generated.trace.sources:
            link(element(source.element), file_artifact, source.rule)
        lines = generated.content.split("\n")
        for region in generated.trace.regions:
            snippet = "\n".join(lines[region.start_line - 1:region.end_line])
            fragment = CodeArtifact(f"{region.element_name} ({region.start_line}-{region.end_line})", generated.path,
                                    _sha256(snippet), start_line=region.start_line, end_line=region.end_line,
                                    parent=file_artifact)
            artifacts.append(fragment)
            writes.append(fragment)
            link(element(ElementRef(region.element_id, region.element_type, region.element_name), region.owner),
                 fragment, region.rule)

    run = ToolCall(tool, 0, status=ExecutionStatus.SUCCEEDED, started_at=started_at, ended_at=ended_at)
    operation = CodeGeneration([actor], status=ExecutionStatus.SUCCEEDED, started_at=started_at, ended_at=ended_at,
                               tool_calls=[run], reads=reads, writes=writes)
    trace_links = [TraceLink(source, target, operation, [evidence], Confidence(1.0, timestamp=ended_at),
                             timestamp=ended_at) for source, target, evidence in links]
    return TraceModel(actors=[actor], tools=[tool], artifacts=artifacts, operations=[operation],
                      trace_links=trace_links)
