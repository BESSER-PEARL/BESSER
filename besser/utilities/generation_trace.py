"""Trace links from model elements to the code a generator renders from them.

A template marks the block it renders for an element::

    {% trace class, "class" %}
    class {{ class.name }}(Base):
        __tablename__ = "{{ class.name.lower() }}"
    {% endtrace %}

:class:`TraceExtension` wraps the rendered block in invisible markers that carry
the element's persistent id. :func:`extract_trace_regions` removes them and
returns the line range of each block, so the generated file is byte-identical
to an untraced render. :func:`build_generation_trace` turns those ranges into a
:class:`~besser.BUML.metamodel.traceability.TraceModel` fragment for one
generation run.

An element without an ``id`` (a model built in Python rather than in the
editor) cannot be referenced from a trace, so its block is rendered untraced.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from jinja2 import nodes
from jinja2.ext import Extension
from markupsafe import Markup

from besser.BUML.metamodel.structural import NamedElement
from besser.BUML.metamodel.traceability import (
    CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ModelArtifact, OperationStatus, Person,
    RuleEvidence, TraceLink, TraceModel,
)

logger = logging.getLogger(__name__)

# Control characters never emitted by a template; json.dumps escapes them inside the payload.
_START, _END = "\x02", "\x03"
_CLOSE = f"{_START}/trace{_END}"
_MARKER = re.compile(f"{_START}(/trace|trace:[^{_END}]*){_END}")


@dataclass(frozen=True)
class TraceRegion:
    """The lines of a generated file rendered from one model element by one template rule.

    Lines are 1-based and inclusive.
    """
    element_id: str
    element_type: str
    element_name: str
    rule: str
    start_line: int
    end_line: int


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
        element_id = getattr(element, "id", None)
        if not element_id:
            return body
        element_type = type(element).__name__
        payload = json.dumps({
            "id": element_id,
            "type": element_type,
            "name": element.name if isinstance(element, NamedElement) else element_id,
            "rule": f"{template}#{label or element_type}",
        })
        start = f"{_START}trace:{payload}{_END}"
        if isinstance(body, Markup):
            return Markup(start) + body + Markup(_CLOSE)
        return f"{start}{body}{_CLOSE}"


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
        ))
    regions.sort(key=lambda r: (r.start_line, -r.end_line))
    return clean, regions


def shift_regions(regions: Sequence[TraceRegion], lines: int) -> List[TraceRegion]:
    """``regions`` moved down by ``lines``, for a header added above the rendered text."""
    return [replace(r, start_line=r.start_line + lines, end_line=r.end_line + lines) for r in regions]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_generation_trace(
    regions: Sequence[TraceRegion],
    *,
    file_path: str,
    content: str,
    model_id: str,
    model_name: str,
    generator: str,
    generator_version: Optional[str],
    actor_name: str,
    max_stored_content_bytes: int,
    started_at: datetime,
    ended_at: datetime,
) -> TraceModel:
    """The trace fragment of one generation run that wrote one file.

    It holds a CodeGeneration operation performed by ``actor_name`` with the
    generator, consuming the model and producing the file and one fragment per
    region; and one trace link per region, from the element to its fragment,
    with the template rule as evidence and confidence 1 (the link is made by
    construction). The model and the file are linked too.

    The file content is stored only when it is at most
    ``max_stored_content_bytes`` (UTF-8); its hash always is. The fragment's
    ``project_id`` is None: the caller merges it into the project trace.
    """
    actor = Person(actor_name)
    tool = CodeGenerator(generator, generator_version)
    model_artifact = ModelArtifact(model_name, model_id, "DomainModel")
    stored = content if len(content.encode("utf-8")) <= max_stored_content_bytes else None
    if stored is None:
        logger.warning("Generated file %s is over %d bytes; the trace keeps only its hash.",
                       file_path, max_stored_content_bytes)
    file_artifact = CodeArtifact(file_path, file_path, _sha256(content), content=stored)

    trace = TraceModel(actors=[actor], tools=[tool], artifacts=[model_artifact, file_artifact])
    operation = CodeGeneration([actor], status=OperationStatus.SUCCEEDED,
                               started_at=started_at, ended_at=ended_at, uses=[tool],
                               consumes=[model_artifact], produces=[file_artifact])
    trace.add_operation(operation)

    def link(source, target, rule: str):
        evidence = RuleEvidence(rule, timestamp=ended_at)
        trace.add_trace_link(TraceLink(source, target, operation, [evidence],
                                       Confidence(1.0, timestamp=ended_at), timestamp=ended_at))

    link(model_artifact, file_artifact, f"generator:{generator}")
    lines = content.split("\n")
    elements: dict = {}
    for region in regions:
        element = elements.get(region.element_id)
        if element is None:
            element = trace.add_artifact(ModelArtifact(region.element_name, region.element_id,
                                                       region.element_type, parent=model_artifact))
            elements[region.element_id] = element
            operation.consumes = operation.consumes + [element]
        snippet = "\n".join(lines[region.start_line - 1:region.end_line])
        fragment = trace.add_artifact(CodeArtifact(
            f"{region.element_name} ({region.start_line}-{region.end_line})", file_path, _sha256(snippet),
            start_line=region.start_line, end_line=region.end_line, parent=file_artifact))
        operation.produces = operation.produces + [fragment]
        link(element, fragment, region.rule)
    return trace
