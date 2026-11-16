"""Line ranges of the parts of a JSON document, to trace the JSON generators.

The JSON generators build a Python structure and dump it with ``json.dumps(..., indent=N)``, so there is
no template block to mark. ``json.dumps`` with an indent puts every scalar, empty container, opening and
closing bracket on a line of its own, whatever the indent, so the lines of any part of the document follow
from the structure alone (:func:`json_line_ranges`), and the output is not touched.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from besser.utilities.generation_trace import FileTrace, TraceRegion, element_ref

logger = logging.getLogger(__name__)

Path = Tuple[Any, ...]


def json_line_ranges(document: Any) -> Dict[Path, Tuple[int, int]]:
    """The 1-based inclusive line range of every part of ``json.dumps(document, indent=...)``.

    A part is named by its path: the keys (and list positions) leading to it from the root, ``()`` for the
    root itself. A ``"key": value`` pair spans the lines of its value.
    """
    ranges: Dict[Path, Tuple[int, int]] = {}

    def layout(value: Any, start: int, path: Path) -> int:
        """Record ``value``'s range from line ``start``; return the next free line."""
        if isinstance(value, dict) and value:
            children: Iterable = value.items()
        elif isinstance(value, (list, tuple)) and value:
            children = enumerate(value)
        else:
            ranges[path] = (start, start)
            return start + 1
        line = start + 1  # the opening bracket takes the first line
        for key, child in children:
            line = layout(child, line, path + (key,))
        ranges[path] = (start, line)  # the closing bracket takes the last one
        return line + 1

    layout(document, 1, ())
    return ranges


class JsonTraceBuilder:
    """Collects ``(element, rule, path)`` entries while a document is built and turns them into regions."""

    def __init__(self, template: str):
        self.template = template
        self.entries: List[Tuple[Any, str, Path]] = []

    def add(self, element, label: str, path: Path) -> None:
        """Record that the part of the document at ``path`` is rendered from ``element`` (rule ``<template>#<label>``).

        An element without an ``id`` is skipped, as in a template's ``{% trace %}``.
        """
        if element_ref(element) is not None:
            self.entries.append((element, f"{self.template}#{label}", path))

    def drop_path(self, path: Path) -> None:
        """Forget the entries at ``path``: a later key of the same name replaced that part of the document."""
        self.entries = [e for e in self.entries if e[2] != path]

    def regions(self, document: Any, line_count: Optional[int] = None) -> List[TraceRegion]:
        """The regions of ``document`` (as dumped with an indent), sorted by position.

        ``line_count`` is the number of lines of the text actually written; when it differs from the layout
        the document did not dump as assumed, and no region is returned (and a warning logged) rather than
        wrong ones.
        """
        ranges = json_line_ranges(document)
        if line_count is not None and ranges[()][1] != line_count:
            logger.warning("JSON trace: expected %d lines but the document has %d; its regions are left out.",
                           ranges[()][1], line_count)
            return []
        regions = []
        for element, rule, path in self.entries:
            if path not in ranges:
                continue
            ref = element_ref(element)
            owner = element_ref(getattr(element, "owner", None))
            start, end = ranges[path]
            regions.append(TraceRegion(ref.element_id, ref.element_type, ref.element_name, rule, start, end, owner))
        regions.sort(key=lambda r: (r.start_line, -r.end_line))
        return regions


def write_json_traced(generator, relative_path: str, text: str, document: Any, builder: Optional[JsonTraceBuilder],
                      *sources, output_dir: Optional[str] = None) -> str:
    """Write ``text`` (the dump of ``document``) through ``generator._write_traced`` and record the regions
    ``builder`` collected for it next to the whole-file ``sources``."""
    written = generator._write_traced(relative_path, text, *sources, output_dir=output_dir)
    if builder is not None and builder.entries:
        key = relative_path.replace("\\", "/")
        trace = generator.traced_files[key]
        regions = builder.regions(document, len(written.rstrip("\n").split("\n")))
        generator.traced_files[key] = FileTrace(tuple(regions), trace.sources)
    return written
