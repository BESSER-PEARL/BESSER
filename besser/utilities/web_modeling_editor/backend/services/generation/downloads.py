"""What a generation request downloads, and how a generator's output becomes it.

The service builds a :class:`Generation`: the :class:`GeneratedOutput` download, the
models the run read and, on the download, what each delivered file was rendered from.
``routers/generation_router.py`` turns the download into the HTTP response (ZIPs
streamed, single files sent whole); the traced endpoints build the run's trace from
the same :class:`Generation`, so both deliver the same bytes.
"""

import asyncio
import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple, Union

from besser.BUML.metamodel.traceability import LLM
from besser.generators import GeneratorInterface
from besser.utilities.generation_trace import FileTrace, shift_file_trace
from besser.utilities.provenance import add_generation_provenance_to_zip, get_besser_version
from besser.utilities.web_modeling_editor.backend.config import (
    get_filename_for_generator,
    get_generator_info,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    safe_path as _safe_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedOutput:
    """A generator's download: a ZIP archive or one text file, held in memory.

    ``traced_files`` is what the generators recorded about the delivered files
    (``GeneratorInterface.traced_files``), by path in the download (inside the ZIP, or the
    single file's download name) and with the line numbers of the delivered text.
    ``ai_files`` are the delivered files an LLM wrote or changed (``GeneratorInterface.ai_files``,
    same paths) and ``ai_llm`` that LLM, so a trace records them with AI evidence.
    """

    content: Union[io.BytesIO, bytes]
    file_name: str
    is_zip: bool
    # Single files from ``file_output`` send a non-ASCII name as RFC 6266
    # ``filename*``; ZIP names and the neural-network file name go out as is.
    encode_file_name: bool = False
    traced_files: Mapping[str, FileTrace] = field(default_factory=dict)
    ai_files: FrozenSet[str] = frozenset()
    ai_llm: Optional[LLM] = None

    @property
    def media_type(self) -> str:
        return "application/zip" if self.is_zip else "text/plain"


@dataclass(frozen=True)
class DiagramRead:
    """A diagram a generation read: its editor id and title, the B-UML model converted from it
    (None when the path hands the diagram to a helper that converts it itself) and the model's
    metaclass name (``DomainModel``, ``GUIModel``, ``Agent``, ...)."""

    diagram_id: Optional[str]
    title: str
    metaclass: str
    model: Any = None

    @classmethod
    def of(cls, diagram_id: Optional[str], title: Optional[str], model) -> "DiagramRead":
        """The read of a converted model; a blank title falls back to the model's name."""
        return cls(diagram_id, (title or "").strip() or model.name, type(model).__name__, model)


@dataclass(frozen=True)
class Generation:
    """What one generation request produced: the download and the diagrams it was generated from."""

    output: GeneratedOutput
    diagrams: Tuple[DiagramRead, ...]


def _delivered_path(generator: GeneratorInterface, path: str, root: str) -> Optional[str]:
    """``path`` (relative to ``generator``'s output folder) relative to the delivery ``root``, or None
    when the file is outside it."""
    base = os.path.realpath(generator.output_dir or generator.build_generation_dir())
    relative = os.path.relpath(os.path.join(base, path), root).replace(os.sep, "/")
    return None if relative == ".." or relative.startswith("../") else relative


def delivery_traces(generators: Iterable[GeneratorInterface], delivery_root: str) -> Dict[str, FileTrace]:
    """The ``traced_files`` of ``generators``, by path relative to ``delivery_root`` (the folder whose
    content is delivered). A generator records paths relative to its own output folder, which can
    be a subfolder of the delivery (a web app version) or hold it (Django's project folder); a file
    outside the delivery is not delivered, so its trace is dropped."""
    root = os.path.realpath(delivery_root)
    traces: Dict[str, FileTrace] = {}
    for generator in generators:
        for path, trace in generator.traced_files.items():
            relative = _delivered_path(generator, path, root)
            if relative is None:
                logger.debug("Traced file %s is outside the delivery; its trace is dropped.", path)
                continue
            traces[relative] = trace
    return traces


def delivery_ai_files(generators: Iterable[GeneratorInterface],
                      delivery_root: str) -> Tuple[FrozenSet[str], Optional[LLM]]:
    """The ``ai_files`` of ``generators`` by path relative to ``delivery_root`` (as
    :func:`delivery_traces`), and the LLM that wrote them (the last generator's that names one)."""
    root = os.path.realpath(delivery_root)
    paths, llm = set(), None
    for generator in generators:
        delivered = {_delivered_path(generator, path, root) for path in generator.ai_files} - {None}
        if delivered:
            paths |= delivered
            llm = generator.ai_llm or llm
    return frozenset(paths), llm


def zip_directory(source_dir: str) -> io.BytesIO:
    """Zip every file under ``source_dir`` into an in-memory archive.

    Entry names are relative to ``source_dir``, so the archive root is the
    directory's content rather than the directory itself.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, source_dir)
                zip_file.write(file_path, arc_name)
    return zip_buffer


async def zip_output(
    source_dir: str,
    generator_type: str,
    options: Optional[dict] = None,
    file_name: Optional[str] = None,
    generators: Sequence[GeneratorInterface] = (),
) -> GeneratedOutput:
    """Zip ``source_dir`` into a download, with ``BESSER_GENERATION.md`` at its root.

    Args:
        source_dir: Directory whose contents become the archive root.
        generator_type: Generator key, recorded in the provenance file and used
            to look up the download filename.
        options: Generation options recorded in the provenance file.
        file_name: Download filename override; defaults to
            ``get_filename_for_generator(generator_type)``.
        generators: The generators that wrote into ``source_dir``, whose traces the
            download carries (see :func:`delivery_traces`).

    Zipping reads every generated file, so it runs in a worker thread, off
    the event loop.
    """
    zip_buffer = await asyncio.to_thread(zip_directory, source_dir)
    await asyncio.to_thread(add_generation_provenance_to_zip, zip_buffer, generator_type, options)
    ai_files, ai_llm = delivery_ai_files(generators, source_dir)
    return GeneratedOutput(zip_buffer, file_name or get_filename_for_generator(generator_type), is_zip=True,
                           traced_files=delivery_traces(generators, source_dir), ai_files=ai_files, ai_llm=ai_llm)


def agent_zip_output(zip_buffer: io.BytesIO, file_name: str) -> GeneratedOutput:
    """An agent ZIP built by the agent helpers, with its provenance file added."""
    add_generation_provenance_to_zip(zip_buffer, "agent")
    return GeneratedOutput(zip_buffer, file_name, is_zip=True)


# Single-file downloads get a leading version comment only where a comment
# line is always valid; JSON, BPMN XML (the XML declaration must come first)
# and Turtle ship unchanged.
_VERSION_COMMENT_PREFIX = {".py": "#", ".sql": "--"}


def with_version_comment(content: bytes, file_name: str) -> bytes:
    """Prefix ``content`` with ``Generated by BESSER <version>`` when its type allows a comment."""
    prefix = _VERSION_COMMENT_PREFIX.get(os.path.splitext(file_name)[1].lower())
    if not prefix:
        return content
    return f"{prefix} Generated by BESSER {get_besser_version()}\n".encode("utf-8") + content


def single_file_output(temp_dir: str, generated_name: str, file_name: str,
                       generators: Sequence[GeneratorInterface] = (),
                       encode_file_name: bool = False) -> GeneratedOutput:
    """The file ``generated_name`` of ``temp_dir`` downloaded as ``file_name``, with its version
    comment, and its trace moved below that comment."""
    with open(_safe_path(temp_dir, generated_name), "rb") as f:
        generated = f.read()
    delivered = with_version_comment(generated, file_name)
    header_lines = delivered[:len(delivered) - len(generated)].count(b"\n")
    trace = delivery_traces(generators, temp_dir).get(generated_name.replace(os.sep, "/"))
    traced_files = {file_name: shift_file_trace(trace, header_lines)} if trace else {}
    return GeneratedOutput(delivered, file_name, is_zip=False, encode_file_name=encode_file_name,
                           traced_files=traced_files)


def file_output(temp_dir: str, generator_type: str,
                generators: Sequence[GeneratorInterface] = ()) -> GeneratedOutput:
    """The single file a generator wrote to ``temp_dir``, as a download (with the
    traces ``generators`` recorded for it)."""
    files = os.listdir(temp_dir)
    if not files:
        raise ValueError(f"{generator_type} generation failed: No output files were created.")

    # Match against the expected filename from generator config
    generator_info = get_generator_info(generator_type)
    expected_name = get_filename_for_generator(generator_type)
    if expected_name in files:
        file_name = expected_name
    else:
        # Fall back to matching by extension, then first file
        ext = generator_info.file_extension if generator_info else ""
        matching = [f for f in sorted(files) if f.endswith(ext)] if ext else []
        file_name = matching[0] if matching else sorted(files)[0]

    # Use the proper filename from config, unless the generator's own name is
    # meaningful (e.g. Supabase's ``<timestamp>_<slug>.sql`` migration name).
    if generator_info and generator_info.keep_generated_filename:
        proper_filename = file_name
    else:
        proper_filename = expected_name
    return single_file_output(temp_dir, file_name, proper_filename, generators, encode_file_name=True)
