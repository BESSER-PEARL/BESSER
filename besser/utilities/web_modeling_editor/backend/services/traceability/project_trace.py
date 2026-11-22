"""A project's trace through the B-UML export and import of the project.

The B-UML export writes the trace as its TRACEABILITY section without file contents
(``trace_model_builder``); the JSON export, done by the editor, keeps them. A project file
must stay importable: when the file with its trace would be over the import's size cap
(``MAX_BUML_SIZE``), the trace is left out, the file says so in a comment, and the import
reports it. On import the trace is best effort: a section that does not load gives no trace
and a warning, never a refused project.
"""

import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.traceability import TraceModel
from besser.utilities.buml_code_builder.project_builder import project_to_code
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    MAX_BUML_SIZE,
    TEMP_DIR_PREFIX,
    TRACE_LEFT_OUT_NOTE,
)
from besser.utilities.web_modeling_editor.backend.services.converters import project_trace_to_json
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError

logger = logging.getLogger(__name__)


def project_code_with_trace(project: Project, trace: Optional[TraceModel], emit_ids: bool) -> Tuple[str, Optional[str]]:
    """The B-UML code of ``project`` with its trace, and a warning when the trace was left out.

    The trace is left out when the file with it would be over ``MAX_BUML_SIZE``, the size the
    import accepts; the file then starts with ``TRACE_LEFT_OUT_NOTE``, so its import says so too.
    """
    code = _project_code(project, trace, emit_ids)
    size = len(code.encode("utf-8"))
    if trace is None or size <= MAX_BUML_SIZE:
        return code, None
    warning = (f"The project trace was left out of the B-UML file: with it the file is {size / 2**20:.1f} MB, "
               f"over the {MAX_BUML_SIZE / 2**20:.1f} MB a B-UML import accepts. The JSON export keeps it.")
    logger.warning("Project %r: %s", project.name, warning)
    return f"{TRACE_LEFT_OUT_NOTE} {warning}\n{_project_code(project, None, emit_ids)}", warning


def _project_code(project: Project, trace: Optional[TraceModel], emit_ids: bool) -> str:
    with tempfile.TemporaryDirectory(prefix=f"{TEMP_DIR_PREFIX}{uuid.uuid4().hex}_") as temp_dir:
        path = os.path.join(temp_dir, "project.py")
        project_to_code(project=project, file_path=path, emit_ids=emit_ids, trace=trace)
        with open(path, encoding="utf-8") as file:
            return file.read()


def import_project_trace(content: str, project: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The trace file JSON of a project file, and warnings for what kept it from being imported.

    Best effort: the project imports whatever happens to its trace. A file exported without its
    trace (``TRACE_LEFT_OUT_NOTE``) and a TRACEABILITY section that does not load both give no
    trace and a warning.
    """
    warnings = []
    note = next((line for line in content.splitlines()[:3] if line.startswith(TRACE_LEFT_OUT_NOTE)), None)
    if note is not None:
        warnings.append(note[len(TRACE_LEFT_OUT_NOTE):].strip())
    try:
        return project_trace_to_json(content, project), warnings
    except ConversionError as exc:
        logger.warning("Project import: the trace is left out: %s", exc)
        return None, [*warnings, f"The project was imported without its trace, which could not be loaded: {exc}"]
