"""SAT consistency checker for Alloy-based validation."""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncGenerator, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy_generator.instance_generator.alloy_solver import AlloySolver
from besser.generators.alloy_generator.step_3_alloy_to_buml import (
    AlloyToBesserConverter,
)
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)

logger = logging.getLogger(__name__)

SCOPE_STEPS = [5, 8, 9, 10]  # Scopes to try in order.
TIMEOUT_SECONDS = 50


def _resolve_first_instance_xml(exec_output_dir: str, solutions: list[dict[str, Any]]) -> str | None:
    base_path = Path(exec_output_dir)

    def _iter_solution_xml_paths() -> Iterable[Path]:
        for solution in solutions or []:
            for instance in solution.get("instances", []) or []:
                if isinstance(instance, str):
                    yield Path(instance)
                    continue

                if isinstance(instance, dict):
                    for key in ("xml", "path", "file", "instance", "filename"):
                        value = instance.get(key)
                        if isinstance(value, str):
                            yield Path(value)
                            break

    for xml_path in _iter_solution_xml_paths():
        candidate = xml_path if xml_path.is_absolute() else (base_path / xml_path)
        if candidate.suffix.lower() == ".xml" and candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

    # Fallback: pick the first XML in filesystem order (deterministic).
    xml_candidates = sorted(base_path.rglob("*.xml"))
    if xml_candidates:
        return str(xml_candidates[0])

    return None


def _first_instance_source(solutions: list[dict[str, Any]] | None) -> str | dict | None:
    """Return the first Alloy instance (XML file path or inlined JSON dict)."""
    for solution in solutions or []:
        for instance in solution.get("instances", []) or []:
            if isinstance(instance, str):
                return instance
            if isinstance(instance, dict):
                for key in ("xml", "path", "file", "instance", "filename"):
                    value = instance.get(key)
                    if isinstance(value, str):
                        return value
                return instance
    return None


def _alloy_xml_to_frontend_object_model(
    xml_instance_path: str, reference_class_model: dict[str, Any]
) -> dict[str, Any]:
    """Convert an Alloy XML instance into the frontend ObjectDiagram JSON format."""
    converter = AlloyToBesserConverter(xml_instance_path)
    converter.parse_xml()
    object_buml_code = converter.generate_object_diagram_code()
    return object_buml_to_json(object_buml_code, reference_class_model)

#----------------------------------------------------------------------
def convert_json_to_buml(input_data: DiagramInput) -> DomainModel | dict[str, Any]:
    """Step 1: Convert diagram JSON to BUML model.
    Returns BUML DomainModel on success, or a dict error response.
    """
    diagram_type = input_data.model.get("type") if input_data.model else None
    if diagram_type != "ClassDiagram":
        return {
            "sat": None,
            "isValid": False,
            "message": "Semantic  Check is only available for Class Diagrams.",
            "errors": [],
            "warnings": [],
        }
    json_data = {"title": input_data.title, "model": input_data.model}
    return process_class_diagram(json_data)

#----------------------------------------------------------------------
def validate_buml_structure(buml_model: DomainModel) -> tuple[list[str], list[str]]:
    """Step 2: Run structural validation on the BUML model.

    Returns (errors, warnings). Never raises — exceptions are caught and
    returned as errors so callers always get a consistent (list, list) tuple.
    """
    try:
        result = buml_model.validate(raise_exception=False)
        return result.get("errors", []), result.get("warnings", [])
    except Exception as e:
        return [f"Structural validation error: {e!s}"], []

#----------------------------------------------------------------------

def validate_ocl_constraints(
    buml_model: DomainModel,
    structural_warnings: list[str] | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Step 3: Validate OCL constraints through the BOCL-based pipeline.

    Returns (all_warnings, error_response).
    - On success: (warnings, None)
    - On invalid OCL: (warnings, dict)
    """
    ocl_result = check_ocl_constraint(buml_model, object_model=None)
    ocl_errors = list(ocl_result.get("invalid_constraints", []))


    # Promote OCL warnings (malformed syntax or missing classes/fields) to blocking
    # errors, as valid OCL invariants are essential for SAT execution.
    conversion_warnings = list(getattr(buml_model, "ocl_warnings", []) or [])
    ocl_tokens = ("ocl", "constraint", "precondition", "postcondition", "invariant")
    blocking_ocl_conversion_issues = [
        warning.replace("Warning", "Error")
        for warning in conversion_warnings
        if any(token in warning.lower() for token in ocl_tokens)
    ]
    ocl_errors.extend(blocking_ocl_conversion_issues)

    if not ocl_result.get("success", True) and not ocl_errors:
        ocl_errors.append(ocl_result.get("message", "OCL validation failed."))
    all_warnings = structural_warnings or []
    if ocl_errors:
        return all_warnings, {
            "sat": None,
            "isValid": False,
            "message": " OCL constraints are invalid — SAT check skipped.",
            "errors": ocl_errors,
            "warnings": all_warnings,
        }
    return all_warnings, None

#----------------------------------------------------------------------
def run_alloy_sat_validation(
    buml_model: DomainModel,
    all_warnings: list[str] | None = None,
    scope: int = 5,
    output_type: str = "json",
    temp_dir: str | None = None,
) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None, str]:
    """Steps 4-6: Generate Alloy input, execute the analyzer, and parse SAT output.

    Delegates to AlloySolver.run_sat_validation().
    Returns (parsed_data, error_response, exec_output_dir).
    """
    warnings = all_warnings or []
    cm = tempfile.TemporaryDirectory() if temp_dir is None else nullcontext(temp_dir)
    with cm as td:
        solver = AlloySolver(buml_model, scope=scope, output_dir=td)
        parsed, error, exec_output_dir = solver.run_sat_validation(
            structural_warnings=warnings, output_type=output_type, temp_dir=td,
        )
        if error:
            return None, {**error, "warnings": warnings}, exec_output_dir
        return parsed, None, exec_output_dir

async def check_alloy_consistency_stream(input_data: DiagramInput) -> AsyncGenerator[str, None]:
    """Stream SAT check results trying increasing scopes.

    Yields SSE-formatted strings. Stops when SAT is found, timeout is reached,
    or all scopes are exhausted.
    """
    buml_model = convert_json_to_buml(input_data)
    if isinstance(buml_model, dict):
        yield _sse({**buml_model, "done": True})
        return

    structural_errors, structural_warnings = validate_buml_structure(buml_model)
    if structural_errors:
        yield _sse({
            "sat": None,
            "isValid": False,
            "message": " Structural validation failed — SAT check skipped.",
            "errors": structural_errors,
            "warnings": structural_warnings,
            "done": True,
        })
        return

    all_warnings, ocl_error = validate_ocl_constraints(buml_model, structural_warnings)
    if ocl_error:
        yield _sse({**ocl_error, "done": True})
        return

    # Steps 4-6: iterate scopes
    for scope in SCOPE_STEPS:
        yield _sse({
            "sat": None,
            "done": False,
            "message": f"🔍 Trying scope {scope}...",
            "scope": scope,
        })

        try:
            parsed, error, _ = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda s=scope: run_alloy_sat_validation(buml_model, all_warnings, scope=s)
                ),
                timeout=TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            yield _sse({
                "sat": False,
                "isValid": False,
                "done": True,
                "message": f"⏱️ Timeout after {TIMEOUT_SECONDS}s with scope {scope} — model may be unsatisfiable.",
                "errors": [],
                "warnings": all_warnings,
            })
            return

        if error:
            yield _sse({**error, "done": True})
            return

        sat, first_command_name, _ = parsed
        if sat:
            yield _sse({
                "sat": True,
                "isValid": True,
                "done": True,
                "message": f" SAT found with scope {scope} (command: {first_command_name}).",
                "errors": [],
                "warnings": all_warnings,
                "scope": scope,
            })
            return

        yield _sse({
            "sat": False,
            "done": False,
            "message": f" UNSAT with scope {scope}. Trying larger scope...",
            "scope": scope,
        })

    # All scopes exhausted without finding SAT
    yield _sse({
        "sat": False,
        "isValid": False,
        "done": True,
        "message": f" UNSAT with all scopes tried ({SCOPE_STEPS}). Model is likely unsatisfiable.",
        "errors": [],
        "warnings": all_warnings,
    })


def _run_alloy_do_scope(
    buml_model: DomainModel,
    all_warnings: list[str],
    scope: int,
    temp_dir: str,
) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None, str]:
    """Run Alloy (XML output) for a single scope inside a caller-owned temp_dir.

    Wraps run_alloy_sat_validation in XML mode so the caller keeps access to
    the XML instances after return.
    """
    return run_alloy_sat_validation(
        buml_model, all_warnings, scope=scope,
        output_type="xml", temp_dir=os.path.join(temp_dir, f"scope_{scope}"),
    )


async def generate_alloy_do_stream(input_data: DiagramInput) -> AsyncGenerator[str, None]:
    """Stream Object Diagram generation trying increasing scopes.

    Yields SSE-formatted progress events per scope. Stops at the first SAT
    instance (converting it to a frontend Object Diagram), on timeout, or when
    all scopes are exhausted.
    """
    # Steps 1-3: pre-validation (same flow as check_alloy_consistency_stream)
    buml_model = convert_json_to_buml(input_data)
    if isinstance(buml_model, dict):
        yield _sse({**buml_model, "done": True})
        return

    structural_errors, structural_warnings = validate_buml_structure(buml_model)
    if structural_errors:
        yield _sse({
            "sat": None,
            "isValid": False,
            "message": " Structural validation failed — SAT check skipped.",
            "errors": structural_errors,
            "warnings": structural_warnings,
            "done": True,
        })
        return

    all_warnings, ocl_error = validate_ocl_constraints(buml_model, structural_warnings)
    if ocl_error:
        yield _sse({**ocl_error, "done": True})
        return

    # Steps 4-6: iterate scopes until SAT is found
    with tempfile.TemporaryDirectory() as temp_dir:
        for scope in SCOPE_STEPS:
            yield _sse({
                "sat": None,
                "done": False,
                "message": f"🔍 Trying scope {scope}...",
                "scope": scope,
            })

            try:
                parsed, error, exec_output_dir = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=scope: _run_alloy_do_scope(buml_model, all_warnings, s, temp_dir)
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield _sse({
                    "sat": False,
                    "isValid": False,
                    "done": True,
                    "message": f"⏱️ Timeout after {TIMEOUT_SECONDS}s with scope {scope} — model may be unsatisfiable.",
                    "errors": [],
                    "warnings": all_warnings,
                })
                return

            if error:
                yield _sse({**error, "done": True})
                return

            sat, first_command_name, solutions = parsed
            if not sat:
                yield _sse({
                    "sat": False,
                    "done": False,
                    "message": f" UNSAT with scope {scope}. Trying larger scope...",
                    "scope": scope,
                })
                continue

            # SAT → locate XML instance → convert to frontend Object Diagram JSON
            yield _sse({
                "sat": True,
                "done": False,
                "message": (
                    f"✅ SAT found with scope {scope} "
                    f"(command: {first_command_name}). Generating Object Diagram..."
                ),
                "scope": scope,
            })

            loop = asyncio.get_event_loop()
            try:
                xml_instance_path = await loop.run_in_executor(
                    None, _resolve_first_instance_xml, exec_output_dir, solutions
                )
                if not xml_instance_path:
                    logger.warning("SAT=true but no Alloy XML instance was found in %s", exec_output_dir)
                    yield _sse({
                        "sat": True,
                        "isValid": False,
                        "done": True,
                        "message": (
                            f" Model is satisfiable (command: {first_command_name}), "
                            "but no instance XML was found."
                        ),
                        "errors": [],
                        "warnings": all_warnings,
                        "scope": scope,
                    })
                    return

                object_model = await loop.run_in_executor(
                    None, _alloy_xml_to_frontend_object_model, xml_instance_path, input_data.model
                )
            except Exception as exc:
                logger.exception("Failed to convert Alloy instance to frontend ObjectDiagram")
                yield _sse({
                    "sat": True,
                    "isValid": False,
                    "done": True,
                    "message": (
                        f" Model is satisfiable (command: {first_command_name}), "
                        "but instance conversion failed."
                    ),
                    "error": str(exc),
                    "warnings": all_warnings,
                    "scope": scope,
                })
                return

            yield _sse({
                "sat": True,
                "isValid": True,
                "done": True,
                "message": f" Model is satisfiable (command: {first_command_name}).",
                "errors": [],
                "warnings": all_warnings,
                "scope": scope,
                "object_model": object_model,
            })
            return

        # All scopes exhausted without finding SAT
        yield _sse({
            "sat": False,
            "isValid": False,
            "done": True,
            "message": f" UNSAT with all scopes tried ({SCOPE_STEPS}). Model is likely unsatisfiable.",
            "errors": [],
            "warnings": all_warnings,
        })


def _sse(data: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"