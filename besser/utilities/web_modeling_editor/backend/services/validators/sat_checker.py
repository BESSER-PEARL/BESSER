"""
Semantic consistency checker (SAT-based) for UML-BESSER class diagrams.

This module orchestrates the web editor's semantic consistency workflows:
- conversion of frontend class diagram JSON into a BUML model,
- structural and OCL constraint validation of the model,
- SAT/consistency checking via the Alloy solver (delegated to the
  ``alloy_solver`` module) across increasingly larger scopes,
- conversion of satisfying Alloy instances back into frontend ObjectDiagram JSON.

Results are streamed as Server-Sent Events (SSE).
"""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from typing import Any

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy import AlloySolver
from besser.generators.alloy.instance_generator.alloy_analyzer_executor import (
    AlloyResult,
)
from besser.generators.alloy.translate_ocl_alloy import DATES_DICT
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    object_buml_to_json,
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import (
    check_ocl_constraint,
)

logger = logging.getLogger(__name__)

SCOPE_STEPS = [5, 8, 9, 10]  # Scopes to be used when checking semantic consistency
TIMEOUT_SECONDS = 50


#----------------------------------------------------------------------
def convert_json_to_buml(input_data: DiagramInput) -> DomainModel | dict[str, Any]:
    """
    Converts a diagram in JSON format to a corresponding BUML model.

    If the provided diagram is not a class diagram, no conversion is performed, 
    and a dictionary containing an unsupported operation message is produced.
    """
    diagram_type = input_data.model.get("type") if input_data.model else None
    if diagram_type != "ClassDiagram":
        return _error_payload(
            "Semantic  Check is only available for Class Diagrams.",
        )
    json_data = {"title": input_data.title, "model": input_data.model}
    return process_class_diagram(json_data)

#----------------------------------------------------------------------
def validate_buml_structure(buml_model: DomainModel) -> tuple[list[str], list[str]]:
    """
    Checks the structural (syntactic) consistency of a BUML model. 

    Delegates the checking into buml_model.validate() functionality.

    When validation does not raise exceptions, the obtained errors and warnings is returned.
    If exceptions are thrown, a message indicating structural validation error is produced. 
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
    """
    Validates the syntax of OCL constraints, resorting to check_ocl_constraint() 
    functionality.

    Since semantic consistency check requires the OCL syntax checking to fully pass,
    all errors and warnings of the syntactic check are treated as errors.

    Result is a dictionary with errors and warnings if validation failed, 
    None if validation passed.
    Parameter structural_warnings is streamed into the output too. 
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

    if not ocl_result.get("success", True) or ocl_errors:
        ocl_errors.append(ocl_result.get("message", "OCL validation failed."))
    all_warnings = structural_warnings or []
    if ocl_errors:
        return all_warnings, _error_payload(
            " OCL constraints are invalid — SAT check skipped.",
            errors=ocl_errors,
            warnings=all_warnings,
        )
    return all_warnings, None

#----------------------------------------------------------------------

async def check_consistency_alloy(input_data: DiagramInput) -> AsyncGenerator[str, None]:
    """
    Performs semantic satisfiability check of a BUML class diagram.

    The semantic satisfiability check involves:
    - syntactic check of the structure of the class diagram
    - syntactic check of the OCL constraints, if present
    - translation of class diagram and OCL constraints into an Alloy specification
    - Checks for consistency of the Alloy specification for increasingly larger
    scopes, stopping when SAT is found, timeout is reached, or all scopes are 
    exhausted.

    Result is yielded as SSE-formatted strings. 
    """
    buml_model = convert_json_to_buml(input_data)
    if isinstance(buml_model, dict):
        yield _sse({**buml_model, "done": True})
        return

    structural_errors, structural_warnings = validate_buml_structure(buml_model)
    if structural_errors:
        yield _event_failure(
            " Structural validation failed — SAT check skipped.",
            errors=structural_errors,
            warnings=structural_warnings,
        )
        return

    all_warnings, ocl_error = validate_ocl_constraints(buml_model, structural_warnings)
    if ocl_error:
        yield _sse({**ocl_error, "done": True})
        return

    # Steps 4-6: iterate scopes
    with tempfile.TemporaryDirectory() as temp_dir:
        for scope in SCOPE_STEPS:
            yield _event_progress(f"🔍 Trying scope {scope}...", scope=scope)

            try:
                check = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_alloy_sat_validation, buml_model, all_warnings, scope=scope,
                        output_dir=os.path.join(temp_dir, f"scope_{scope}"),
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield _event_failure(
                    f"⏱️ Timeout after {TIMEOUT_SECONDS}s with scope {scope} — model may be unsatisfiable.",
                    sat=False,
                    warnings=all_warnings,
                )
                return
            except Exception as exc:
                logger.exception("Unexpected error during Alloy SAT check (scope %s)", scope)
                yield _event_failure(
                    f" Semantic check failed with an unexpected error at scope {scope}.",
                    sat=False,
                    errors=[str(exc)] if str(exc).strip() else [],
                    warnings=all_warnings,
                )
                return

            if check["error"]:
                yield _sse({**check["error"], "done": True})
                return

            if check["sat"]:
                yield _event_success(
                    f" SAT found with scope {scope} (command: {check['command_name']}).",
                    warnings=all_warnings,
                    scope=scope,
                )
                return

            yield _event_progress(
                f" UNSAT with scope {scope}. Trying larger scope...",
                sat=False,
                scope=scope,
            )

        # All scopes exhausted without finding SAT
        yield _event_failure(
            f" UNSAT with all scopes tried ({SCOPE_STEPS}). Model is likely unsatisfiable.",
            sat=False,
            warnings=all_warnings,
        )


async def generate_object_diagram_alloy(input_data: DiagramInput) -> AsyncGenerator[str, None]:
    """
    Generates object diagram that complies with constraints of a BUML class diagram,
    incluing OCL constraints, if present.

    The generation of the semantically consistent object diagram involves:
    - syntactic check of the structure of the class diagram
    - syntactic check of the OCL constraints, if present
    - translation of class diagram and OCL constraints into an Alloy specification
    - Checks for consistency of the Alloy specification for increasingly larger
    scopes, stopping when SAT is found, timeout is reached, or all scopes are
    exhausted.
    - Translation of one (the first) Alloy instance back into a front-end object 
    diagram.

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
        yield _event_failure(
            " Structural validation failed — SAT check skipped.",
            errors=structural_errors,
            warnings=structural_warnings,
        )
        return

    all_warnings, ocl_error = validate_ocl_constraints(buml_model, structural_warnings)
    if ocl_error:
        yield _sse({**ocl_error, "done": True})
        return

    # Steps 4-6: iterate scopes until SAT is found
    with tempfile.TemporaryDirectory() as temp_dir:
        for scope in SCOPE_STEPS:
            yield _event_progress(f"🔍 Trying scope {scope}...", scope=scope)

            try:
                check = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_alloy_sat_validation, buml_model, all_warnings, scope=scope,
                        output_dir=os.path.join(temp_dir, f"scope_{scope}"),
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                yield _event_failure(
                    f"⏱️ Timeout after {TIMEOUT_SECONDS}s with scope {scope} — model may be unsatisfiable.",
                    sat=False,
                    warnings=all_warnings,
                )
                return
            except Exception as exc:
                logger.exception("Unexpected error during Alloy SAT check (scope %s)", scope)
                yield _event_failure(
                    f" Semantic check failed with an unexpected error at scope {scope}.",
                    sat=False,
                    errors=[str(exc)] if str(exc).strip() else [],
                    warnings=all_warnings,
                )
                return

            if check["error"]:
                yield _sse({**check["error"], "done": True})
                return

            if not check["sat"]:
                yield _event_progress(
                    f" UNSAT with scope {scope}. Trying larger scope...",
                    sat=False,
                    scope=scope,
                )
                continue

            # SAT → convert the first BUML object instance to frontend Object Diagram JSON
            yield _event_progress(
                f"✅ SAT found with scope {scope} "
                f"(command: {check['command_name']}). Generating Object Diagram...",
                sat=True,
                scope=scope,
            )

            if not check["buml_instances"]:
                logger.warning("SAT=true but no BUML object instance was generated")
                yield _event_failure(
                    f" Model is satisfiable (command: {check['command_name']}), "
                    "but no object diagram was generated.",
                    sat=True,
                    warnings=all_warnings,
                    scope=scope,
                )
                return

            try:
                object_model = await asyncio.to_thread(
                    object_buml_to_json, check["buml_instances"][0], input_data.model
                )
            except Exception as exc:
                logger.exception("Failed to convert Alloy instance to frontend ObjectDiagram")
                yield _event_failure(
                    f" Model is satisfiable (command: {check['command_name']}), "
                    "but instance conversion failed.",
                    sat=True,
                    error=str(exc),
                    warnings=all_warnings,
                    scope=scope,
                )
                return

            yield _event_success(
                f" Model is satisfiable (command: {check['command_name']}).",
                warnings=all_warnings,
                scope=scope,
                object_model=object_model,
                dates_dict=dict(DATES_DICT),
            )
            return

        # All scopes exhausted without finding SAT
        yield _event_failure(
            f" UNSAT with all scopes tried ({SCOPE_STEPS}). Model is likely unsatisfiable.",
            sat=False,
            warnings=all_warnings,
        )


def run_alloy_sat_validation(
    buml_model: DomainModel,
    all_warnings: list[str] | None = None,
    scope: int = 5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Generate an Alloy specification and execute it, returning the result.
    Args:
        buml_model: BUML domain model to check.
        all_warnings: Web-layer warnings to fold into the response.
        scope: Alloy scope (max atoms per signature).
        output_dir: Where the specification is generated and analyzed. When
            ``None`` the solver creates a temporary directory.

    Returns:
        A flat dict with keys ``sat``, ``command_name``, ``buml_instances``,
        ``output_dir`` and ``error``. ``buml_instances`` is a list of BUML
        object-diagram code strings (empty unless SAT). ``error`` is ``None``
        on success; on translation failure or timeout it holds the SSE-ready
        error response.
    """
    warnings = all_warnings or []
    try:
        # Generate the Alloy specification (model.als)
        solver = AlloySolver(buml_model, output_dir=output_dir, scope=scope)
    except ValueError as exc:
        msg = str(exc)
        return {
            "sat": None,
            "command_name": "",
            "buml_instances": [],
            "output_dir": output_dir or "output",
            "error": _error_payload(
                msg,
                errors=[msg] if msg else [],
                warnings=warnings,
            ),
        }

    # Execute the generated specification, producing BUML object instances.
    # The generated code is emitted in the dialect the (development) web-editor
    # object_buml_to_json converter can parse (plain attribute assignments and
    # literal attribute values) instead of the executable setattr/datetime form.
    result, buml_instances = solver.generate_object_diagrams(num_instances=1)

    if result == AlloyResult.TIMEOUT:
        return {
            "sat": False,
            "command_name": "",
            "buml_instances": [],
            "output_dir": solver.alloy_output_dir,
            "error": _error_payload(
                f"Alloy Analyzer timed out with scope {scope}.",
                sat=False,
                warnings=warnings,
            ),
        }

    return {
        "sat": result == AlloyResult.SAT,
        "command_name":"instance_model",
        "buml_instances": buml_instances,
        "output_dir": solver.alloy_output_dir,
        "error": None,
    }


def _error_payload(
    message: str,
    *,
    sat: bool | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Builds an SSE-ready error payload (without the ``done`` flag)."""
    return {
        "sat": sat,
        "isValid": False,
        "message": message,
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        **extra,
    }


def _event_progress(message: str, *, sat: bool | None = None, **extra: Any) -> str:
    """Formats a non-terminal progress event as an SSE line."""
    return _sse({"sat": sat, "done": False, "message": message, **extra})


def _event_success(
    message: str,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> str:
    """Formats a terminal SAT success event as an SSE line."""
    return _sse({
        "sat": True,
        "isValid": True,
        "done": True,
        "message": message,
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        **extra,
    })


def _event_failure(
    message: str,
    *,
    sat: bool | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    **extra: Any,
) -> str:
    """Formats a terminal failure event as an SSE line."""
    return _sse({
        **_error_payload(message, sat=sat, errors=errors, warnings=warnings, **extra),
        "done": True,
    })


def _sse(data: dict[str, Any]) -> str:
    """
    Formats a dict as an SSE data line.
    """
    return f"data: {json.dumps(data)}\n\n"
