"""
Semantic consistency of class diagrams, checked with the Alloy Analyzer.

The two functions behind the ``/semantic-consistency-check`` and ``/generate-object-diagram``
endpoints search for an instance of a class diagram, and its OCL invariants, within increasingly
larger scopes, and stream their progress as Server-Sent Events.  A search that finds nothing
reports what it searched (e.g. "No instance within 11 objects per class and integers in
[-1024, 1023]."): a larger scope might still hold an instance.  Its events carry the reason in ``reason``: ``"unsat"`` when the bounds hold no
instance, ``"timeout"`` when the Alloy Analyzer timed out, which ends the search.

The search takes at most :func:`time_budget` seconds, given in its first event
(``time_budget_seconds``), so that clients can wait for it.
"""

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from besser.BUML.metamodel.object import ObjectModel
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy import AlloyTranslationError
from besser.utilities.buml_code_builder.domain_model_builder import object_model_to_code
from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.services.converters import (
    object_buml_to_json,
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.alloy_analyzer_executor import AlloyResult
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.alloy_solver import AlloySolver
from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import check_ocl_constraint

logger = logging.getLogger(__name__)

# Objects per class searched, in turn, until an instance is found.
DEFAULT_SCOPES = (5, 7, 9, 11)

# Seconds the Alloy Analyzer may run, per scope, before the search times out.
DEFAULT_TIMEOUT_SECONDS = 30

# Seconds allowed, per scope, for the work outside the Alloy Analyzer: translating the model,
# reading the instance and converting it into an object diagram.
SCOPE_OVERHEAD_SECONDS = 5


def time_budget(scopes: Sequence[int], timeout: int) -> int:
    """The seconds a search of *scopes* takes at most, when each analysis times out after *timeout* seconds."""
    return len(scopes) * (timeout + SCOPE_OVERHEAD_SECONDS)


async def check_semantic_consistency(
    input_data: DiagramInput,
    scopes: Sequence[int] = DEFAULT_SCOPES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> AsyncGenerator[str, None]:
    """Searches for an instance of the class diagram *input_data* within each of the *scopes*, in turn,
    each analysis timing out after *timeout* seconds; yields the progress as SSE lines."""
    async for event in _search(input_data, scopes, timeout, with_object_diagram=False):
        yield event


async def generate_object_diagram(
    input_data: DiagramInput,
    scopes: Sequence[int] = DEFAULT_SCOPES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> AsyncGenerator[str, None]:
    """Searches for an instance of the class diagram *input_data* as :func:`check_semantic_consistency`
    does; the event of the instance found carries it as an object diagram (``object_model``)."""
    async for event in _search(input_data, scopes, timeout, with_object_diagram=True):
        yield event


async def _search(
    input_data: DiagramInput, scopes: Sequence[int], timeout: int, with_object_diagram: bool
) -> AsyncGenerator[str, None]:
    if not scopes:
        raise ValueError("The semantic check needs at least one scope")
    prepared = _prepare(input_data)
    if isinstance(prepared, str):
        yield prepared
        return
    model, warnings = prepared

    with tempfile.TemporaryDirectory(prefix="besser_alloy_") as temp_dir:
        try:
            solver = AlloySolver(model, output_dir=temp_dir)
        except RuntimeError:
            logger.exception("The Alloy Analyzer is not available")
            yield _failure("The Alloy Analyzer is not available on this server.", warnings=warnings)
            return

        for index, scope in enumerate(scopes):
            try:
                bounds = await asyncio.to_thread(solver.generate_specification, scope)
                # The first event of the search tells clients how long it can take
                budget = {"time_budget_seconds": time_budget(scopes, timeout)} if index == 0 else {}
                yield _progress(f"Searching within {bounds}...", scope=bounds.objects, **budget)
                result, object_model = await asyncio.to_thread(
                    _analyze, solver, scope, timeout, with_object_diagram
                )
            except AlloyTranslationError as error:
                yield _failure(
                    "The model cannot be translated to Alloy.", errors=[str(error)], warnings=warnings
                )
                return
            except Exception:
                logger.exception("Unexpected error in the semantic check at scope %s", scope)
                yield _failure(f"The semantic check failed unexpectedly at scope {scope}.", warnings=warnings)
                return

            if result == AlloyResult.TIMEOUT:
                yield _failure(
                    f"Timed out searching within {bounds}.",
                    warnings=warnings,
                    scope=bounds.objects,
                    reason="timeout",
                )
                return
            if result == AlloyResult.SAT:
                fields: dict[str, Any] = {"scope": bounds.objects}
                if with_object_diagram:
                    fields["object_model"] = _object_model_to_json(object_model, input_data.model, temp_dir)
                yield _success(f"Instance found within {bounds}.", warnings=warnings, **fields)
                return
            if scope != scopes[-1]:
                yield _progress(f"No instance within {bounds}.", sat=False, scope=bounds.objects, reason="unsat")

        yield _failure(
            f"No instance within {bounds}.", sat=False, warnings=warnings, scope=bounds.objects, reason="unsat"
        )


def _prepare(input_data: DiagramInput) -> tuple[DomainModel, list[str]] | str:
    """The domain model of the class diagram *input_data* and its warnings, or the failure event
    of a diagram that cannot be checked: not a class diagram, invalid, or with invalid OCL."""
    diagram_type = input_data.model.get("type") if input_data.model else None
    if diagram_type != "ClassDiagram":
        return _failure("Semantic consistency is only checked for class diagrams.")

    try:
        model = process_class_diagram({"title": input_data.title, "model": input_data.model})
    except Exception as error:
        logger.exception("Cannot convert the class diagram for the semantic check")
        return _failure("The class diagram cannot be converted: the semantic check was skipped.", errors=[str(error)])
    validation = model.validate(raise_exception=False)
    warnings = list(validation.get("warnings", []))
    if validation.get("errors"):
        return _failure(
            "The class diagram is invalid: the semantic check was skipped.",
            errors=validation["errors"],
            warnings=warnings,
        )

    # The OCL invariants must be syntax checked before their translation into Alloy
    ocl_result = check_ocl_constraint(model)
    ocl_errors = list(ocl_result.get("invalid_constraints", []))
    if not ocl_result.get("success", True) and ocl_result.get("message"):
        ocl_errors.append(ocl_result["message"])
    # Invariants the converter could not parse are dropped from the model: checking without them is wrong
    ocl_errors.extend(getattr(model, "ocl_warnings", None) or [])
    if ocl_errors:
        return _failure(
            "The OCL constraints are invalid: the semantic check was skipped.", errors=ocl_errors, warnings=warnings
        )
    return model, warnings


def _analyze(
    solver: AlloySolver, scope: int, timeout: int, with_object_diagram: bool
) -> tuple[AlloyResult, ObjectModel | None]:
    """Runs the analysis of *scope*; returns its result and, if requested and found, an object model."""
    if not with_object_diagram:
        return solver.check_consistency(scope=scope, timeout=timeout), None
    result, object_models = solver.generate_object_models(scope=scope, timeout=timeout)
    return result, object_models[0] if object_models else None


def _object_model_to_json(object_model: ObjectModel, domain_json: dict[str, Any], directory: str) -> dict[str, Any]:
    """The object diagram, for the editor, of *object_model*, an instance of the class diagram *domain_json*."""
    path = os.path.join(directory, "object_model.py")
    object_model_to_code(object_model, path)
    with open(path, encoding="utf-8") as f:
        return object_buml_to_json(f.read(), domain_json)


# ---------------------------------------------------------------------------
# Server-Sent Events
# ---------------------------------------------------------------------------

def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _progress(message: str, sat: bool | None = None, **fields: Any) -> str:
    """A non-terminal event."""
    return _sse({"sat": sat, "done": False, "message": message, **fields})


def _success(message: str, warnings: list[str] | None = None, **fields: Any) -> str:
    """The terminal event of an instance found."""
    return _sse({
        "sat": True, "isValid": True, "done": True, "message": message,
        "errors": [], "warnings": list(warnings or []), **fields,
    })


def _failure(
    message: str,
    sat: bool | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    **fields: Any,
) -> str:
    """The terminal event of a search that found no instance, could not run or failed."""
    return _sse({
        "sat": sat, "isValid": False, "done": True, "message": message,
        "errors": list(errors or []), "warnings": list(warnings or []), **fields,
    })
