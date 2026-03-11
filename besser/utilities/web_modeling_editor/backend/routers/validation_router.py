"""
Validation Router

Handles all diagram validation endpoints for the BESSER web modeling editor backend.
"""

import logging

from fastapi import APIRouter, HTTPException

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
)
from besser.utilities.web_modeling_editor.backend.models.responses import (
    ValidationResponse,
)

# Backend services - Converters
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_state_machine,
    process_agent_diagram,
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.constants.user_buml_model import (
    domain_model as user_reference_domain_model,
)

# Backend services - Validators
from besser.utilities.web_modeling_editor.backend.services.validators import (
    check_ocl_constraint,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["validation"])


@router.post("/validate-diagram", response_model=ValidationResponse)
async def validate_diagram(input_data: DiagramInput):
    """
    Validate diagram by converting to BUML and running metamodel validation.

    This is the unified validation endpoint that:
    1. Converts JSON to BUML (construction validation)
    2. Calls .validate() method on the model for structured metamodel validation
    3. For ClassDiagram/ObjectDiagram: runs OCL constraint checks if conversion succeeded
    4. Returns unified validation results with errors, warnings, and OCL results
    """
    try:
        diagram_type = input_data.model.get("type") if input_data.model else None
        validation_errors = []
        validation_warnings = []
        buml_model = None
        object_model = None

        # Step 1: Convert to BUML (construction validation)
        try:
            if diagram_type == "ClassDiagram":
                json_data = {
                    "title": input_data.title,
                    "model": input_data.model
                }
                buml_model = process_class_diagram(json_data)

                # Run structured metamodel validation
                validation_result = buml_model.validate(raise_exception=False)
                validation_errors.extend(validation_result.get("errors", []))
                validation_warnings.extend(validation_result.get("warnings", []))

            elif diagram_type == "ObjectDiagram":
                reference_data = input_data.model.get("referenceDiagramData", {})
                if not reference_data:
                    return {
                        "isValid": False,
                        "errors": ["Object diagram requires reference class diagram data"],
                        "warnings": [],
                        "message": "\u274c Validation failed"
                    }

                # Process the reference class diagram first
                reference_json = {
                    "title": reference_data.get("title", "Reference Classes"),
                    "model": {
                        "elements": reference_data.get("elements", {}),
                        "relationships": reference_data.get("relationships", {})
                    }
                }
                buml_model = process_class_diagram(reference_json)

                # Validate the domain model first
                domain_validation = buml_model.validate(raise_exception=False)
                validation_errors.extend(domain_validation.get("errors", []))
                validation_warnings.extend(domain_validation.get("warnings", []))

                # If domain model is valid, process and validate object model
                if domain_validation.get("success", False):
                    object_model = process_object_diagram(input_data.model_dump(), buml_model)
                    object_validation = object_model.validate(raise_exception=False)
                    validation_errors.extend(object_validation.get("errors", []))
                    validation_warnings.extend(object_validation.get("warnings", []))

            elif diagram_type == "UserDiagram":
                buml_model = user_reference_domain_model
                domain_validation = buml_model.validate(raise_exception=False)
                validation_errors.extend(domain_validation.get("errors", []))
                validation_warnings.extend(domain_validation.get("warnings", []))

                if domain_validation.get("success", False):
                    object_model = process_object_diagram(input_data.model_dump(), buml_model)
                    object_validation = object_model.validate(raise_exception=False)
                    validation_errors.extend(object_validation.get("errors", []))
                    validation_warnings.extend(object_validation.get("warnings", []))

            elif diagram_type == "StateMachineDiagram":
                state_machine = process_state_machine(input_data.model_dump())

                # StateMachine does not inherit DomainModel.validate(), so
                # perform basic structural validation here.
                if not state_machine.states:
                    validation_errors.append("State machine has no states.")
                else:
                    has_initial = any(s.initial for s in state_machine.states)
                    if not has_initial:
                        validation_errors.append("State machine has no initial state.")

                    # Check for duplicate state names
                    state_names = [s.name for s in state_machine.states]
                    seen = set()
                    for sn in state_names:
                        if sn in seen:
                            validation_errors.append(f"Duplicate state name: '{sn}'.")
                        seen.add(sn)

            elif diagram_type == "AgentDiagram":
                agent_model = process_agent_diagram(input_data.model_dump())
                validation_result = agent_model.validate(raise_exception=False)
                validation_errors.extend(validation_result.get("errors", []))
                validation_warnings.extend(validation_result.get("warnings", []))

            elif diagram_type == "GUINoCodeDiagram":
                return {
                    "isValid": True,
                    "message": "\u2705 GUI diagram is valid",
                    "errors": [],
                    "warnings": []
                }
            else:
                return {
                    "isValid": False,
                    "errors": [f"Unsupported diagram type: {diagram_type}"],
                    "warnings": [],
                    "message": "\u274c Validation failed"
                }

        except ValueError as e:
            # Construction validation errors (from BUML creation setters)
            error_msg = str(e)
            validation_errors.append(error_msg)
        except Exception as e:
            validation_errors.append(f"Validation error: {str(e)}")

        # Step 2: If BUML model created successfully AND it's a diagram with OCL support
        ocl_results = None
        if buml_model and diagram_type in ["ClassDiagram", "ObjectDiagram"] and len(validation_errors) == 0:
            try:
                if diagram_type == "ObjectDiagram" and object_model:
                    ocl_results = check_ocl_constraint(buml_model, object_model)
                else:
                    ocl_results = check_ocl_constraint(buml_model)

                # Add OCL warnings if present
                if hasattr(buml_model, "ocl_warnings") and buml_model.ocl_warnings:
                    validation_warnings.extend(buml_model.ocl_warnings)

            except Exception as e:
                validation_warnings.append(f"OCL check warning: {str(e)}")

        # Step 3: Build unified response
        is_valid = len(validation_errors) == 0
        response = {
            "isValid": is_valid,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "message": "\u2705 Diagram is valid" if is_valid else "\u274c Validation failed"
        }

        # Add OCL-specific results if available
        if ocl_results:
            response["valid_constraints"] = ocl_results.get("valid_constraints", [])
            response["invalid_constraints"] = ocl_results.get("invalid_constraints", [])
            if ocl_results.get("message"):
                response["ocl_message"] = ocl_results["message"]

        return response

    except Exception as e:
        logger.error("Error in validate_diagram: %s", str(e))
        return {
            "isValid": False,
            "errors": [f"Unexpected error: {str(e)}"],
            "warnings": [],
            "message": "\u274c Validation failed"
        }


@router.post("/check-ocl", response_model=ValidationResponse)
async def check_ocl(input_data: DiagramInput):
    """
    Deprecated: Use /validate-diagram instead.
    This endpoint is kept for backwards compatibility and redirects to the new unified validation.
    """
    logger.warning("/check-ocl is deprecated. Use /validate-diagram instead.")
    return await validate_diagram(input_data)
