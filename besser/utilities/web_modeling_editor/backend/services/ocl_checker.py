from besser.BUML.notations.ocl.ocl_parser import ocl_parser
from besser.BUML.metamodel.structural import Class

def check_ocl_constraint(domain_model):
    """
    Check all OCL constraints in the domain model
    """
    try:
        if not domain_model.constraints:
            return {
                "success": False,
                "message": "No OCL constraints found in the model"
            }

        valid_constraints = []
        invalid_constraints = []

        for constraint in domain_model.constraints:
            try:
                result = ocl_parser(constraint, domain_model, None)
                if result is not None:  # If parser returns something, it's an error
                    invalid_constraints.append(f"❌ '{constraint.expression}'\n   Error: {result}")
                else:
                    valid_constraints.append(f"✅ '{constraint.expression}'")
            except Exception as e:
                invalid_constraints.append(f"❌ '{constraint.expression}'\n   Error: {str(e)}")

        # Create detailed message with line breaks
        message_parts = []
        if valid_constraints:
            message_parts.append("Valid constraints:")
            message_parts.extend(valid_constraints)
        if invalid_constraints:
            if valid_constraints:  # Add extra line break between valid and invalid sections
                message_parts.append("")
            message_parts.append("Invalid constraints:")
            message_parts.extend(invalid_constraints)

        return {
            "success": len(invalid_constraints) == 0,
            "message": "\n".join(message_parts)
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error checking OCL constraints: {str(e)}"
        }
