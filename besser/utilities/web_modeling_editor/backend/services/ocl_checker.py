from besser.BUML.notations.ocl.OCLParserWrapper import OCLParserWrapper

def check_ocl_constraint(domain_model):
    """
    Check all OCL constraints in the domain model
    """
    try:
        if not domain_model.constraints:
            return {
                "success": True,
                "message": "No OCL constraints found in the model"
            }

        valid_constraints = []
        invalid_constraints = []
        parser = OCLParserWrapper(domain_model, None)

        for constraint in domain_model.constraints:
            try:
                result = parser.parse(constraint)
                if result is True:  # Parser returns True for valid constraints
                    valid_constraints.append(f"✅ '{constraint.expression}'")
                else:
                    invalid_constraints.append(f"❌ '{constraint.expression}'\n   Error: Parsing failed")
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
