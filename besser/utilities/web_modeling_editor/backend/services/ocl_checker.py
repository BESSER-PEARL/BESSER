from besser.BUML.notations.ocl.OCLParserWrapper import OCLParserWrapper
from bocl.OCLWrapper import OCLWrapper

def check_ocl_constraint(domain_model, object_model = None):
    """
    Check all OCL constraints in the domain model
    """
    try:
        if not domain_model.constraints:
            return {
                "success": True,
                "message": "No OCL constraints found in the model",
                "valid_constraints": [],
                "invalid_constraints": []
            }

        valid_constraints = []
        invalid_constraints = []
        if object_model is None:
            parser = OCLParserWrapper(domain_model, None)
        else:
            parser = OCLWrapper(domain_model, object_model)

        for constraint in domain_model.constraints:
            try:
                if object_model is None:
                    # Use parse method for OCLParserWrapper (syntax checking only)
                    result = parser.parse(constraint)
                    if result is True:  # Parser returns True for valid constraints
                        valid_constraints.append(f"✅ '{constraint.expression}'")
                    else:
                        invalid_constraints.append(f"❌ '{constraint.expression}' - Error: Parsing failed \n")
                else:
                    # Use evaluate method for OCLWrapper (evaluation with object model)
                    result = parser.evaluate(constraint)
                    if result is True:
                        valid_constraints.append(f"✅ '{constraint.expression}' - Evaluates to: True \n")
                    elif result is False:
                        valid_constraints.append(f"⚠️ '{constraint.expression}' - Evaluates to: False \n")
                    else:
                        valid_constraints.append(f"✅ '{constraint.expression}' - Evaluates to: {result} \n")
            except Exception as e:
                invalid_constraints.append(f"❌ '{constraint.expression}' - Error: {str(e)} \n")

        return {
            "success": len(invalid_constraints) == 0,
            "valid_constraints": valid_constraints,
            "invalid_constraints": invalid_constraints,
            "message": f"Found {len(valid_constraints)} valid and {len(invalid_constraints)} invalid constraints"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"{str(e)}",
            "valid_constraints": [],
            "invalid_constraints": []
        }
