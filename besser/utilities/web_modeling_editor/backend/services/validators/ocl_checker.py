from besser.BUML.notations.ocl.OCLParserWrapper import OCLParserWrapper
from bocl.OCLWrapper import OCLWrapper
import re

def extract_context_class_name(expression):
    """Extract the context class name from an OCL expression"""
    try:
        # Look for pattern: context ClassName
        match = re.search(r'context\s+(\w+)', expression, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    except:
        return ""

def is_basic_ocl_syntax_valid(expression):
    """Basic OCL syntax validation without full parsing"""
    try:
        # Check for basic OCL structure
        if not expression.strip():
            return False
        
        # Must start with context
        if not expression.strip().lower().startswith('context'):
            return False
        
        # Check for balanced parentheses
        open_parens = expression.count('(')
        close_parens = expression.count(')')
        if open_parens != close_parens:
            return False
        
        # Check for basic OCL keywords that suggest valid structure
        ocl_keywords = ['inv', 'pre', 'post', 'self', 'collect', 'select', 'exists', 'forall', 'size']
        has_ocl_keywords = any(keyword in expression.lower() for keyword in ocl_keywords)
        
        return has_ocl_keywords
    except:
        return False

def check_ocl_constraint(domain_model, object_model = None):
    """
    Check all OCL constraints in the domain model
    """
    try:
        if not domain_model.constraints:
            return {
                "success": True,
                "message": "",
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
                        invalid_constraints.append(f"❌ '{constraint.expression}' - Error: Invalid OCL syntax")
                else:
                    # Check if there are instances of the context class in the object model
                    context_class_name = extract_context_class_name(constraint.expression)
                    context_instances = [obj for obj in object_model.objects 
                                        if hasattr(obj, 'classifier') and obj.classifier.name.lower() == context_class_name.lower()]

                    if not context_instances:
                        # No instances of the context class exist, skip evaluation
                        # valid_constraints.append(f"⚠️ '{constraint.expression}' - No instances of '{context_class_name}' found to evaluate constraint")
                        continue
                    
                    # Use evaluate method for OCLWrapper (evaluation with object model)
                    result = parser.evaluate(constraint)
                    if result is True:
                        valid_constraints.append(f"✅ '{constraint.expression}' - Evaluates to: True")
                    elif result is False:
                        invalid_constraints.append(f"❌ '{constraint.expression}' - Constraint violation: Evaluates to False")
                    else:
                        valid_constraints.append(f"✅ '{constraint.expression}' - Evaluates to: {result}")
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
