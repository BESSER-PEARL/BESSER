from antlr4 import InputStream, CommonTokenStream
from bocl.OCLWrapper import OCLWrapper
from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.metamodel.structural import Class
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.error_handling import BOCLErrorListener, BOCLSyntaxError
import re


def _parse_only(expression: str) -> None:
    """Run the OCL lexer + parser for syntax validation without evaluating.

    Raises BOCLSyntaxError if the expression is syntactically invalid.
    """
    input_stream = InputStream(expression)
    lexer = BOCLLexer(input_stream)
    lexer.removeErrorListeners()
    error_listener = BOCLErrorListener()
    lexer.addErrorListener(error_listener)

    stream = CommonTokenStream(lexer)
    parser = BOCLParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)

    parser.oclFile()

    if error_listener.has_errors():
        raise BOCLSyntaxError(error_listener.errors)

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

def _collect_all_constraints(domain_model):
    """Yield ``(label, constraint)`` for every OCL constraint in the model.

    The validator no longer looks only at ``domain_model.constraints`` —
    it also walks each class's methods' ``.pre`` and ``.post`` lists, so
    method contracts are validated alongside class invariants. Labels
    include enough disambiguation (class name, kind, target method) to
    map the result back to a specific constraint box in the editor.
    """
    for c in domain_model.constraints:
        ctx_name = c.context.name if getattr(c, "context", None) is not None else "?"
        yield (f"[{ctx_name} inv {c.name}]", c)
    for cls in domain_model.types:
        if not isinstance(cls, Class):
            continue
        for method in getattr(cls, "methods", []) or []:
            for c in getattr(method, "pre", []) or []:
                yield (f"[{cls.name}::{method.name} pre {c.name}]", c)
            for c in getattr(method, "post", []) or []:
                yield (f"[{cls.name}::{method.name} post {c.name}]", c)


def check_ocl_constraint(domain_model, object_model = None):
    """
    Check all OCL constraints in the domain model — class invariants plus
    method preconditions and postconditions.
    """
    try:
        constraints = list(_collect_all_constraints(domain_model))
        if not constraints:
            return {
                "success": True,
                "message": "",
                "valid_constraints": [],
                "invalid_constraints": []
            }

        valid_constraints = []
        invalid_constraints = []
        parser = OCLWrapper(domain_model, object_model)

        for label, constraint in constraints:
            description = getattr(constraint, "description", None)
            # Suffix appended to every message so non-technical users see the
            # plain-language reason alongside the raw OCL expression.
            explanation_suffix = f" — {description}" if description else ""
            try:
                if object_model is None:
                    # No object model -> syntax-only check.
                    # OCLConstraint instances were parsed at conversion time
                    # and carry an AST, so syntax is valid by construction.
                    # Bare Constraint (legacy or non-OCL languages) still gets
                    # the lex/parse round-trip via _parse_only.
                    if isinstance(constraint, OCLConstraint):
                        valid_constraints.append(
                            f"✅ {label} '{constraint.expression}'{explanation_suffix}"
                        )
                    else:
                        try:
                            _parse_only(constraint.expression)
                            valid_constraints.append(
                                f"✅ {label} '{constraint.expression}'{explanation_suffix}"
                            )
                        except BOCLSyntaxError as syntax_err:
                            invalid_constraints.append(
                                f"❌ {label} '{constraint.expression}' - {syntax_err}{explanation_suffix}"
                            )
                else:
                    # Evaluate against the object model. Skip if there are no
                    # instances of the context class to evaluate against.
                    context_class_name = constraint.context.name if getattr(constraint, "context", None) is not None else extract_context_class_name(constraint.expression)
                    context_instances = [
                        obj for obj in object_model.objects
                        if hasattr(obj, 'classifier') and obj.classifier.name.lower() == context_class_name.lower()
                    ]

                    if not context_instances:
                        continue

                    result = parser.evaluate(constraint)
                    if result is True:
                        valid_constraints.append(
                            f"✅ {label} '{constraint.expression}' - Evaluates to: True{explanation_suffix}"
                        )
                    elif result is False:
                        # Prefer the natural-language description as the primary
                        # violation reason when one is provided.
                        violation_reason = description if description else "Evaluates to False"
                        invalid_constraints.append(
                            f"❌ {label} '{constraint.expression}' - Constraint violation: {violation_reason}"
                        )
                    else:
                        valid_constraints.append(
                            f"✅ {label} '{constraint.expression}' - Evaluates to: {result}{explanation_suffix}"
                        )
            except Exception as e:
                invalid_constraints.append(
                    f"❌ {label} '{constraint.expression}' - Error: {str(e)}{explanation_suffix} \n"
                )

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
