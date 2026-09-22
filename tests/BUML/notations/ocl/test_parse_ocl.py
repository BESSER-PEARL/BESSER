"""Tests for besser.BUML.notations.ocl.api.parse_ocl."""

import pytest

from besser.BUML.metamodel.ocl.ocl import OCLConstraint
from besser.BUML.notations.ocl import parse_ocl, BOCLSyntaxError


def test_parse_returns_ocl_constraint(model):
    result = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(result, OCLConstraint)
    assert result.context.name == "Employee"
    assert result.language == "OCL"


def test_parse_resolves_context_from_text(model):
    result = parse_ocl("context Department inv: self.name <> ''", model)
    assert result.context.name == "Department"


def test_parse_explicit_context_class_overrides_text(model):
    employee = next(t for t in model.types if getattr(t, "name", None) == "Employee")
    # No header in the text — but explicit context_class supplied.
    result = parse_ocl(
        "context Employee inv: self.age > 16",
        model,
        context_class=employee,
    )
    assert result.context is employee


def test_parse_raises_for_unknown_context(model):
    with pytest.raises(ValueError, match="Nonexistent"):
        parse_ocl("context Nonexistent inv: 1 = 1", model)


def test_parse_raises_for_missing_header(model):
    # No `context X inv:` prefix and no explicit context_class → ValueError.
    with pytest.raises(ValueError, match="header"):
        parse_ocl("self.age > 16", model)


def test_parse_raises_bocl_syntax_error_on_lex_or_parse_failure(model):
    # Unbalanced parenthesis → ANTLR error.
    with pytest.raises(BOCLSyntaxError):
        parse_ocl("context Employee inv: (self.age > 16", model)


def test_parse_raises_bocl_syntax_error_on_unresolved_property(model):
    # Property doesn't exist on the context class — must surface as
    # BOCLSyntaxError, not a bare Exception.
    with pytest.raises(BOCLSyntaxError, match="not found"):
        parse_ocl("context Employee inv: self.nonexistent > 16", model)


def test_parse_iterator_constraint(model):
    result = parse_ocl(
        "context Department inv: self.employee->forAll(e | e.age > 16)",
        model,
    )
    assert isinstance(result, OCLConstraint)
    assert result.context.name == "Department"


# Tests for OCLConstraint.ast / .expression separation (Pre-work B)

def test_ocl_constraint_ast_is_the_parsed_tree(model):
    """``OCLConstraint.ast`` exposes the parsed AST."""
    from besser.BUML.metamodel.ocl.ocl import OCLExpression
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(constraint.ast, OCLExpression)


def test_ocl_constraint_expression_is_source_text(model):
    """``OCLConstraint.expression`` returns pretty-printed OCL source text."""
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    assert isinstance(constraint.expression, str)
    assert "self.age" in constraint.expression


def test_ocl_constraint_rejects_non_ast_expression(model):
    """Constructing OCLConstraint with a string raises TypeError."""
    employee = next(t for t in model.types if getattr(t, "name", None) == "Employee")
    with pytest.raises(TypeError, match="OCLExpression"):
        OCLConstraint(name="bad", context=employee, expression="self.age > 16")


def test_ocl_constraint_ast_setter_refreshes_expression(model):
    """Reassigning .ast updates .expression to a fresh pretty-print."""
    c1 = parse_ocl("context Employee inv: self.age > 16", model)
    c2 = parse_ocl("context Employee inv: self.age > 99", model)
    c1.ast = c2.ast
    assert "99" in c1.expression


def test_ocl_constraint_ast_setter_rejects_non_ast(model):
    constraint = parse_ocl("context Employee inv: self.age > 16", model)
    with pytest.raises(TypeError, match="OCLExpression"):
        constraint.ast = "not an ast"


def test_ocl_allInstances(model):
    constraint = parse_ocl("context Department inv: Department.allInstances()->size() > 16", model)


# ===========================================================================
# Tests for newly added grammar constructs
# ===========================================================================

# ---------------------------------------------------------------------------
# allInstances — dot form  (expression.allInstances())
# ---------------------------------------------------------------------------

def test_allinstances_dot_form_returns_constraint(model):
    """Employee.allInstances() dot form parses without error."""
    result = parse_ocl(
        "context Employee inv: Employee.allInstances()->size() > 0", model
    )
    assert isinstance(result, OCLConstraint)


def test_allinstances_dot_form_ast_structure(model):
    """Employee.allInstances() dot form produces ALLInstances OCE with a TypeExp source."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression, TypeExp
    constraint = parse_ocl(
        "context Employee inv: Employee.allInstances()->size() > 0", model
    )
    # root is the comparison expression >  →  arguments[0] = Size, arguments[2] = 0
    root = constraint.ast
    size_exp = root.arguments[0]
    assert isinstance(size_exp, OperationCallExpression)
    assert size_exp.operation == "Size"
    all_inst = size_exp.source
    assert isinstance(all_inst, OperationCallExpression)
    assert all_inst.operation == "ALLInstances"
    assert isinstance(all_inst.source, TypeExp)
    assert all_inst.source.name == "Employee"


def test_allinstances_dot_form_matches_doublecolon_form(model):
    """Employee.allInstances() and Employee::allInstances() produce identical operation names."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
    dot_form = parse_ocl(
        "context Employee inv: Employee.allInstances()->size() > 0", model
    )
    colon_form = parse_ocl(
        "context Employee inv: Employee::allInstances()->size() > 0", model
    )
    dot_op = dot_form.ast.arguments[0].source
    colon_op = colon_form.ast.arguments[0].source
    assert isinstance(dot_op, OperationCallExpression)
    assert isinstance(colon_op, OperationCallExpression)
    assert dot_op.operation == colon_op.operation == "ALLInstances"


# ---------------------------------------------------------------------------
# asSet  (expression->asSet())
# ---------------------------------------------------------------------------

def test_arrow_asset_returns_constraint(model):
    """self.employee->asSet() parses without error."""
    result = parse_ocl(
        "context Department inv: self.employee->asSet()->size() > 0", model
    )
    assert isinstance(result, OCLConstraint)


def test_arrow_asset_ast_structure(model):
    """->asSet() wraps the source collection in an ASSET OperationCallExpression."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
    constraint = parse_ocl(
        "context Department inv: self.employee->asSet()->size() > 0", model
    )
    root = constraint.ast
    size_exp = root.arguments[0]
    assert isinstance(size_exp, OperationCallExpression)
    assert size_exp.operation == "Size"
    as_set_exp = size_exp.source
    assert isinstance(as_set_exp, OperationCallExpression)
    assert as_set_exp.operation == "ASSET"
    assert as_set_exp.arguments == []


def test_arrow_asset_iterator_type_inference(model):
    """Iterator element-type inference traverses ->asSet() transparently.

    self.employee->asSet() still yields Employee elements, so the forAll body
    can navigate e.age without an explicit type annotation on e.
    """
    result = parse_ocl(
        "context Department inv: self.employee->asSet()->forAll(e | e.age > 16)",
        model,
    )
    assert isinstance(result, OCLConstraint)


# ---------------------------------------------------------------------------
# intersection  (expression->intersection(expression))
# ---------------------------------------------------------------------------

def test_arrow_intersection_returns_constraint(model):
    """->intersection(...) parses without error."""
    result = parse_ocl(
        "context Employee inv: "
        "Employee::allInstances()->intersection(Employee::allInstances())->size() > 0",
        model,
    )
    assert isinstance(result, OCLConstraint)


def test_arrow_intersection_ast_structure(model):
    """->intersection(arg) produces an INTERSECTION OCE with exactly one argument."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
    constraint = parse_ocl(
        "context Employee inv: "
        "Employee::allInstances()->intersection(Employee::allInstances())->size() > 0",
        model,
    )
    root = constraint.ast
    size_exp = root.arguments[0]
    assert isinstance(size_exp, OperationCallExpression)
    assert size_exp.operation == "Size"
    inter_exp = size_exp.source
    assert isinstance(inter_exp, OperationCallExpression)
    assert inter_exp.operation == "INTERSECTION"
    assert len(inter_exp.arguments) == 1


def test_arrow_intersection_argument_is_allinstances(model):
    """The argument of ->intersection() is itself an ALLInstances expression."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
    constraint = parse_ocl(
        "context Employee inv: "
        "Employee::allInstances()->intersection(Employee::allInstances())->size() > 0",
        model,
    )
    inter_exp = constraint.ast.arguments[0].source
    arg = inter_exp.arguments[0]
    assert isinstance(arg, OperationCallExpression)
    assert arg.operation == "ALLInstances"


# ---------------------------------------------------------------------------
# isUnique  (expression->isUnique(expression))
# ---------------------------------------------------------------------------

def test_arrow_isunique_returns_constraint(model):
    """->isUnique(expr) parses without error."""
    result = parse_ocl(
        "context Employee inv: Employee::allInstances()->isUnique(self.age)",
        model,
    )
    assert isinstance(result, OCLConstraint)


def test_arrow_isunique_ast_structure(model):
    """->isUnique(expr) is an ISUNIQUE OCE with one argument and ALLInstances as source."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
    constraint = parse_ocl(
        "context Employee inv: Employee::allInstances()->isUnique(self.age)",
        model,
    )
    isunique_exp = constraint.ast
    assert isinstance(isunique_exp, OperationCallExpression)
    assert isunique_exp.operation == "ISUNIQUE"
    assert len(isunique_exp.arguments) == 1
    assert isinstance(isunique_exp.source, OperationCallExpression)
    assert isunique_exp.source.operation == "ALLInstances"


def test_arrow_isunique_argument_is_property_navigation(model):
    """The argument of ->isUnique() can be a property navigation expression."""
    from besser.BUML.metamodel.ocl.ocl import OperationCallExpression, PropertyCallExpression
    constraint = parse_ocl(
        "context Employee inv: Employee::allInstances()->isUnique(self.age)",
        model,
    )
    isunique_exp = constraint.ast
    arg = isunique_exp.arguments[0]
    assert isinstance(arg, PropertyCallExpression)
    assert arg.property.name == "age"


def test_arrow_isunique_with_collection_property(model):
    """->isUnique on a collection property navigated from self parses without error."""
    result = parse_ocl(
        "context Department inv: self.employee->isUnique(self.minSalary)",
        model,
    )
    assert isinstance(result, OCLConstraint)
