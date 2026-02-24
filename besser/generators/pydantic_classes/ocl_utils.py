"""
OCL Constraint Utilities for Pydantic Generator

This module uses the BESSER OCL parser infrastructure to extract validation 
information from OCL constraint expressions for use in Pydantic field validators.

It leverages the existing ANTLR-based parser in besser/BUML/notations/ocl
and traverses the parsed OCL expression tree similar to the B-OCL-Interpreter.
"""

import re
from typing import Optional, Dict, Any, List
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLListener import BOCLListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker

from besser.BUML.metamodel.ocl.ocl import (
    OperationCallExpression, PropertyCallExpression, 
    IntegerLiteralExpression, RealLiteralExpression, 
    StringLiteralExpression, BooleanLiteralExpression,
    DateLiteralExpression, InfixOperator, LoopExp
)


def parse_ocl_constraint(constraint, domain_model) -> Optional[Dict[str, Any]]:
    """
    Parse an OCL constraint using the BESSER OCL parser and extract validation info.
    
    This uses the ANTLR-based parser to properly parse OCL expressions,
    similar to the B-OCL-Interpreter's update_logical_exp function.
    
    Args:
        constraint: A Constraint object from the domain model
        domain_model: The DomainModel for context resolution
        
    Returns:
        A dict with keys: property, operator, value, python_operator
        or for compound expressions: property, python_expression, message.
        Returns None if the expression cannot be parsed for validation.
    """
    try:
        # Parse using the ANTLR OCL parser - similar to OCLParserWrapper.parse()
        # but we keep a reference to the rootHandler
        input_stream = InputStream(constraint.expression)
        root_handler = Root_Handler(domain_model, None)
        root_handler.set_context(constraint.context)
        
        lexer = BOCLLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = BOCLParser(stream)
        tree = parser.oclFile()
        
        listener = BOCLListener(root_handler)
        listener.preprocess(constraint.expression)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        
        # Get the parsed expression tree from the root handler
        root = root_handler.get_root()
        
        if root is None:
            # Fallback to regex parsing if no root was produced
            return _fallback_parse(constraint.expression)
        
        # Extract validation info from the parsed tree
        result = _extract_from_tree(root)
        if result:
            return result
        
        # If tree extraction failed, try fallback
        return _fallback_parse(constraint.expression)
        
    except Exception as e:
        # Fallback to regex parsing if OCL parser fails
        return _fallback_parse(constraint.expression)


def _extract_from_tree(tree) -> Optional[Dict[str, Any]]:
    """
    Extract validation info from a parsed OCL expression tree.
    
    Handles simple expressions of the form: self.property <operator> value
    Similar to the update_logical_exp function in B-OCL-Interpreter.
    
    Args:
        tree: The root of the parsed OCL expression tree
        
    Returns:
        Dict with validation info or None
    """
    if isinstance(tree, OperationCallExpression):
        result = _extract_from_operation(tree)
        if result:
            return result
    
    # Check if we need to traverse source to find the operation
    if hasattr(tree, 'source') and tree.source is not None:
        return _extract_from_tree(tree.source)
    
    return None


def _extract_from_operation(op_exp: OperationCallExpression) -> Optional[Dict[str, Any]]:
    """
    Extract validation info from an OperationCallExpression.
    
    Based on the B-OCL-Interpreter's handling in update_logical_exp (lines 417-439).
    
    Args:
        op_exp: An OperationCallExpression node
        
    Returns:
        Dict with validation info or None
    """
    args = op_exp.arguments
    
    property_name = None
    operator = None
    value = None
    value_repr = None
    value_type = None
    
    for arg in args:
        if isinstance(arg, PropertyCallExpression):
            property_name = arg.property.name
        elif isinstance(arg, InfixOperator):
            operator = arg.get_infix_operator()
        elif isinstance(arg, IntegerLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'int'
        elif isinstance(arg, RealLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'float'
        elif isinstance(arg, StringLiteralExpression):
            value = arg.value
            value_repr = f"'{arg.value}'"
            value_type = 'str'
        elif isinstance(arg, BooleanLiteralExpression):
            value = arg.value
            value_repr = str(arg.value)
            value_type = 'bool'
        elif isinstance(arg, DateLiteralExpression):
            value = arg.value
            value_repr = f"'{arg.value}'"
            value_type = 'date'
    
    # Also check the referredOperation for the operator
    if operator is None and op_exp.referredOperation is not None:
        if hasattr(op_exp.referredOperation, 'get_infix_operator'):
            operator = op_exp.referredOperation.get_infix_operator()
        elif hasattr(op_exp.referredOperation, 'operator'):
            operator = op_exp.referredOperation.operator
    
    if property_name is None or operator is None or value is None:
        return None
    
    # Map OCL operators to Python operators
    operator_map = {
        '>': '>',
        '<': '<',
        '>=': '>=',
        '<=': '<=',
        '=': '==',
        '<>': '!='
    }
    
    python_operator = operator_map.get(operator, operator)
    
    return {
        'property': property_name,
        'operator': operator,
        'python_operator': python_operator,
        'value': value,
        'value_repr': value_repr,
        'value_type': value_type
    }


def _extract_expression_body(expression: str) -> str:
    """
    Extract the OCL expression body from a full constraint string.
    """
    expression = expression.strip()
    match = re.search(
        r"context\s+\w+\s+inv(?:\s+\w+)?\s*:\s*(.+)",
        expression,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return expression


def _apply_outside_quotes(text: str, transform) -> str:
    """
    Apply a transformation function only to text outside single-quoted strings.
    """
    parts = text.split("'")
    for idx in range(0, len(parts), 2):
        parts[idx] = transform(parts[idx])
    return "'".join(parts)


def _replace_outside_quotes(text: str, pattern: str, repl: str) -> str:
    """
    Regex replace outside single-quoted strings.
    """
    return _apply_outside_quotes(text, lambda part: re.sub(pattern, repl, part))


def _collapse_whitespace_outside_quotes(text: str) -> str:
    """
    Collapse consecutive whitespace outside single-quoted strings.
    """
    return _apply_outside_quotes(text, lambda part: re.sub(r"\s+", " ", part)).strip()


def _normalize_ocl_expression(expression: str, property_name: str) -> str:
    """
    Normalize an OCL expression into a Python expression for a single property.
    """
    normalized = expression
    normalized = _replace_outside_quotes(normalized, r"<>", "!=")
    normalized = _replace_outside_quotes(normalized, r"(?<![<>=!])=(?!=)", "==")
    normalized = _replace_outside_quotes(
        normalized,
        rf"\bself\.{re.escape(property_name)}\b",
        "v"
    )
    return _collapse_whitespace_outside_quotes(normalized)


def _fallback_parse(expression: str) -> Optional[Dict[str, Any]]:
    """
    Fallback regex-based parser for simple OCL constraint expressions.
    Used when the ANTLR parser doesn't produce usable results.
    
    Args:
        expression: The OCL constraint expression string
        
    Returns:
        A dict with keys: property, operator, value, python_operator
        or for compound expressions: property, python_expression, message.
        Returns None if the expression cannot be parsed.
    """
    # Extract the core OCL expression body (after "inv:")
    expression_body = _extract_expression_body(expression)
    if not expression_body:
        return None
    
    # Identify properties referenced in the expression
    properties = re.findall(r"\bself\.(\w+)\b", expression_body)
    if not properties:
        return None
    
    unique_properties = set(properties)
    if len(unique_properties) != 1:
        # Only support single-property constraints for field validators
        return None
    
    property_name = unique_properties.pop()
    multiple_references = len(properties) > 1
    
    # Pattern to match: self.<property> <operator> <value>
    simple_pattern = rf"self\.{re.escape(property_name)}\s*(>=|<=|<>|>|<|=)\s*(.+)"
    match = re.fullmatch(simple_pattern, expression_body.strip())
    if not match or multiple_references:
        # Handle compound expressions like "self.age > 10 and self.age < 20"
        python_expression = _normalize_ocl_expression(expression_body, property_name)
        message_expression = _replace_outside_quotes(
            python_expression,
            r"\bv\b",
            ""
        )
        message_expression = _collapse_whitespace_outside_quotes(message_expression)
        return {
            'property': property_name,
            'python_expression': python_expression,
            'message': f"{property_name} must be {message_expression}"
        }
    
    operator = match.group(1)
    value_str = match.group(2).strip()
    
    # Map OCL operators to Python operators
    operator_map = {
        '>': '>',
        '<': '<',
        '>=': '>=',
        '<=': '<=',
        '=': '==',
        '<>': '!='
    }
    
    python_operator = operator_map.get(operator)
    if not python_operator:
        return None
    
    # Parse the value
    parsed_value = _parse_value(value_str)
    
    return {
        'property': property_name,
        'operator': operator,
        'python_operator': python_operator,
        'value': parsed_value['value'],
        'value_repr': parsed_value['repr'],
        'value_type': parsed_value['type']
    }


def _parse_value(value_str: str) -> Dict[str, Any]:
    """
    Parse an OCL value and determine its type.
    """
    value_str = value_str.strip()
    
    # Check for integer
    try:
        int_val = int(value_str)
        return {'value': int_val, 'repr': str(int_val), 'type': 'int'}
    except ValueError:
        pass
    
    # Check for float
    try:
        float_val = float(value_str)
        return {'value': float_val, 'repr': str(float_val), 'type': 'float'}
    except ValueError:
        pass
    
    # Check for string (single quotes in OCL)
    if value_str.startswith("'") and value_str.endswith("'"):
        str_val = value_str[1:-1]
        return {'value': str_val, 'repr': f"'{str_val}'", 'type': 'str'}
    
    # Check for boolean
    if value_str.lower() in ('true', 'false'):
        bool_val = value_str.lower() == 'true'
        return {'value': bool_val, 'repr': str(bool_val), 'type': 'bool'}
    
    # Default: treat as raw value
    return {'value': value_str, 'repr': value_str, 'type': 'unknown'}


def get_constraints_for_class(constraints: set, class_name: str, domain_model) -> List[Dict[str, Any]]:
    """
    Get all parsed constraints for a specific class.
    
    Args:
        constraints: Set of Constraint objects from the domain model
        class_name: Name of the class to get constraints for
        domain_model: The DomainModel for parsing context
        
    Returns:
        List of parsed constraint dicts
    """
    parsed = []
    
    for constraint in constraints:
        if constraint.context.name == class_name and constraint.language == "OCL":
            result = parse_ocl_constraint(constraint, domain_model)
            if result:
                result['constraint_name'] = constraint.name
                parsed.append(result)
    
    return parsed


def build_constraints_map(domain_model) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping from class names to their parsed OCL constraints.
    
    Args:
        domain_model: The DomainModel object
        
    Returns:
        Dict mapping class name -> list of parsed constraints
    """
    constraints_map = {}
    
    if not domain_model.constraints:
        return constraints_map
    
    for cls in domain_model.get_classes():
        class_constraints = get_constraints_for_class(
            domain_model.constraints, 
            cls.name,
            domain_model
        )
        if class_constraints:
            constraints_map[cls.name] = class_constraints
    
    return constraints_map
