"""
Constraint Suggestion Service

Uses LLM to generate helpful suggestions for fixing invalid OCL constraints.
"""

import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_constraint_suggestion(constraint: str) -> Dict[str, any]:
    """
    Generate a suggestion for fixing an invalid constraint using LLM.

    Args:
        constraint: The full constraint string from validation

    Returns:
        Dictionary with:
        - issue: Brief description of the problem
        - suggestion: Detailed suggestion on how to fix
        - steps: List of actionable steps
    """
    # Log what we received
    logger.info(f"Generating suggestion for constraint: {constraint[:200]}...")

    # Parse the constraint to extract useful information
    class_name = extract_class_name(constraint)
    constraint_rule = extract_constraint_rule(constraint)

    logger.info(f"Extracted class_name: {class_name}, constraint_rule: {constraint_rule[:100]}...")

    # Check if OpenAI or Anthropic API key is available
    openai_key = os.getenv('OPENAI_API_KEY')

    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if openai_key:
        return generate_with_openai(class_name, constraint_rule, constraint, openai_key)
    elif anthropic_key:
        return generate_with_anthropic(class_name, constraint_rule, constraint, anthropic_key)
    else:
        # Fallback to rule-based suggestion
        return generate_fallback_suggestion(class_name, constraint_rule, constraint)


def extract_class_name(constraint: str) -> str:
    """Extract class name from constraint string."""
    import re

    # Try OCL bracket format: [ClassName inv ...]
    match = re.search(r'\[([A-Za-z_][A-Za-z0-9_]*)\s+inv\s+', constraint)
    if match:
        return match.group(1)

    # Try OCL context format: context ClassName inv
    match = re.search(r'context\s+([A-Za-z_][A-Za-z0-9_]*)\s+inv\s+', constraint)
    if match:
        return match.group(1)

    return "Unknown"


def extract_constraint_rule(constraint: str) -> str:
    """Extract the actual constraint rule from the full string."""
    import re

    # Try to extract from single quotes
    match = re.search(r"'([^']*)'", constraint)
    if match:
        return match.group(1)

    # Try to extract from double quotes
    match = re.search(r'"([^"]*)"', constraint)
    if match:
        return match.group(1)

    return constraint


def generate_with_openai(class_name: str, constraint_rule: str, full_constraint: str, api_key: str) -> Dict[str, any]:
    """Generate suggestion using OpenAI API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Detect if this is a syntax error warning or a constraint violation
        is_syntax_error = 'Invalid OCL syntax' in full_constraint or 'Property' in full_constraint and 'not found' in full_constraint

        if is_syntax_error:
            prompt = f"""You are a UML/OCL expert helping users fix OCL constraint syntax errors.

Class: {class_name}
Constraint: {constraint_rule}
Full error message: {full_constraint}

This is a SYNTAX ERROR in the constraint definition itself, not a validation failure.

Provide a helpful suggestion in JSON format with these fields:
- issue: A brief description of the syntax error (1-2 sentences)
- suggestion: Clear explanation of how to fix the constraint syntax (2-3 sentences)
- steps: A list of 2-4 actionable steps to correct the constraint

Focus on fixing the CONSTRAINT DEFINITION (property names, syntax, etc.), not object values.
Keep it concise and actionable."""
        else:
            prompt = f"""You are a UML/OCL expert helping users fix constraint violations in their object diagrams.

Class: {class_name}
Constraint: {constraint_rule}
Full violation message: {full_constraint}

Provide a helpful suggestion in JSON format with these fields:
- issue: A brief, clear description of what's wrong (1-2 sentences)
- suggestion: A detailed explanation of how to fix it (2-3 sentences)
- steps: A list of 2-4 actionable steps to fix the issue

Focus on practical advice for fixing the object instances, not changing the constraint itself.
Keep it concise and actionable."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful UML/OCL expert assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=500
        )

        import json
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return generate_fallback_suggestion(class_name, constraint_rule, full_constraint)


def generate_with_anthropic(class_name: str, constraint_rule: str, full_constraint: str, api_key: str) -> Dict[str, any]:
    """Generate suggestion using Anthropic Claude API."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Detect if this is a syntax error warning or a constraint violation
        is_syntax_error = 'Invalid OCL syntax' in full_constraint or ('Property' in full_constraint and 'not found' in full_constraint)

        if is_syntax_error:
            prompt = f"""You are a UML/OCL expert helping users fix OCL constraint syntax errors.

Class: {class_name}
Constraint: {constraint_rule}
Full error message: {full_constraint}

This is a SYNTAX ERROR in the constraint definition itself, not a validation failure.

Provide a helpful suggestion in JSON format with these fields:
- issue: A brief description of the syntax error (1-2 sentences)
- suggestion: Clear explanation of how to fix the constraint syntax (2-3 sentences)
- steps: A list of 2-4 actionable steps to correct the constraint

Focus on fixing the CONSTRAINT DEFINITION (property names, syntax, etc.), not object values.
Keep it concise and actionable."""
        else:
            prompt = f"""You are a UML/OCL expert helping users fix constraint violations in their object diagrams.

Class: {class_name}
Constraint: {constraint_rule}
Full violation message: {full_constraint}

Provide a helpful suggestion in JSON format with these fields:
- issue: A brief, clear description of what's wrong (1-2 sentences)
- suggestion: A detailed explanation of how to fix it (2-3 sentences)
- steps: A list of 2-4 actionable steps to fix the issue

Focus on practical advice for fixing the object instances, not changing the constraint itself.
Keep it concise and actionable."""

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        import json
        # Extract JSON from response
        content = message.content[0].text
        # Try to find JSON in the response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(content[json_start:json_end])
            return result
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return generate_fallback_suggestion(class_name, constraint_rule, full_constraint)


def generate_fallback_suggestion(class_name: str, constraint_rule: str, full_constraint: str) -> Dict[str, any]:
    """Generate a basic rule-based suggestion when AI is not available."""
    import re

    print(f"\n\n=== FALLBACK DEBUG ===")
    print(f"full_constraint: {full_constraint[:200]}")
    print(f"'Invalid OCL syntax' in full_constraint: {'Invalid OCL syntax' in full_constraint}")
    print(f"'Property' in full_constraint: {'Property' in full_constraint}")
    print(f"'not found' in full_constraint: {'not found' in full_constraint}")

    logger.info(f"Fallback: Checking full_constraint for syntax error: {full_constraint[:150]}...")

    # Check if this is a syntax error warning
    is_syntax_error = 'Invalid OCL syntax' in full_constraint or ('Property' in full_constraint and 'not found' in full_constraint)

    print(f"is_syntax_error = {is_syntax_error}")
    print(f"=== END DEBUG ===\n\n")

    logger.info(f"Fallback: is_syntax_error = {is_syntax_error}")

    if is_syntax_error:
        # Extract the problematic property name if present
        property_match = re.search(r"Property '([^']+)' not found", full_constraint)
        if property_match:
            bad_property = property_match.group(1)
            issue = f"The constraint references a non-existent property '{bad_property}' in class '{class_name}'."
            suggestion = f"Check the constraint definition. The property '{bad_property}' does not exist in the '{class_name}' class. Verify the correct property name from your class diagram."
            steps = [
                f"Open the '{class_name}' class in your Class Diagram",
                "Review the list of attributes/properties",
                f"Correct '{bad_property}' to the actual property name in the constraint",
                "Save and re-validate the constraint"
            ]
        else:
            issue = f"The constraint has invalid OCL syntax."
            suggestion = "Review the constraint definition for syntax errors such as typos in property names, missing operators, or incorrect OCL keywords."
            steps = [
                "Check the constraint text for typos",
                "Verify all property names exist in the class",
                "Ensure OCL operators and keywords are correct",
                "Re-validate after fixing the syntax"
            ]
    else:
        # Parse common constraint violation patterns
        issue = f"Objects of class '{class_name}' violate the constraint."

        suggestion = "Review the constraint rule and ensure all object instances satisfy the specified conditions."

        steps = [
            f"Identify all '{class_name}' objects in your Object Diagram",
            "Check each object's attribute values against the constraint",
            "Modify the values that cause the violation",
            "Re-run validation to verify the fix"
        ]

        # Try to provide more specific guidance based on constraint patterns
        if re.search(r'>\s*0', constraint_rule) or re.search(r'>=\s*0', constraint_rule):
            issue = f"One or more '{class_name}' objects have negative or zero values."
            suggestion = "The constraint requires positive values. Check for any attributes with negative or zero values and change them to positive numbers."
            steps = [
                f"Find all '{class_name}' objects in the diagram",
                "Look for attributes with negative or zero values",
                "Change these values to positive numbers (greater than 0)",
                "Save and re-validate the diagram"
            ]
        elif re.search(r'<\s*0', constraint_rule):
            issue = f"One or more '{class_name}' objects have non-negative values."
            suggestion = "The constraint requires negative values. Ensure all relevant attributes are less than zero."
        elif 'null' in constraint_rule.lower() or 'undefined' in constraint_rule.lower():
            issue = f"Some '{class_name}' objects have null or empty values."
            suggestion = "The constraint requires all values to be defined. Ensure no attributes are left empty or null."
            steps = [
                f"Check all '{class_name}' objects for empty/null values",
                "Fill in missing attribute values",
                "Ensure all required fields have valid data",
                "Re-validate after making changes"
            ]

    return {
        "issue": issue,
        "suggestion": suggestion,
        "steps": steps
    }
