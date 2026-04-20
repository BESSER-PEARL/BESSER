"""
OCL constraint parsing utilities for converting JSON to BUML format.
"""

import re
from besser.BUML.metamodel.structural import DomainModel, Constraint


def _extract_inline_description(block: str) -> tuple[str, str]:
    """Split an OCL constraint block into its expression and an inline description.

    OCL supports ``--`` single-line comments. We treat any text following ``--``
    on a line as a natural-language description of the constraint. The returned
    expression has the comment stripped so it can be fed to the OCL parser
    without relying on comment-handling behaviour.

    Returns:
        tuple[str, str]: ``(expression_without_comments, description_or_empty_string)``.
    """
    description_parts: list[str] = []
    cleaned_lines: list[str] = []
    for raw_line in block.split('\n'):
        match = re.search(r'--\s*(.*?)\s*$', raw_line)
        if match and match.group(1):
            description_parts.append(match.group(1))
            raw_line = raw_line[:match.start()].rstrip()
        cleaned_lines.append(raw_line)
    cleaned = '\n'.join(cleaned_lines)
    description = ' '.join(description_parts).strip()
    return cleaned, description


def process_ocl_constraints(
    ocl_text: str,
    domain_model: DomainModel,
    counter: int,
    default_description: str = None,
) -> tuple[list, list]:
    """Process OCL constraints and convert them to BUML Constraint objects.

    Supports multiple constraints separated by ``context`` keywords in the
    same OCL text block.  Each ``context ... inv ...`` section is parsed as
    a separate Constraint.

    Natural-language explanations can be attached to a constraint by either
    passing ``default_description`` (applied to every constraint extracted
    from this block) or by embedding an OCL ``--`` comment inline (per
    constraint). An inline comment always wins over ``default_description``.
    """
    if not ocl_text:
        return [], []

    constraints = []
    warnings = []
    constraint_count = 0

    domain_classes = {cls.name.lower(): cls for cls in domain_model.types}

    # Split the OCL text into individual constraint blocks.
    # Each block starts with the keyword "context" (case-insensitive).
    # We use re.split to split on word-boundary "context" while keeping the
    # delimiter so each chunk still contains its own "context ..." prefix.
    blocks = re.split(r'(?i)(?=\bcontext\b)', ocl_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract inline `--` description before collapsing newlines so that
        # comments (which terminate at end-of-line) are attributed correctly.
        cleaned_block, inline_description = _extract_inline_description(block)

        # Collapse newlines within a single constraint block
        line = cleaned_block.replace('\n', ' ').strip()

        if not line.lower().startswith('context'):
            continue

        # Extract context class name
        parts = line.split()
        if len(parts) < 4:  # Minimum: "context ClassName inv name:"
            warnings.append(f"Warning: Malformed OCL constraint (too few tokens): {line}")
            continue

        context_class_name = parts[1]
        context_class = domain_classes.get(context_class_name.lower())

        if not context_class:
            warning_msg = f"Warning: Context class {context_class_name} not found"
            warnings.append(warning_msg)
            continue

        constraint_count += 1
        constraint_name = f"constraint_{context_class_name}_{counter}_{constraint_count}"

        description = inline_description or (default_description or None)

        constraints.append(
            Constraint(
                name=constraint_name,
                context=context_class,
                expression=line,
                language="OCL",
                description=description,
            )
        )

    return constraints, warnings
