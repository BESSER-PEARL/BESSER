"""
OCL constraint parsing utilities for converting JSON to BUML format.
"""

import re
from besser.BUML.metamodel.structural import DomainModel, Constraint


def process_ocl_constraints(ocl_text: str, domain_model: DomainModel, counter: int) -> tuple[list, list]:
    """Process OCL constraints and convert them to BUML Constraint objects.

    Supports multiple constraints separated by ``context`` keywords in the
    same OCL text block.  Each ``context ... inv ...`` section is parsed as
    a separate Constraint.
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

        # Collapse newlines within a single constraint block
        line = block.replace('\n', ' ').strip()

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

        constraints.append(
            Constraint(
                name=constraint_name,
                context=context_class,
                expression=line,
                language="OCL"
            )
        )

    return constraints, warnings
