"""
OCL constraint parsing utilities for converting JSON to BUML format.
"""

import re
from besser.BUML.metamodel.structural import DomainModel, Constraint


def process_ocl_constraints(ocl_text: str, domain_model: DomainModel, counter: int) -> tuple[list, list]:
    """Process OCL constraints and convert them to BUML Constraint objects."""
    if not ocl_text:
        return [], []

    constraints = []
    warnings = []
    lines = re.split(r'[,]', ocl_text)
    constraint_count = 1

    domain_classes = {cls.name.lower(): cls for cls in domain_model.types}

    for line in lines:

        line = line.strip().replace('\n', '')
        if not line or not line.lower().startswith('context'):
            continue

        # Extract context class name
        parts = line.split()
        if len(parts) < 4:  # Minimum: "context ClassName inv name:"
            continue

        context_class_name = parts[1]
        context_class = domain_classes.get(context_class_name.lower())

        if not context_class:
            warning_msg = f"Warning: Context class {context_class_name} not found"
            warnings.append(warning_msg)
            continue

        constraint_name = f"constraint_{context_class_name}_{counter}_{constraint_count}"
        constraint_count += 1

        constraints.append(
            Constraint(
                name=constraint_name,
                context=context_class,
                expression=line,
                language="OCL"
            )
        )

    return constraints, warnings
