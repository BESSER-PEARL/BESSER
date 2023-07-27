from metamodel.structural.structural import DomainModel, Class, Constraint


def get_constraints_for_class(model:DomainModel, cl: Class) -> set[Constraint]:
    return {constraint for constraint in model.constraints if constraint.context == cl}

