from __future__ import annotations

from typing import Dict, List

from besser.BUML.metamodel.structural import DomainModel


def _sorted_association_ends(association) -> list:
    """Return association ends in a deterministic order."""
    return sorted(association.ends, key=lambda end: (end.type.name, end.name or ""))


def get_foreign_keys(model: DomainModel) -> Dict[str, List[str]]:
    """
    Return a mapping of association name -> [class_name_with_fk, fk_property_name].
    """
    fkeys: Dict[str, List[str]] = {}

    for association in model.associations:
        ends = _sorted_association_ends(association)
        if len(ends) != 2:
            continue

        end0, end1 = ends[0], ends[1]

        # One-to-one
        if end0.multiplicity.max == 1 and end1.multiplicity.max == 1:
            if end0.multiplicity.min > 0 and end1.multiplicity.min == 0:
                fkeys[association.name] = [end1.type.name, end0.name]
            elif end1.multiplicity.min > 0 and end0.multiplicity.min == 0:
                fkeys[association.name] = [end0.type.name, end1.name]
            else:
                fkeys[association.name] = [end0.type.name, end1.name]

        # Many-to-one
        elif end0.multiplicity.max > 1 and end1.multiplicity.max <= 1:
            fkeys[association.name] = [end0.type.name, end1.name]

        elif end0.multiplicity.max <= 1 and end1.multiplicity.max > 1:
            fkeys[association.name] = [end1.type.name, end0.name]

    return fkeys
