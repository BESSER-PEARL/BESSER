"""Normalization rule modules.

Each module defines a list of :class:`Rule` instances. The engine consumes the
concatenated list in a fixed order (see :func:`build_default_rules`).
"""

from besser.BUML.notations.ocl.normalization.rules.boolean import BOOLEAN_RULES
from besser.BUML.notations.ocl.normalization.rules.cnf import CNF_RULES
from besser.BUML.notations.ocl.normalization.rules.collection import COLLECTION_RULES
from besser.BUML.notations.ocl.normalization.rules.iterator import ITERATOR_RULES
from besser.BUML.notations.ocl.normalization.rules.all_instances import ALL_INSTANCES_RULES
from besser.BUML.notations.ocl.normalization.rules.multiplicity import MULTIPLICITY_RULES


def build_default_rules():
    """Return the full ordered rule list applied at every traversal."""
    return [
        # Constant collapse + simple structural rewrites first so cheaper rules
        # don't keep firing on already-simplified output of bigger rewrites.
        *BOOLEAN_RULES,
        # Sugar elimination — implies, xor, if-bool.
        *CNF_RULES,
        # Collection sugar — isEmpty, reject.
        *COLLECTION_RULES,
        # Multiplicity-aware fast paths fire before iterator rewrites so they
        # can collapse trivially-true chain forms.
        *MULTIPLICITY_RULES,
        # Size-of-select fusions and exists→forAll.
        *ITERATOR_RULES,
        # allInstances over the constraint context class.
        *ALL_INSTANCES_RULES,
    ]
