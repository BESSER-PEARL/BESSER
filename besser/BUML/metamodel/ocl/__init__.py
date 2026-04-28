from .ocl import *

from besser.BUML.metamodel.ocl.walk import walk
from besser.BUML.metamodel.ocl.clone import clone
from besser.BUML.metamodel.ocl.predicates import (
    is_op, is_and, is_or, is_xor, is_implies, is_not,
    is_size, is_isempty, is_allinstances,
    is_comparison, is_atomic_type_test,
    is_loop, is_loop_with_n_iterators,
    is_self, is_bool_const,
)
from besser.BUML.metamodel.ocl.chain import (
    is_chain_from_self, walk_chain_from_self,
    chain_min_multiplicity, chain_max_multiplicity,
)
from besser.BUML.metamodel.ocl.substitute import substitute, ScopeStack
