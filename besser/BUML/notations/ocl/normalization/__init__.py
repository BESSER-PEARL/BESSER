"""B-OCL normalization: rewrite a constraint into canonical form.

Public entry point: :func:`normalize`. The rule-set and driver internals are
documented in :mod:`besser.BUML.notations.ocl.normalization.normalize` and the
``rules`` submodule.
"""

from besser.BUML.notations.ocl.normalization.normalize import normalize, Context
from besser.BUML.notations.ocl.normalization.simplify import simplify_constraint

__all__ = ["normalize", "Context", "simplify_constraint"]
