"""On-demand pre-analysis of OCL invariants.

:func:`analyze_invariant` computes the structural events that may violate one
invariant and registers a :class:`ViolatingEvent` link for each. It does not
run during parsing or validation, and it does not change the constraint class.
"""

from besser.BUML.notations.ocl.incremental.analyze import (
    AnalysisResult,
    analyze_invariant,
    analyze_invariants,
)

__all__ = ["AnalysisResult", "analyze_invariant", "analyze_invariants"]
