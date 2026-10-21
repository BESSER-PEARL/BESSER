"""On-demand pre-analysis of OCL invariants.

:func:`analyze_invariant` computes the structural events that may violate one
invariant and stores them on the constraint. It does not run during parsing
or validation.
"""

from besser.BUML.notations.ocl.incremental.analyze import (
    AnalysisResult,
    analyze_invariant,
    analyze_invariants,
)

__all__ = ["AnalysisResult", "analyze_invariant", "analyze_invariants"]
