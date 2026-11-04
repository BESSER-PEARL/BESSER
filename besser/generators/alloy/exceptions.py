"""Exceptions of the Alloy translation."""


class AlloyTranslationError(ValueError):
    """Raised for domain models, or OCL invariants, that cannot be translated into Alloy."""
