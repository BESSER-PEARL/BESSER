"""Custom exception hierarchy for BESSER web modeling editor."""


class BesserError(Exception):
    """Base exception for all BESSER errors."""
    pass


class ConversionError(BesserError):
    """Raised when diagram conversion fails."""
    pass


class ValidationError(BesserError):
    """Raised when diagram validation fails."""
    pass


class GenerationError(BesserError):
    """Raised when code generation fails."""
    pass


class ConfigurationError(BesserError):
    """Raised when configuration is invalid."""
    pass


class CodeValidationError(ValidationError):
    """Raised when a CustomCodeAction fails structural or semantic validation.

    Subclasses :class:`ValidationError` so ``@handle_endpoint_errors`` and the
    app-level exception handlers map it to HTTP 400 with its message, like any
    other user-input validation failure.
    """
    pass


class LegacyDiagramFormatError(ConversionError):
    """Raised when a UML diagram arrives in the legacy v3 editor format.

    The converters read only the v4 ``{nodes, edges}`` shape; a v3
    ``{elements, relationships}`` payload would otherwise convert to an empty
    model. Subclasses :class:`ConversionError`, so it maps to HTTP 400.
    """
    pass
