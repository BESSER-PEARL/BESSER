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
