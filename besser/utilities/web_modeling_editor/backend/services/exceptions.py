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


class GovernanceDslValidationError(ValidationError):
    """Raised when a gateway's Governance DSL is syntactically or semantically invalid."""
    pass


class GenerationError(BesserError):
    """Raised when code generation fails."""
    pass


class UpstreamServiceError(BesserError):
    """Raised when a third-party service the request depends on fails.

    Mapped to HTTP 502 by ``@handle_endpoint_errors``. The message is shown to
    the client verbatim, so it must be written for an end user and must never
    contain the upstream response body.
    """
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
