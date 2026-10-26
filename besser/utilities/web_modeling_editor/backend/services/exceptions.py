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


class UpstreamError(BesserError):
    """Raised when an external service the backend calls on the user's behalf fails.

    ``status_code`` is the HTTP status the endpoint answers with: 404 (the resource does not
    exist there), 413 (its answer is too large), 502 (it failed), 503 (this server is busy) or
    504 (it took too long).
    """

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


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
