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


class ConfigurationError(BesserError):
    """Raised when configuration is invalid."""
    pass
