"""Errors raised by the durable SmartGen state foundation."""


class DurableStateError(RuntimeError):
    """Base class for durable-state failures."""


class DurableStateConfigurationError(DurableStateError):
    """The selected durable-state backend is not safely configured."""


class MissingOptionalDependencyError(DurableStateConfigurationError):
    """A production adapter dependency is not installed."""


class RecordAlreadyExistsError(DurableStateError):
    """A record with the same durable identity already exists."""


class RecordNotFoundError(DurableStateError):
    """A requested durable record does not exist."""


class OptimisticLockError(DurableStateError):
    """A run update used a stale record version."""


class LeaseLostError(DurableStateError):
    """A worker attempted a write without its current unexpired lease."""


class InvalidRunTransitionError(DurableStateError):
    """A run status transition violates the lifecycle contract."""


class IdempotencyConflictError(DurableStateError):
    """An idempotency key was reused for a different request."""


class StorageIntegrityError(DurableStateError):
    """Stored bytes or queue data failed an integrity check."""
