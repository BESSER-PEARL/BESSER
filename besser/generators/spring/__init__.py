"""Spring Boot backend generator.

Only :class:`SpringBackendGenerator` is public: the entity, repository, service,
controller and HTTP writers are internal helpers it composes, and they are not
usable on their own.
"""

from .spring_backend_generator import (
    DEFAULT_JAVA_VERSION,
    DEFAULT_SPRING_APP_NAME,
    DEFAULT_SPRING_BOOT_VERSION,
    SpringBackendGenerator,
)

__all__ = [
    "SpringBackendGenerator",
    "DEFAULT_SPRING_BOOT_VERSION",
    "DEFAULT_JAVA_VERSION",
    "DEFAULT_SPRING_APP_NAME",
]
