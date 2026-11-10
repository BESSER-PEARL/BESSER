"""Environment configuration of the deterministic generators."""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, SettingsConfigDict

# django-admin startproject/startapp and the SQL DDL dump take a few seconds;
# two minutes leaves room for a slow host while still freeing a stuck worker.
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 120.0


def _positive_float(value) -> float:
    """A blank, unparsable or non-positive value gives the default."""
    try:
        parsed = float(str(value).strip())
    except ValueError:
        raise PydanticUseDefault() from None
    if not parsed > 0:
        raise PydanticUseDefault()
    return parsed


class GeneratorSettings(BaseSettings):
    """The generators' environment variables (all optional). Read per call."""

    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore", validate_default=False)

    subprocess_timeout_seconds: Annotated[float, BeforeValidator(_positive_float)] = Field(
        DEFAULT_SUBPROCESS_TIMEOUT_SECONDS, validation_alias="BESSER_GENERATOR_SUBPROCESS_TIMEOUT",
        description="Seconds a generator's child process (django-admin, the SQL DDL script) may run.")


def subprocess_timeout() -> float:
    """The timeout, in seconds, for a generator's child process."""
    return GeneratorSettings().subprocess_timeout_seconds
