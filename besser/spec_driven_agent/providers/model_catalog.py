"""The LLM model catalog: which providers and models a deployment offers.

The key dialog's provider and model lists, the endpoint of a gateway provider
(e.g. ``pia``) and the rates a gateway bills are configuration, not code. BESSER
ships a public default (``data/model_catalog.json``); a deployment points
``BESSER_MODEL_CATALOG`` at its own JSON file, which is merged over it:

* ``providers.<id>`` replaces that provider's entry; ``null`` removes it;
  a new id is appended.
* ``route_prices.<host>`` replaces that host's rate table; ``null`` removes it.
* ``"replace": true`` drops the default catalog entirely.

Keys starting with ``_`` are notes and are ignored.

A provider entry::

    {"label": "PIA (LIST)", "hint": "...", "key_prefix": "sk-",
     "base_url": "https://gateway.example/v1",
     "models": [{"id": "m", "label": "M", "default": true,
                 "price": {"input": 1.0, "output": 5.0,
                           "cache_read": 1.0, "cache_write": 1.0}}]}

``base_url`` makes the provider a gateway: the server sends its calls there
over the OpenAI-compatible protocol, so the browser never holds the host.
Prices are USD per 1M tokens; a missing cache rate bills at the input rate.
Public vendor list prices stay in ``data/model_prices.json``; a model's
``price`` here overrides them on that provider's route. A model marked
``"default": true`` is the provider's default when a request names none. An
optional ``"request"`` object sets the model's request quirks
(``forced_tool_choice``, ``thinking_can_be_disabled``,
``reasoning_effort_for_tools``) over model_settings' built-in row, so a new
model with a known quirk needs no release.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

logger = logging.getLogger(__name__)

CATALOG_ENV_VAR = "BESSER_MODEL_CATALOG"
DEFAULT_CATALOG_PATH = Path(__file__).with_name("data") / "model_catalog.json"



class ModelCatalogError(ValueError):
    """The catalog file is missing or malformed."""


@dataclass(frozen=True)
class ModelCatalog:
    providers: dict[str, dict[str, Any]]
    # host -> lower-cased model id -> rates
    route_prices: dict[str, dict[str, dict[str, float]]]
    # lower-cased model id -> rates, from providers without a base_url
    listed_prices: dict[str, dict[str, float]]
    # lower-cased model id -> request quirks (see _Request)
    request_overrides: dict[str, dict[str, Any]]

    def base_url(self, provider: str) -> str | None:
        """The endpoint of a gateway provider, or ``None``."""
        entry = self.providers.get(provider)
        return entry["base_url"] if entry else None

    def flagged_default(self, provider: str) -> str | None:
        """The model the catalog marks ``"default": true`` for ``provider``."""
        entry = self.providers.get(provider)
        if not entry:
            return None
        return next((m["id"] for m in entry["models"] if m["default"]), None)

    def default_model(self, provider: str) -> str | None:
        entry = self.providers.get(provider)
        if not entry or not entry["models"]:
            return None
        flagged = [m["id"] for m in entry["models"] if m["default"]]
        return flagged[0] if flagged else entry["models"][0]["id"]

    def route_price(self, model_id: str, route: str | None) -> dict[str, float] | None:
        """The rate a gateway bills for ``model_id`` on ``route`` (URL or host)."""
        if not route:
            return None
        host = (urlparse(route).hostname or route).lower()
        return self.route_prices.get(host, {}).get(model_id.lower())

    def request_override(self, model_id: str) -> dict[str, Any]:
        """Request quirks the catalog sets for ``model_id`` (possibly empty)."""
        return self.request_overrides.get(model_id.lower(), {})

    def listed_price(self, model_id: str) -> dict[str, float] | None:
        """A rate the catalog sets for a vendor model (no gateway)."""
        return self.listed_prices.get(model_id.lower())

    def public_providers(self) -> list[dict[str, Any]]:
        """The providers as the key dialog needs them: no endpoint, no rates."""
        return [
            {
                "id": provider_id,
                "label": entry["label"],
                "hint": entry["hint"],
                "key_prefix": entry["key_prefix"],
                "models": [
                    {"id": m["id"], "label": m["label"], "default": m["default"]}
                    for m in entry["models"]
                ],
            }
            for provider_id, entry in self.providers.items()
        ]


def _read(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ModelCatalogError(f"Model catalog {path} cannot be read: {exc}") from exc
    except ValueError as exc:
        raise ModelCatalogError(f"Model catalog {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelCatalogError(f"Model catalog {path} must be a JSON object")
    return data


def _merge(base: dict, override: dict) -> dict:
    if override.get("replace") is True:
        base = {}
    merged = {}
    for section in ("providers", "route_prices"):
        table = dict(base.get(section) or {})
        for key, value in (override.get(section) or {}).items():
            if value is None:
                table.pop(key, None)
            else:
                table[key] = value
        merged[section] = table
    return merged


_Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Entry(BaseModel):
    # A typo such as ``base_ur1`` would otherwise load as a provider with no
    # gateway, and ``"default": "true"`` as a model that is not the default.
    model_config = ConfigDict(extra="forbid", strict=True)


class _Price(_Entry):
    """USD per 1M tokens; a missing cache rate bills at the input rate."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)

    def rate(self) -> dict[str, float]:
        return {"input": self.input, "output": self.output,
                "cache_read": self.input if self.cache_read is None else self.cache_read,
                "cache_write": self.input if self.cache_write is None else self.cache_write}


class _Request(_Entry):
    """A model's request quirks, overriding model_settings' row for its id.
    Only the fields a catalog sets are applied (``exclude_unset``)."""

    forced_tool_choice: bool = False
    thinking_can_be_disabled: bool = False
    reasoning_effort_for_tools: str | None = None


class _Model(_Entry):
    id: _Name
    label: str | None = None
    default: bool = False
    price: _Price | None = None
    request: _Request | None = None


class _Provider(_Entry):
    label: _Name
    hint: str | None = None
    key_prefix: str | None = None
    base_url: str | None = None
    models: list[_Model]

    @field_validator("base_url")
    @classmethod
    def _http_url_with_host(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("base_url must be an http(s) URL with a host")
        return value

    @field_validator("models")
    @classmethod
    def _unique_ids(cls, models: list[_Model]) -> list[_Model]:
        seen: set[str] = set()
        for model in models:
            if model.id.lower() in seen:
                raise ValueError(f"duplicate id {model.id!r}")
            seen.add(model.id.lower())
        return models


def _without_notes(value: Any) -> Any:
    """``value`` with every ``_``-prefixed key (a note) removed, at any depth."""
    if isinstance(value, dict):
        return {k: _without_notes(v) for k, v in value.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(value, list):
        return [_without_notes(v) for v in value]
    return value


def _validated(schema: type[BaseModel], raw: Any, where: str) -> Any:
    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        error = exc.errors()[0]
        loc = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error["loc"])
        raise ModelCatalogError(f"{where}{loc}: {error['msg']}") from None


def _provider(provider_id: str, raw: Any) -> dict[str, Any]:
    entry = _validated(_Provider, raw, f"providers.{provider_id}")
    return {
        "label": entry.label,
        "hint": entry.hint,
        "key_prefix": entry.key_prefix,
        "base_url": entry.base_url,
        "models": [{
            "id": model.id,
            "label": model.label or model.id,
            "default": model.default,
            "price": model.price.rate() if model.price else None,
            "request": model.request.model_dump(exclude_unset=True) if model.request else {},
        } for model in entry.models],
    }


def _build(data: dict) -> ModelCatalog:
    data = _without_notes(data)
    providers_raw = data.get("providers")
    if not isinstance(providers_raw, dict):
        raise ModelCatalogError("providers must be an object")
    providers = {
        provider_id: _provider(provider_id, raw)
        for provider_id, raw in providers_raw.items()
    }
    route_prices: dict[str, dict[str, dict[str, float]]] = {}
    for host, table in (data.get("route_prices") or {}).items():
        if not isinstance(table, dict):
            raise ModelCatalogError(f"route_prices.{host} must be an object")
        route_prices[host.lower()] = {
            model_id.lower(): _validated(_Price, rate, f"route_prices.{host}.{model_id}").rate()
            for model_id, rate in table.items()
        }
    listed_prices: dict[str, dict[str, float]] = {}
    request_overrides: dict[str, dict[str, Any]] = {}
    for entry in providers.values():
        for model in entry["models"]:
            if model["request"]:
                request_overrides.setdefault(model["id"].lower(), {}).update(model["request"])
        priced = {m["id"].lower(): m["price"] for m in entry["models"] if m["price"]}
        if entry["base_url"]:
            host = urlparse(entry["base_url"]).hostname.lower()
            route_prices.setdefault(host, {}).update(priced)
        else:
            listed_prices.update(priced)
    return ModelCatalog(providers, route_prices, listed_prices, request_overrides)


@lru_cache(maxsize=1)
def load_model_catalog() -> ModelCatalog:
    """The default catalog with the deployment's file merged over it.

    Cached for the process: editing the file takes a restart. Raises
    :class:`ModelCatalogError` for a missing or malformed file, so a broken
    deployment catalog fails at startup instead of silently dropping a
    provider.
    """
    data = _read(DEFAULT_CATALOG_PATH)
    override_path = os.environ.get(CATALOG_ENV_VAR, "").strip()
    if override_path:
        data = _merge(data, _read(Path(override_path)))
        logger.info("Model catalog: %s merged over the default", override_path)
    try:
        return _build(data)
    except ModelCatalogError as exc:
        source = override_path or DEFAULT_CATALOG_PATH
        raise ModelCatalogError(f"Model catalog {source}: {exc}") from exc
