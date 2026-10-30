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
``price`` here overrides them on that provider's route.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CATALOG_ENV_VAR = "BESSER_MODEL_CATALOG"
DEFAULT_CATALOG_PATH = Path(__file__).with_name("data") / "model_catalog.json"

_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")


class ModelCatalogError(ValueError):
    """The catalog file is missing or malformed."""


@dataclass(frozen=True)
class ModelCatalog:
    providers: dict[str, dict[str, Any]]
    # host -> lower-cased model id -> rates
    route_prices: dict[str, dict[str, dict[str, float]]]
    # lower-cased model id -> rates, from providers without a base_url
    listed_prices: dict[str, dict[str, float]]

    def base_url(self, provider: str) -> str | None:
        """The endpoint of a gateway provider, or ``None``."""
        entry = self.providers.get(provider)
        return entry["base_url"] if entry else None

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


def _price(raw: Any, where: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ModelCatalogError(f"{where}: price must be an object")
    for field in _PRICE_FIELDS:
        value = raw.get(field)
        if value is None and field.startswith("cache_"):
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ModelCatalogError(f"{where}: price.{field} must be a number >= 0")
    unknown = set(raw) - set(_PRICE_FIELDS)
    if unknown:
        raise ModelCatalogError(f"{where}: unknown price field(s) {sorted(unknown)}")
    rate = {field: float(raw[field]) for field in ("input", "output")}
    for field in ("cache_read", "cache_write"):
        rate[field] = float(raw[field]) if raw.get(field) is not None else rate["input"]
    return rate


def _optional_str(entry: dict, key: str, where: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelCatalogError(f"{where}: {key} must be a string")
    return value


def _provider(provider_id: str, raw: Any) -> dict[str, Any]:
    where = f"providers.{provider_id}"
    if not isinstance(raw, dict):
        raise ModelCatalogError(f"{where} must be an object")
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ModelCatalogError(f"{where}: label is required")
    base_url = _optional_str(raw, "base_url", where)
    if base_url is not None and urlparse(base_url).scheme not in ("http", "https"):
        raise ModelCatalogError(f"{where}: base_url must be an http(s) URL")
    models_raw = raw.get("models")
    if not isinstance(models_raw, list):
        raise ModelCatalogError(f"{where}: models must be a list")
    models, seen = [], set()
    for index, model in enumerate(models_raw):
        model_where = f"{where}.models[{index}]"
        if not isinstance(model, dict):
            raise ModelCatalogError(f"{model_where} must be an object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelCatalogError(f"{model_where}: id is required")
        model_id = model_id.strip()
        if model_id.lower() in seen:
            raise ModelCatalogError(f"{model_where}: duplicate id {model_id!r}")
        seen.add(model_id.lower())
        models.append({
            "id": model_id,
            "label": _optional_str(model, "label", model_where) or model_id,
            "default": model.get("default") is True,
            "price": (_price(model["price"], model_where)
                      if model.get("price") is not None else None),
        })
    return {
        "label": label.strip(),
        "hint": _optional_str(raw, "hint", where),
        "key_prefix": _optional_str(raw, "key_prefix", where),
        "base_url": base_url,
        "models": models,
    }


def _build(data: dict) -> ModelCatalog:
    providers_raw = data.get("providers")
    if not isinstance(providers_raw, dict):
        raise ModelCatalogError("providers must be an object")
    providers = {
        provider_id: _provider(provider_id, raw)
        for provider_id, raw in providers_raw.items()
        if not provider_id.startswith("_")
    }
    route_prices: dict[str, dict[str, dict[str, float]]] = {}
    for host, table in (data.get("route_prices") or {}).items():
        if host.startswith("_"):
            continue
        if not isinstance(table, dict):
            raise ModelCatalogError(f"route_prices.{host} must be an object")
        route_prices[host.lower()] = {
            model_id.lower(): _price(rate, f"route_prices.{host}.{model_id}")
            for model_id, rate in table.items()
            if not model_id.startswith("_")
        }
    listed_prices: dict[str, dict[str, float]] = {}
    for entry in providers.values():
        priced = {m["id"].lower(): m["price"] for m in entry["models"] if m["price"]}
        if entry["base_url"]:
            host = urlparse(entry["base_url"]).hostname.lower()
            route_prices.setdefault(host, {}).update(priced)
        else:
            listed_prices.update(priced)
    return ModelCatalog(providers, route_prices, listed_prices)


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
