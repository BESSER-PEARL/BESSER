"""The model catalog moved to :mod:`besser.llm.catalog`; this keeps the old import path."""

from besser.llm.catalog import ModelCatalogError, load_model_catalog

__all__ = ["ModelCatalogError", "load_model_catalog"]
