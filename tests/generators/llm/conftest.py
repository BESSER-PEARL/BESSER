"""Shared fixtures for the LLM harness tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_model_catalog(monkeypatch):
    """Never resolve a context window from the network in tests.

    ``effective_threshold`` consults the keyless tier's model catalog once per
    process. On a host with ``BESSER_FREE_LLM_BASE_URL`` exported that would be
    a real request, and its answer would change every threshold assertion in
    this directory. Pin the cache as already loaded and empty; a test that
    wants a catalog resets ``_CATALOG_LOADED`` and injects ``_fetch_catalog``.
    """
    import besser.generators.llm.compaction as compaction
    monkeypatch.setattr(compaction, "_CATALOG", {}, raising=False)
    monkeypatch.setattr(compaction, "_CATALOG_LOADED", True, raising=False)
