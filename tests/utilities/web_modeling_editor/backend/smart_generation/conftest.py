"""Shared SmartGen test configuration."""

import pytest


@pytest.fixture(autouse=True)
def configure_local_smartgen_auth(monkeypatch):
    """Tests opt into the local principal unless they exercise auth."""
    monkeypatch.setenv("BESSER_SMART_GEN_AUTH_REQUIRED", "false")
