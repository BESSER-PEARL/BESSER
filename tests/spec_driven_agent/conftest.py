"""Shared fixtures for the LLM harness tests."""

import os
import warnings

import pytest


def pytest_configure(config):
    """Keep the shell tests runnable where no namespace sandbox can start.

    ``run_command`` fails closed when the sandbox is mandatory and bubblewrap
    cannot run — no package, or a kernel/container that forbids unprivileged
    user namespaces. That is the right production behaviour and the wrong one
    for a developer box, where it would turn every pre-existing shell test red
    for a reason that has nothing to do with the test. ``test_shell_sandbox``
    sets the policy explicitly and is unaffected either way.
    """
    from besser.spec_driven_agent.execution import sandbox

    if os.environ.get(sandbox.SANDBOX_POLICY_ENV):
        return
    if not sandbox.sandbox_supported_platform():
        return
    problem = sandbox.sandbox_selftest_error()
    if problem:
        os.environ[sandbox.SANDBOX_POLICY_ENV] = "off"
        warnings.warn(
            "No shell sandbox on this host "
            f"({problem}); running the shell tests unconfined.",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _no_model_catalog(monkeypatch):
    """Never resolve a context window from the network in tests.

    ``effective_threshold`` consults the keyless tier's model catalog once per
    process. On a host with ``BESSER_FREE_LLM_BASE_URL`` exported that would be
    a real request, and its answer would change every threshold assertion in
    this directory. Pin the cache as already loaded and empty; a test that
    wants a catalog resets ``_CATALOG_LOADED`` and injects ``_fetch_catalog``.
    """
    import besser.spec_driven_agent.agent.compaction as compaction
    monkeypatch.setattr(compaction, "_CATALOG", {}, raising=False)
    monkeypatch.setattr(compaction, "_CATALOG_LOADED", True, raising=False)
