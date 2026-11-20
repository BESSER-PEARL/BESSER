"""Proxy variables must reach the agent's subprocesses.

On the LIST VM (production since 2026-11-17) the internet is reachable only
through ``HTTPS_PROXY=http://prox.list.lu:8888``. ``_safe_subprocess_env``
dropped the proxy variables, so every ``npm install`` / ``pip install`` the
agent ran hung until its 300-600 s timeout. Runs 2f4f526e and 6acc8e59 (a
one-class app) spent ~2360 of their 2400 s cap waiting on installs and ended
"time limit". In the worker, ``curl`` with the sandbox env timed out (exit 28);
with the container env it answered 200.
"""
import os

import pytest

from besser.spec_driven_agent.execution.process import _safe_subprocess_env

LIST_PROXY = "http://prox.list.lu:8888"
NO_PROXY = "localhost,127.0.0.1,::1,.list.lu,besser-wme-backend"


@pytest.mark.parametrize("name", ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                                  "ALL_PROXY", "all_proxy"])
def test_proxy_url_reaches_the_subprocess(monkeypatch, name):
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin", name: LIST_PROXY})

    assert _safe_subprocess_env().get(name) == LIST_PROXY


@pytest.mark.parametrize("name", ["NO_PROXY", "no_proxy"])
def test_no_proxy_reaches_the_subprocess(monkeypatch, name):
    """Without it, local probes (127.0.0.1) would be sent to the proxy."""
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin", name: NO_PROXY})

    assert _safe_subprocess_env().get(name) == NO_PROXY


@pytest.mark.parametrize("value", ["http://svc:s3cret@prox.list.lu:8888",
                                   "http://svc@prox.list.lu:8888"])
def test_a_proxy_url_with_credentials_is_not_exposed(monkeypatch, value):
    """The agent can `printenv`; a proxy password must not reach it."""
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin", "HTTPS_PROXY": value})

    assert "HTTPS_PROXY" not in _safe_subprocess_env()
