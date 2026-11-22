"""``besser-wme-backend``: argument parsing and the missing-dependency message.

No server is started: ``backend.run`` is replaced by a recorder.
"""
import sys

import pytest

from besser.utilities.web_modeling_editor.backend import cli

BACKEND_MODULE = "besser.utilities.web_modeling_editor.backend.backend"


@pytest.fixture
def run_calls(monkeypatch):
    from besser.utilities.web_modeling_editor.backend import backend

    calls = []
    monkeypatch.setattr(backend, "run", lambda **kwargs: calls.append(kwargs))
    return calls


def test_defaults_match_the_documented_backend_address(run_calls):
    assert cli.main([]) == 0
    assert run_calls == [{"host": "0.0.0.0", "port": 9000, "log_level": "info"}]


def test_host_port_and_log_level_are_passed_through(run_calls):
    assert cli.main(["--host", "127.0.0.1", "--port", "9100", "--log-level", "debug"]) == 0
    assert run_calls == [{"host": "127.0.0.1", "port": 9100, "log_level": "debug"}]


def test_backend_main_uses_the_same_parser(run_calls):
    from besser.utilities.web_modeling_editor.backend import backend

    backend.main(["--port", "9001"])
    assert run_calls == [{"host": "0.0.0.0", "port": 9001, "log_level": "info"}]


@pytest.mark.parametrize("port", ["0", "65536", "http"])
def test_an_invalid_port_is_a_usage_error(port, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--port", port])
    assert exc.value.code == 2
    assert "port" in capsys.readouterr().err


def test_missing_backend_dependency_prints_the_install_command(monkeypatch, capsys):
    import besser.utilities.web_modeling_editor.backend as backend_pkg

    # Force a fresh import of backend.py with FastAPI unavailable.
    monkeypatch.delitem(sys.modules, BACKEND_MODULE, raising=False)
    monkeypatch.delattr(backend_pkg, "backend", raising=False)
    monkeypatch.setitem(sys.modules, "fastapi", None)

    assert cli.main([]) == 1
    err = capsys.readouterr().err
    assert "missing module: 'fastapi'" in err
    assert f"pip install -r {cli.REQUIREMENTS_FILE}" in err


def test_a_missing_besser_module_is_not_disguised_as_an_install_step(monkeypatch):
    import besser.utilities.web_modeling_editor.backend as backend_pkg

    monkeypatch.delitem(sys.modules, BACKEND_MODULE, raising=False)
    monkeypatch.delattr(backend_pkg, "backend", raising=False)
    monkeypatch.setitem(sys.modules, "besser.utilities.web_modeling_editor.backend.middleware", None)

    with pytest.raises(ModuleNotFoundError):
        cli.main([])


def test_the_requirements_file_ships_beside_the_cli():
    # setup.cfg packages it so a PyPI install can point at it.
    assert cli.REQUIREMENTS_FILE.is_file()
