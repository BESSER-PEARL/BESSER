"""Shared helper: run a client against a freshly generated backend."""
import json
import subprocess
import sys
import textwrap

import pytest

from besser.generators.backend import BackendGenerator

_PREAMBLE = textwrap.dedent("""
    import json
    import logging
    import main_api
    from fastapi.testclient import TestClient

    class _Records(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(f"{record.levelname}:{record.name}:{record.getMessage()}")

    log = _Records()
    logging.getLogger().addHandler(log)
    client = TestClient(main_api.app, raise_server_exceptions=False)
""")


@pytest.fixture
def run_generated_backend(tmp_path):
    """Generate the backend of ``model`` and run ``script`` against it in a fresh process.

    The script has ``client`` (a TestClient of the generated app) and ``log``
    (``log.lines``: every log record as ``LEVEL:logger:message``), and prints
    one JSON document as its last line, which is returned. The generated files
    are in ``tmp_path``.
    """
    def run(model, script: str, **generator_options):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        BackendGenerator(model=model, output_dir=str(tmp_path), **generator_options).generate()
        result = subprocess.run([sys.executable, "-c", _PREAMBLE + textwrap.dedent(script)], cwd=str(tmp_path),
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    return run
