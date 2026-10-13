"""``besser-wme-backend``: start the Web Modeling Editor backend.

This module is the console-script entry point registered in ``setup.cfg``.
It deliberately imports nothing beyond the standard library at module level:
the backend's own dependencies (FastAPI, openpyxl, ...) are not part of the
core ``besser`` install, so the backend module is imported only after the
arguments are parsed, and a missing dependency is reported as an install
instruction instead of an ``ImportError`` traceback.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_LOG_LEVEL = "info"
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")

PROG = "besser-wme-backend"
REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")
REQUIREMENTS_REPO_PATH = "besser/utilities/web_modeling_editor/backend/requirements.txt"


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 1 and 65535, got {port}")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser shared by the console script and ``backend.py``."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Start the BESSER Web Modeling Editor backend (FastAPI on uvicorn). "
            "The API is served under /besser_api."
        ),
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"interface to bind (default: {DEFAULT_HOST}, every interface; "
             "use 127.0.0.1 to accept local connections only)",
    )
    parser.add_argument(
        "--port", type=_port, default=DEFAULT_PORT,
        help=f"port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--log-level", choices=LOG_LEVELS, default=DEFAULT_LOG_LEVEL,
        help=f"uvicorn log level (default: {DEFAULT_LOG_LEVEL})",
    )
    return parser


def missing_dependency_message(module_name: str) -> str:
    """Explain how to install the backend dependencies when ``module_name`` is missing."""
    if REQUIREMENTS_FILE.is_file():
        requirements = str(REQUIREMENTS_FILE)
    else:
        requirements = REQUIREMENTS_REPO_PATH
    return (
        f"{PROG}: the Web Modeling Editor backend needs packages that the core "
        f"'besser' install does not include (missing module: {module_name!r}).\n"
        "Install them with:\n\n"
        f"    pip install -r {requirements}\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv``, check the backend dependencies, and run the server.

    Returns a process exit status; the server call blocks until shutdown.
    """
    args = build_parser().parse_args(argv)
    try:
        from besser.utilities.web_modeling_editor.backend import backend
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        # A module missing from besser itself is a packaging bug, not an
        # install step the user skipped: keep the traceback for that.
        if not missing or missing == "besser":
            raise
        print(missing_dependency_message(missing), file=sys.stderr)
        return 1
    backend.run(host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
