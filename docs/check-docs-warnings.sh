#!/usr/bin/env bash
# Compatibility entry point; the same portable checker runs on Windows and CI.
set -euo pipefail
python "$(dirname "$0")/check_docs.py" "$@"
