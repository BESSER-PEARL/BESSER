"""Check that the web editor's committed OpenAPI schema matches this backend.

The editor (the ``frontend`` submodule) commits the backend's ``app.openapi()`` as
``packages/webapp/openapi/besser-openapi.json`` and generates its typed API client
from it. This exports the schema the way the editor's ``scripts/export-openapi.py``
does and fails on any difference. Skips, with a notice, when the checked-out
submodule has no such file.

Usage: python .github/scripts/check_openapi.py
"""
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "besser/utilities/web_modeling_editor/frontend/packages/webapp/openapi/besser-openapi.json"

sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

if not COMMITTED.is_file():
    print(f"::notice::{COMMITTED.relative_to(ROOT).as_posix()} not in the checked-out editor; skipping the check")
    sys.exit(0)

from besser.utilities.web_modeling_editor.backend.backend import app  # noqa: E402

current = json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
committed = COMMITTED.read_text(encoding="utf-8")
if current == committed:
    print("The editor's OpenAPI schema is up to date.")
    sys.exit(0)

sys.stdout.writelines(difflib.unified_diff(
    committed.splitlines(keepends=True), current.splitlines(keepends=True),
    "editor: openapi/besser-openapi.json", "backend: app.openapi()",
))
print("::error::The editor's OpenAPI schema is stale. In the editor, run `npm run api:generate` "
      "in packages/webapp and commit the result, then update the submodule pointer.")
sys.exit(1)
