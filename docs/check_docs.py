"""Build the docs and reject build errors or unexpected warnings.

Run from any directory: python docs/check_docs.py [--offline] [--output PATH].
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import re
import sys

from sphinx.application import Sphinx

# An outage at docs.python.org or sphinx-doc.org is not a defect in these docs;
# cross-references to those inventories then render as plain text.
_INVENTORY_UNREACHABLE = "failed to reach any of the inventories"


def unexpected_warnings(text: str) -> list[str]:
    """Warning and error lines that fail the build."""
    return [line for line in text.splitlines()
            if re.search(r"WARNING|ERROR", line) and _INVENTORY_UNREACHABLE not in line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip external Python/Sphinx inventories")
    parser.add_argument("--output", type=Path, help="HTML output directory")
    args = parser.parse_args()
    docs = Path(__file__).resolve().parent
    output = args.output.resolve() if args.output else docs / "build" / "html"
    warnings = StringIO()
    overrides = {"intersphinx_mapping": {}} if args.offline else {}
    try:
        # A full rebuild must not load a search index from a previous theme.
        (output / "searchindex.js").unlink(missing_ok=True)
        app = Sphinx(
            srcdir=str(docs / "source"),
            confdir=str(docs / "source"),
            outdir=str(output),
            doctreedir=str(output.parent / "doctrees"),
            buildername="html",
            confoverrides=overrides,
            status=StringIO(),
            warning=warnings,
            freshenv=True,
        )
        app.build(force_all=True)
    except Exception as exc:
        print(warnings.getvalue(), file=sys.stderr)
        print(f"Documentation build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    text = warnings.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    (output.parent / "warnings.log").write_text(text, encoding="utf-8")
    unexpected = unexpected_warnings(text)
    if _INVENTORY_UNREACHABLE in text:
        print("Note: an external intersphinx inventory was unreachable; its links render as text.")
    if app.statuscode or unexpected:
        print("\n".join(unexpected[:25]) or text, file=sys.stderr)
        if len(unexpected) > 25:
            print(f"... {len(unexpected) - 25} additional warnings in the full log.", file=sys.stderr)
        print(f"Documentation check failed. Full log: {output.parent / 'warnings.log'}", file=sys.stderr)
        return 1
    print(f"Documentation built: {output} ({len(text.splitlines())} warning log lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
