"""The files that set BESSER's defaults for a coding agent: ``AGENTS.md``, ``CLAUDE.md`` and ``.ignore``.

Every token an agent reads is re-read on each later turn of its session, so the guide is short
and tells it what not to read: generated code, BESSER's docs and sources, the model file.

``besser new`` and ``besser init`` write these files; ``besser build`` keeps the ``.ignore`` block
in step with the outputs of ``besser.toml``; ``besser guide`` prints the guide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

AGENTS_FILENAME = "AGENTS.md"
CLAUDE_FILENAME = "CLAUDE.md"
# ripgrep (and so Claude Code's Grep tool), fd and ag skip what it lists; git does not read it.
IGNORE_FILENAME = ".ignore"
# Claude Code reads CLAUDE.md, not AGENTS.md; this line imports the other file.
CLAUDE_IMPORT = "@AGENTS.md"

# Not the "besser:begin" markers of a Spec-Driven push, which owns that block of AGENTS.md.
_GUIDE_BEGIN, _GUIDE_END = "<!-- besser-guide:begin -->", "<!-- besser-guide:end -->"
_IGNORE_BEGIN, _IGNORE_END = "# besser:begin", "# besser:end"

GUIDE = """\
## BESSER: model-driven code

The folders listed in `besser.toml` (the `dir` of each output) are generated from the B-UML model by the
`besser` command. These rules keep generated code, the model file and BESSER itself out of your context.

Read:
- `besser info model.py`: the whole model in a few hundred tokens (`name:type`, `?` optional, `=default`,
  associations as `role:Class[multiplicity]`, constraints). `besser info model.py --class X`: one class with
  its association ends, routes and the generated names to import. Read these, not `model.py`.
- Never open, grep or edit generated code or `.besser_generated.json` (`.ignore` hides them), BESSER's docs,
  sources or `--help`.

Change the model with `besser model` (one line back; the file is written only if the result is valid):
- `besser model add-class Invoice number:str total:float paid:bool=false [--parent Base]`
- `besser model add-attribute Contact source:str? score:int=0`
- `besser model add-association invoice_contact invoices:Invoice[*] contact:Contact[1]`
- `besser model set-enum Stage LEAD WON LOST` (creates or replaces the literals)
- `besser model rename Customer Account`, `rename Contact.mail email`, `rename Stage.WON CLOSED`
- `besser model remove Contact.phone` (a class goes with its associations)
- `besser model add-method Deal "discount(rate:float) -> float"`, `add-constraint Deal "self.amount >= 0"`
- Several changes in one call: `besser model batch - <<'EOF'` with one change per line.
Do not write Python scripts that patch `model.py`; edit it by hand only where a command says so.

Then one command: `besser apply` (validates, builds every output keeping your edits, runs the `test` command
of `besser.toml`). One line on success; on failure only the errors or the failing tests. Never `--force`
or `overwrite = true`: they discard edits.

Hand-written code:
- Never in a generated file. Backend routes, middleware and startup hooks go in `extensions.py` beside the
  generated `main_api.py`: `before_routers(app)` adds routes that win over the generated ones, `setup(app)`
  adds the rest. Other code lives outside the generated folders and imports the generated modules
  (`sql_alchemy.X`, `pydantic_classes.XCreate`, `database.get_db`). A build never overwrites either.
- Trust the build: check behaviour with tests that `besser apply` runs, not by reading the output.
- A model `.py` file only builds the model. A GUI model goes in the same file as its domain model.

What the `backend` generator gives per class `X` (path segment: the class name in lower case): `POST /x/`,
`GET /x/`, `GET|PUT|DELETE /x/{x_id}/`, plus `/x/bulk/`, `/x/count/`, `/x/paginated/`, `/x/search/`, routes
per association end and `/health`, `/statistics`. Bodies take the attributes and each association end by
name (`"author": 3`, `"books": [1, 2]`); an end with lower bound 1 is required (400 without it). `POST`, `PUT`
and `GET /x/{x_id}/` answer `{"x": {attributes, "id", "<end>_id" per to-one end}, "<...>_ids": [...]}`;
`GET /x/` a list of the inner objects. OCL constraints are enforced (422) only when they compare one
attribute with a number or use `matches('regex')`; write other rules in `extensions.py`. A `Method` becomes
`POST /x/{x_id}/methods/<name>/`. Run it from its folder: `uvicorn main_api:app` (`PORT`, `DATABASE_URL`).
"""


@dataclass(frozen=True)
class GeneratedDir:
    """An output folder of ``besser.toml``, relative to the project folder, and the file in it that is the user's."""

    path: str
    extension_file: str | None = None


def _with_block(text: str, begin: str, end: str, block: str) -> str:
    """``text`` with ``block`` (markers included) in place of the one between ``begin`` and ``end``, or appended."""
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text, count=1)
    body = text.rstrip()
    return f"{body}\n\n{block}\n" if body else f"{block}\n"


def _has_block(path: Path, begin: str) -> bool:
    try:
        return begin in path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _update(path: Path, new_text: str, changed: list[str]) -> None:
    """Write ``new_text`` to ``path`` when it differs, and add the path to ``changed``."""
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    if new_text != old_text:
        path.write_text(new_text, encoding="utf-8", newline="\n")
        changed.append(path.as_posix())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def ignore_block(generated: Sequence[GeneratedDir]) -> str:
    """The ``.ignore`` lines that keep the generated folders out of searches, between their markers."""
    lines = [_IGNORE_BEGIN, "# Generated by BESSER (besser.toml): a black box, see AGENTS.md."]
    for folder in generated:
        root = "/" + folder.path.strip("/")
        lines.append(f"{root}/*")
        if folder.extension_file:
            # A file in an excluded folder cannot be re-included: each folder on its way is, then emptied.
            prefix = root
            for part in folder.extension_file.split("/")[:-1]:
                prefix = f"{prefix}/{part}"
                lines += [f"!{prefix}/", f"{prefix}/*"]
            lines.append(f"!{root}/{folder.extension_file}")
    lines += ["**/.besser_generated.json", _IGNORE_END]
    return "\n".join(lines)


def write_agent_files(folder: Path, generated: Sequence[GeneratedDir]) -> list[str]:
    """Write or update the guide in ``AGENTS.md``, the import of it in ``CLAUDE.md``, and ``.ignore``.

    What people wrote in those files is kept: the guide and the ignore rules are blocks between
    markers, replaced in place. Returns the files that changed.

    Raises:
        OSError, UnicodeDecodeError: A file cannot be read or written.
    """
    changed: list[str] = []
    agents = folder / AGENTS_FILENAME
    _update(agents, _with_block(_read(agents), _GUIDE_BEGIN, _GUIDE_END,
                                f"{_GUIDE_BEGIN}\n{GUIDE}{_GUIDE_END}"), changed)
    claude = folder / CLAUDE_FILENAME
    claude_text = _read(claude)
    if CLAUDE_IMPORT not in claude_text.split():
        _update(claude, f"{claude_text.rstrip()}\n\n{CLAUDE_IMPORT}\n" if claude_text.strip() else f"{CLAUDE_IMPORT}\n",
                changed)
    ignore = folder / IGNORE_FILENAME
    _update(ignore, _with_block(_read(ignore), _IGNORE_BEGIN, _IGNORE_END, ignore_block(generated)), changed)
    return changed


def sync_ignore(folder: Path, generated: Sequence[GeneratedDir]) -> list[str]:
    """Bring the ``.ignore`` block up to the outputs of ``besser.toml``, when the project has one.

    A project whose ``.ignore`` has no BESSER block did not ask for one: it is left alone.
    Returns the files that changed.
    """
    ignore = folder / IGNORE_FILENAME
    if not _has_block(ignore, _IGNORE_BEGIN):
        return []
    changed: list[str] = []
    _update(ignore, _with_block(_read(ignore), _IGNORE_BEGIN, _IGNORE_END, ignore_block(generated)), changed)
    return changed
