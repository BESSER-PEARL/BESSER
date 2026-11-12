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

Each output `dir` of `besser.toml` is generated from the B-UML model by `besser`.

Read:
- `besser info model.py`: the whole model (`name:type`, `?` optional, `=default`,
  associations as `role:Class[multiplicity]`, constraints); `--class X`: one class. Not `model.py`.
- Never open, grep or edit generated code or `.besser_generated.json` (`.ignore` hides them), BESSER's docs,
  sources or `--help`.

Change the model with `besser model` (one line back; it writes only a valid model).
Types: `str int float bool date datetime time`, or an enumeration's name.
- `besser model add-class Invoice number:str total:float paid:bool=false [--parent Document]`,
  `add-parent Invoice Document`
- `besser model add-attribute Contact source:str? score:int=0 stage:Stage=LEAD`
- `besser model add-association invoice_contact invoices:Invoice[*] contact:Contact[1]`
- `besser model set-enum Stage LEAD WON LOST` (sets all its literals)
- `besser model rename Customer Account`, `rename Contact.mail email`, `rename Stage.WON CLOSED`, an
  association end as `Book.authors` or `book_authors.authors`
- `besser model remove Contact.phone` (a class goes with its associations)
- `besser model add-method Deal "discount(rate:float) -> float"`, `add-constraint Deal "self.amount >= 0"`
- Several changes in one call: `besser model batch - <<'EOF'` with one change per line. To change a type,
  `?`, default or multiplicity: `remove` it, then add it again, in one batch.
Never patch `model.py` with a script; edit it by hand only where a command says so.

Then one command, `besser apply`: it validates, builds every output keeping your edits, runs `test` of
`besser.toml`; one line on success, else only the errors or failing tests. Never `--force` or
`overwrite = true`: they discard edits.

Hand-written code never goes in a generated file; a build never overwrites it:
- Backend routes, middleware and startup hooks: `extensions.py` beside `main_api.py`, with
  `setup(app)` (added after the generated routes) or `before_routers(app)` (routes that win over them). It
  imports `from database import get_db`, `from sql_alchemy import X` (`db.get(X, id)`; an association end
  is a relationship of its name, a to-one end also a column `<end>_id`), `from pydantic_classes import XCreate`.
- Other code lives outside the generated folders. A model `.py` file only builds the model; a GUI model
  goes in the same file as its domain model.
- Check behaviour with tests, not by reading the output: set `test = "python -m pytest -q"` in `besser.toml`
  and give the tests a `client` fixture in `tests/conftest.py` (`backend`: the output's `dir`):
  ```python
  import os, sys, tempfile, pytest
  os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"
  sys.path.insert(0, f"{os.path.dirname(__file__)}/../backend")
  from fastapi.testclient import TestClient
  from main_api import app
  client = pytest.fixture(lambda: TestClient(app))
  ```

The `backend` API per class `X` (`x`: its name in lower case): `POST /x/`, `GET /x/`,
`GET|PUT|DELETE /x/{x_id}/`, a `Method` as `POST /x/{x_id}/methods/<name>/`, and more (`besser info`).
Bodies take the attributes (a date `"2024-01-31"`, an enumeration a literal's name) and each end by name
(`"author": 3`, `"books": [1, 2]`); an end with lower bound 1 is required (422 when missing, 400 when
empty); an unknown id is 400 (404 in a list). An object is `{attributes, "id", "<end>_id" per to-one end}`;
`GET /x/` answers a list of them. `GET /x/{x_id}/` answers `{"x": object}` plus the ids per to-many end:
`"<end>_ids"` for one-to-many, `"<other class>_ids"` (lower case) for many-to-many. `POST` and `PUT` answer
the same if `X` has a to-many end, else the bare object: read it as `body.get("x", body)`. OCL constraints
are enforced (422) only if they compare an attribute with a number or use `matches('regex')`; write other
rules in `extensions.py`. Run `python main_api.py` in its folder (`PORT`, `DATABASE_URL`).
"""

# The GUI lines go after the model commands, before the line that starts with this.
_GUI_GUIDE_BEFORE = "Then one command, `besser apply`"
# The guide's lines on the GUI model, only in a project with an output that reads one (``web_app``):
# every other project would pay for them in each session's context.
GUI_GUIDE = """\
Pages come from the GUI model in `model.py` (`web_app`: React in `<dir>/frontend/`, no hand-written file):
- `besser model add-screen Invoice [--title Bills]`: a page with a table of the class (add, edit, delete),
  opened by a button on the main page; `add-class ... --screen` adds a class with its page.
- Tables follow their class: attributes added, renamed or removed are columns too; a renamed class renames
  the page titled after it, a removed one takes it along. `besser info` lists each page's table.

"""


def guide_text(gui: bool = False) -> str:
    """The guide, with the lines on the GUI model when ``gui`` (an output of the project reads one)."""
    if not gui:
        return GUIDE
    before, after = GUIDE.split(_GUI_GUIDE_BEFORE, 1)
    return f"{before}{GUI_GUIDE}{_GUI_GUIDE_BEFORE}{after}"


@dataclass(frozen=True)
class GeneratedDir:
    """An output folder of ``besser.toml``, relative to the project folder, and the file in it that is the user's."""

    path: str
    extension_file: str | None = None
    reads_gui: bool = False


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
    guide = guide_text(any(output.reads_gui for output in generated))
    _update(agents, _with_block(_read(agents), _GUIDE_BEGIN, _GUIDE_END, f"{_GUIDE_BEGIN}\n{guide}{_GUIDE_END}"),
            changed)
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
