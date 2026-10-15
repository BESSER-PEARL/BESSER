"""Read-only workspace access shared by the scaffold-map extractors.

One :class:`SourceIndex` per map build: it walks the workspace once (links and
special files skipped, dependency and build directories pruned, size and count
capped), caches file text and Python ASTs, and records what it could not read.
Extractors never open files themselves, so every map honours the same limits.
"""

from __future__ import annotations

import ast
import os
from typing import Iterable, Optional

from besser.spec_driven_agent.execution.workspace_fs import read_plain_text, walk_plain
from besser.spec_driven_agent.parsed_source import parse_source

# Never mapped: installed dependencies, build output, caches, run internals.
SKIP_DIRS = frozenset({
    "node_modules", "__pycache__", ".git", "venv", ".venv", "env", "dist", "build",
    ".next", ".gradle", "target", ".pytest_cache", ".mypy_cache", ".agent-trace",
    "coverage", ".turbo", ".cache",
})
MAX_FILES = 2000
MAX_FILE_BYTES = 500_000
# Collection bounds. Code and well-known config files (tier 0) are always
# preferred; data-like files (tier 1: .json, .txt, .md, .yaml ...) are capped
# per directory, so a dataset or a locale folder cannot crowd out the app.
# MAX_FILES of the collected files are mapped: tier 0 first, shallowest first.
_MAX_WALKED = 20_000
_MAX_DATA_FILES_PER_DIR = 50
_CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".html", ".htm", ".css", ".scss",
    ".sass", ".less", ".java", ".kt", ".sql", ".bpmn", ".j2", ".jinja", ".vue", ".svelte", ".prisma",
    ".graphql", ".sh",
})
_CONFIG_NAMES = frozenset({
    "package.json", "tsconfig.json", "jsconfig.json", "requirements.txt", "pyproject.toml",
    "docker-compose.yml", "docker-compose.yaml", "config.yaml", "config.yml", "pom.xml",
    "build.gradle", "settings.gradle",
})
# Text read per build. Past it, further files read as unreadable (with a note)
# rather than holding a whole monorepo in memory.
MAX_TOTAL_BYTES = 12_000_000
# Only files an extractor can read count toward MAX_FILES; images, fonts,
# archives and binaries would otherwise use up the cap and hide the app.
MAPPABLE_SUFFIXES = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".java", ".kt", ".sql", ".ttl", ".rdf", ".owl", ".bpmn",
    ".xml", ".yaml", ".yml", ".toml", ".txt", ".md", ".cfg", ".ini", ".j2", ".jinja", ".vue",
    ".svelte", ".graphql", ".prisma", ".sh",
})
MAPPABLE_NAMES = frozenset({"Dockerfile", "Makefile", "Procfile", "requirements.txt"})


def _tier(name: str) -> int:
    """0 for code and well-known config files, 1 for data-like files."""
    if name in _CONFIG_NAMES or name in MAPPABLE_NAMES or name.startswith(("Dockerfile", "vite.config",
                                                                            "settings", "manage")):
        return 0
    return 0 if os.path.splitext(name)[1].lower() in _CODE_SUFFIXES else 1


def _mappable(name: str) -> bool:
    return name in MAPPABLE_NAMES or name.startswith("Dockerfile") or \
        os.path.splitext(name)[1].lower() in MAPPABLE_SUFFIXES


class SourceIndex:
    """The workspace's mappable files, with cached text, lines and ASTs."""

    def __init__(self, workspace: str, *, max_files: int = MAX_FILES) -> None:
        self.workspace = os.path.realpath(workspace)
        self.files: list[str] = []          # workspace-relative, '/'-separated, sorted
        self.notes: list[str] = []
        self._text: dict[str, Optional[str]] = {}
        self._lines: dict[str, list[str]] = {}
        self._ast: dict[str, Optional[ast.AST]] = {}
        self._bytes_read = 0
        self.syntax_errors: dict[str, str] = {}
        found: list[tuple[int, int, str]] = []      # (tier, depth, rel)
        dropped: dict[str, int] = {}                 # top-level dir -> files left out
        truncated = False
        for root, dirs, names in walk_plain(self.workspace):
            # Hidden directories (.yarn, .cache, .idea ...), dependency and build
            # directories, and vendored Python packages never hold scaffold code.
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                             and d not in ("site-packages", "dist-packages")
                             and not d.endswith((".dist-info", ".egg-info")))
            data_here = 0
            for name in sorted(names):
                if name.startswith(".besser_") or not _mappable(name):
                    continue
                rel = os.path.relpath(os.path.join(root, name), self.workspace).replace(os.sep, "/")
                tier = _tier(name)
                if tier:
                    data_here += 1
                    if data_here > _MAX_DATA_FILES_PER_DIR:
                        top = rel.split("/", 1)[0] if "/" in rel else "."
                        dropped[top] = dropped.get(top, 0) + 1
                        continue
                found.append((tier, rel.count("/"), rel))
            if len(found) >= _MAX_WALKED:
                truncated = True
                break
        found.sort()
        self.files = sorted(rel for _, _, rel in found[:max_files])
        for _, _, rel in found[max_files:]:
            top = rel.split("/", 1)[0] if "/" in rel else "."
            dropped[top] = dropped.get(top, 0) + 1
        self.capped = bool(dropped or truncated)
        if self.capped:
            where = ", ".join(f"{n} under {top}/" if top != "." else f"{n} at the top level"
                              for top, n in sorted(dropped.items(), key=lambda kv: -kv[1])[:4])
            self.notes.append(
                f"{len(self.files)} files mapped; not mapped: {where or 'the rest'}"
                + (" (walk stopped early)" if truncated else "")
            )

    # -- file access -------------------------------------------------------

    def abspath(self, rel: str) -> str:
        return os.path.join(self.workspace, rel)

    def text(self, rel: str) -> Optional[str]:
        """The file's text, or None when unreadable, too large or not UTF-8."""
        if rel not in self._text:
            content: Optional[str] = None
            try:
                size = os.path.getsize(self.abspath(rel))
                if size > MAX_FILE_BYTES:
                    self.notes.append(f"{rel}: larger than {MAX_FILE_BYTES} bytes, not mapped")
                elif self._bytes_read + size > MAX_TOTAL_BYTES:
                    if self._bytes_read <= MAX_TOTAL_BYTES:
                        self.notes.append(f"more than {MAX_TOTAL_BYTES} bytes of source; later files not read")
                        self._bytes_read = MAX_TOTAL_BYTES + 1
                else:
                    self._bytes_read += size
                    content = read_plain_text(self.abspath(rel), root=self.workspace)
                    if content is not None and content.startswith("\ufeff"):
                        content = content[1:]     # a UTF-8 BOM is not code
            except OSError:
                content = None
            self._text[rel] = content
        return self._text[rel]

    def lines(self, rel: str) -> list[str]:
        """The file's lines without line endings (empty when unreadable).

        Split on ``\\n`` only, as ``read_file`` and the AST count lines;
        ``str.splitlines`` would also break on U+2028, form feeds and the like
        and shift every line number below them.
        """
        if rel not in self._lines:
            text = self.text(rel)
            lines: list[str] = []
            if text:
                lines = text.split("\n")
                if lines[-1] == "":
                    lines.pop()
                lines = [line[:-1] if line.endswith("\r") else line for line in lines]
            self._lines[rel] = lines
        return self._lines[rel]

    def python_ast(self, rel: str) -> Optional[ast.AST]:
        """The parsed module, or None; a syntax error is recorded in ``syntax_errors``."""
        if rel not in self._ast:
            tree = None
            text = self.text(rel)
            if text is not None:
                try:
                    tree = parse_source(text, rel)
                except (SyntaxError, ValueError) as exc:
                    line = getattr(exc, "lineno", None)
                    msg = getattr(exc, "msg", str(exc))
                    self.syntax_errors[rel] = f"SyntaxError line {line}: {msg}" if line else f"SyntaxError: {msg}"
            self._ast[rel] = tree
        return self._ast[rel]

    def exists(self, rel: str) -> bool:
        """Whether ``rel`` is a regular file on disk, mapped or not.

        The index may leave files out (caps, unmappable suffixes); check this
        before telling the model a file does not exist.
        """
        rel = rel.replace("\\", "/").lstrip("/")
        if rel in self.files:
            return True
        path = os.path.realpath(self.abspath(rel))
        return path.startswith(self.workspace + os.sep) and os.path.isfile(path) and not os.path.islink(self.abspath(rel))

    def drop_trees(self) -> None:
        """Release the parsed ASTs once the maps are built (texts stay for ownership)."""
        self._ast.clear()

    # -- queries -----------------------------------------------------------

    def under(self, root: str) -> list[str]:
        """Files inside ``root`` ('' = whole workspace)."""
        if not root:
            return list(self.files)
        prefix = root.rstrip("/") + "/"
        return [f for f in self.files if f.startswith(prefix)]

    def with_suffix(self, suffixes: Iterable[str], root: str = "") -> list[str]:
        suffixes = tuple(suffixes)
        return [f for f in self.under(root) if f.endswith(suffixes)]

    def unique_quote(self, rel: str, line_no: int) -> str:
        """Line ``line_no`` verbatim, or '' when it is blank or not unique.

        Unique means the line's text, indentation included, occurs exactly
        once anywhere in the file: ``modify_file`` matches ``old_text`` as a
        substring, so a line that also appears inside a longer line (``{`` in
        ``"compilerOptions": {``) would be an ambiguous anchor.
        """
        lines = self.lines(rel)
        if not 1 <= line_no <= len(lines):
            return ""
        line = lines[line_no - 1]
        if not line.strip():
            return ""
        return line if (self.text(rel) or "").count(line) == 1 else ""

    def first_unique_quote(self, rel: str, start: int, end: int) -> str:
        """The first unique non-blank line in ``start..end`` (inclusive), or ''."""
        for n in range(max(1, start), min(end, len(self.lines(rel))) + 1):
            quote = self.unique_quote(rel, n)
            if quote:
                return quote
        return ""


def node_start(node: ast.AST) -> int:
    """First line of a def/class including its decorators."""
    decorators = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decorators])
