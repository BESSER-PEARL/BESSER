"""Static resolution of a generated frontend: every import must actually exist.

A backend that boots and answers HTTP says nothing about whether its frontend
renders. Three failure shapes leave a blank page:

    - App.jsx imports ./pages/OrderView.jsx, which was never written
    - api.js imports axios, which package.json does not declare
    - App.jsx uses JSX without importing React, in a project with no
      vite.config and no @vitejs/plugin-react, so the classic runtime is
      used and the page dies on mount

All three are decidable from the files on disk with no node_modules, no
install and no shell. ``frontend_build.py`` would also catch them, but it is
gated on ``enable_toolchain_validation and allow_shell``, which a hosted
deployment typically turns off.

The checks deliberately stay narrow - an unresolvable path, an undeclared
package, a JSX file with no React in scope and no automatic runtime
configured. Anything needing a bundler's resolution rules (aliases,
conditional exports, monorepo links) is skipped rather than guessed.

The same pass flags a dependency whose version is "latest", "*", "x" or
empty: the install then depends on what npm publishes that day.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import NamedTuple

from pygments.lexer import Lexer
from pygments.lexers import JavascriptLexer, get_lexer_for_filename
from pygments.token import Comment, String, _TokenType
from pygments.util import ClassNotFound

_EXCLUDED_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__",
                  "dist", "build", ".next", "out", "coverage", "target"}
_SOURCE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
# ".d.ts": `import type { Book } from './types'` is erased before anything loads it.
_RESOLVE_EXT = ("", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".d.ts",
                ".json", ".css", ".scss", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")
# A JS specifier that a TS project writes for its TS source (NodeNext style).
_TS_SOURCE_FOR = {".js": (".ts", ".tsx"), ".jsx": (".ts", ".tsx"), ".mjs": (".mts",), ".cjs": (".cts",)}

# from/import/require/dynamic-import specifiers; one followed by "+" is only
# the prefix of a path built at runtime (``import('./pages/' + name)``).
# Fallback only: it also matches inside strings and JSX text (see _token_specs).
_SPEC_RE = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|\bimport\s+)"""
    r"""(['"])([^'"\n]+)\1(?!\s*\+)""")
# Tokens allowed between a statement-start import/export and its ``from``,
# besides quoted names (``import { "a-b" as ab }``).
_CLAUSE_PUNCTUATION = {"{", "}", ",", "*"}
_IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*\Z")
# A quoted JS string cannot hold a raw line break; one that does means the lexer
# opened a string at an apostrophe in JSX text and read code as string content.
_RAW_LINE_BREAK = re.compile(r"(?<!\\)\n")
# Import statements inside template-literal text: the lexer opened a template at
# a lone backtick in JSX text and read the code after it as template content.
_SWALLOWED_IMPORT = re.compile(
    r"""\n\s*(?:import|export)\b[^\n]*\bfrom\s*['"]|\n\s*import\s*['"]|\b(?:import|require)\s*\(\s*['"]""")

_NODE_BUILTINS = {
    "assert", "buffer", "child_process", "cluster", "console", "constants",
    "crypto", "dgram", "dns", "domain", "events", "fs", "http", "http2",
    "https", "module", "net", "os", "path", "perf_hooks", "process",
    "punycode", "querystring", "readline", "repl", "stream", "string_decoder",
    "timers", "tls", "tty", "url", "util", "v8", "vm", "worker_threads", "zlib",
}

# Packages a bundler or framework provides without a package.json entry.
_IMPLICIT_PACKAGES = {"react/jsx-runtime", "react/jsx-dev-runtime", "vite/client"}

# The JSX-without-React finding names this many files; the fix is one config line.
_JSX_FILES_LISTED = 8

@lru_cache(maxsize=16)
def _lexer_for(suffix: str) -> Lexer:
    try:
        return get_lexer_for_filename(f"x{suffix}", stripnl=False, ensurenl=False)
    except ClassNotFound:  # .babelrc and other config files: JavaScript syntax
        return JavascriptLexer(stripnl=False, ensurenl=False)


def _tokens(text: str, filename: str) -> list[tuple[_TokenType, str]]:
    return list(_lexer_for(os.path.splitext(filename)[1].lower()).get_tokens(text))


def _without_comments(tokens: list[tuple[_TokenType, str]]) -> str:
    """The lexed text without its comments, keeping every line break.

    Lexed rather than matched: a ``//`` or ``/*`` inside a string, template
    or regex literal (``"/api/*"``, `` `${base}//v1` ``) is not a comment.
    """
    out: list[str] = []
    for token, value in tokens:
        # JSX text such as "See https://x.io" lexes as a comment from "//".
        is_url = token in Comment.Single and out and out[-1].endswith(":")
        out.append("\n" * value.count("\n") if token in Comment and not is_url else value)
    return "".join(out)


def _strip_comments(text: str, filename: str) -> str:
    """``text`` without its comments, keeping every line break."""
    return _without_comments(_tokens(text, filename))


def _token_specs(tokens: list[tuple[_TokenType, str]]) -> tuple[str, ...] | None:
    """Import specifiers: whole quoted strings in import position only.

    ``import '<s>'`` and ``import|export <clause> from '<s>'`` at a statement
    start, ``import('<s>')`` and ``require('<s>')``. A ``from`` or ``import``
    inside a string, template or JSX text is not one. None when the lexer lost
    track of the strings, so the caller falls back to ``_SPEC_RE``.
    """
    values: list[str] = []
    quoted: list[bool] = []
    starts_line: list[bool] = []
    line_break = True
    backticks = 0
    for token, value in tokens:
        if token in Comment or not value.strip():
            line_break = line_break or "\n" in value
            continue
        is_quoted = token in String.Single or token in String.Double
        if is_quoted and _RAW_LINE_BREAK.search(value):
            return None
        if token in String.Backtick:
            if value == "`":
                backticks += 1
            elif _SWALLOWED_IMPORT.search(value):
                return None
        values.append(value)
        quoted.append(is_quoted and len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0])
        starts_line.append(line_break)
        line_break = False
    if backticks % 2:  # a template literal left open to the end of the file
        return None

    def statement_start(index: int) -> bool:
        return starts_line[index] or values[index - 1] in (";", "}")

    specs: list[str] = []
    for i in range(1, len(values)):
        value = values[i]
        if not quoted[i] or len(value) < 3:
            continue
        if i + 1 < len(values) and values[i + 1].startswith("+"):
            continue  # the prefix of a path built at runtime
        before = values[i - 1]
        if before == "import":
            found = statement_start(i - 1)
        elif before == "(":
            found = i >= 2 and values[i - 2] in ("import", "require")
        elif before == "from":
            j = i - 2
            while j >= 0 and values[j] not in ("import", "export") and (
                    values[j] in _CLAUSE_PUNCTUATION or quoted[j] or _IDENTIFIER.match(values[j])):
                j -= 1
            found = j >= 0 and values[j] in ("import", "export") and statement_start(j)
        else:
            found = False
        if found:
            specs.append(value[1:-1])
    return tuple(specs)


class _LexedSource(NamedTuple):
    code: str  # the source without its comments
    specs: tuple[str, ...]  # its import specifiers, as written


@lru_cache(maxsize=512)
def _lexed_file(path: str, mtime_ns: int, size: int) -> _LexedSource:
    tokens = _tokens(_read(path), os.path.basename(path))
    code = _without_comments(tokens)
    specs = _token_specs(tokens)
    if specs is None:
        specs = tuple(match.group(2) for match in _SPEC_RE.finditer(code))
    return _LexedSource(code, specs)


def _lexed_source(path: str) -> _LexedSource:
    """A source file lexed once per file version."""
    try:
        stat = os.stat(path)
    except OSError:
        return _LexedSource("", ())
    return _lexed_file(path, stat.st_mtime_ns, stat.st_size)


def _stripped_source(path: str) -> str:
    """A source file without its comments, lexed once per file version."""
    return _lexed_source(path).code


def import_specifiers(path: str) -> tuple[str, ...]:
    """The import specifiers a JS/TS source file makes, read from code tokens."""
    return _lexed_source(path).specs


def _walk(root: str):
    # Imported here so the module stays runnable as a stdlib-only script.
    from besser.utilities.workspace_fs import walk_plain

    for folder, dirs, files in walk_plain(root):
        dirs[:] = sorted(d for d in dirs
                         if d not in _EXCLUDED_DIRS and not d.startswith(".besser_"))
        yield folder, files


def _read(path: str) -> str:
    # Imported here so the module stays runnable as a stdlib-only script.
    from besser.utilities.workspace_fs import open_plain

    try:
        with open_plain(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _package_name(spec: str) -> str:
    parts = spec.split("/")
    if spec.startswith("@") and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def _declared_packages(manifest: dict) -> set[str]:
    names: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies",
                "optionalDependencies", "bundledDependencies"):
        block = manifest.get(key)
        if isinstance(block, dict):
            names.update(block)
        elif isinstance(block, list):
            names.update(block)
    return names


# Specs that name no version range: npm installs whatever is newest at build
# time. Peer ranges are left out; "*" is a normal peer range.
_UNPINNED_VERSIONS = {"latest", "*", "x", ""}


def _unpinned_dependencies(manifest: dict) -> list[tuple[str, str, str]]:
    found = []
    for key in ("dependencies", "devDependencies", "optionalDependencies"):
        block = manifest.get(key)
        if not isinstance(block, dict):
            continue
        for name, version in block.items():
            if isinstance(version, str) and version.strip().lower() in _UNPINNED_VERSIONS:
                found.append((key, name, version))
    return found


def _resolves(base_dir: str, spec: str) -> bool:
    target = os.path.normpath(os.path.join(base_dir, spec))
    for ext in _RESOLVE_EXT:
        candidate = target + ext
        if ext == "" and os.path.isdir(candidate):
            continue
        if os.path.isfile(candidate):
            return True
    if os.path.isdir(target):
        for ext in _RESOLVE_EXT[1:]:
            if os.path.isfile(os.path.join(target, "index" + ext)):
                return True
    stem, ext = os.path.splitext(target)
    return any(os.path.isfile(stem + alt) for alt in _TS_SOURCE_FOR.get(ext, ()))


def _has_path_aliases(project_dir: str) -> bool:
    """Alias config means a bare/absolute specifier may resolve elsewhere."""
    for name in ("tsconfig.json", "jsconfig.json"):
        text = _read(os.path.join(project_dir, name))
        if '"paths"' in text or '"baseUrl"' in text:
            return True
    for folder, files in _walk(project_dir):
        for name in files:
            if name.startswith(("vite.config", "webpack.config", "rollup.config")):
                text = _read(os.path.join(folder, name))
                if "alias" in text:
                    return True
    return False


def _registers_vite_react_plugin(config: str) -> bool:
    """The React plugin is imported AND called inside ``plugins: [...]``.

    Declaring ``@vitejs/plugin-react`` in package.json, or naming it in a
    comment, leaves Vite on esbuild's classic transform.
    """
    # Imported here so the module stays runnable as a stdlib-only script.
    from besser.spec_driven_agent.validation.frontend_source import match_bracket

    bindings = re.findall(
        r"""import\s+(\w+)\s+from\s+['"](?:@vitejs/plugin-react(?:-swc)?|@preact/preset-vite)['"]""",
        config)
    for match in re.finditer(r"\bplugins\s*:\s*\[", config):
        # Bracket-balanced: an earlier plugin's options may hold a "]".
        end = match_bracket(config, match.end() - 1)
        plugins = config[match.end():end]  # end None: the rest of the file
        if any(re.search(rf"\b{binding}\s*\(", plugins) for binding in bindings):
            return True
    return False


def _automatic_jsx_runtime(project_dir: str, manifest: dict) -> bool:
    """True when JSX compiles without React in lexical scope."""
    declared = _declared_packages(manifest)
    if declared & {"react-scripts", "next", "@babel/preset-react", "@rsbuild/plugin-react"}:
        return True
    for name in ("tsconfig.json", "jsconfig.json"):
        text = _read(os.path.join(project_dir, name))
        if re.search(r'"jsx"\s*:\s*"react-jsx(dev)?"', text):
            return True
    for folder, files in _walk(project_dir):
        for name in files:
            if not name.startswith(("vite.config", "babel.config", "webpack.config",
                                    "rsbuild.config", "esbuild")) and name != ".babelrc":
                continue
            text = _stripped_source(os.path.join(folder, name))
            if name.startswith("vite.config"):
                if _registers_vite_react_plugin(text):
                    return True
            elif "plugin-react" in text or "preset-react" in text:
                return True
            if re.search(r"jsx\s*:\s*['\"]automatic['\"]", text):
                return True
            if re.search(r"runtime\s*:\s*['\"]automatic['\"]", text):
                return True
    return False


def collect_frontend_resolution_issues(output_dir: str) -> list[str]:
    """Report imports a generated frontend cannot resolve from its own files.

    Pure filesystem analysis: no install, no bundler, no shell.

    Two severities, by how certain the finding is. An import that resolves to
    no file on disk, and JSX with no React in scope and no automatic runtime,
    are decidable here and carry ``frontend contract:`` - blocker, same family
    as the other frontend findings. An undeclared package is the one a
    bundler can still satisfy (hoisting, a workspace link, an implicit peer),
    so it carries ``frontend dependency:``, which classifies as a warning:
    recorded, not driving the repair loop (an undeclared ``axios`` can still
    leave the page blank). A dependency with no version range is a
    ``frontend contract:`` blocker too.
    """
    workspace = os.path.realpath(output_dir)
    issues: list[str] = []

    for project_dir, files in _walk(workspace):
        if "package.json" not in files:
            continue
        try:
            manifest = json.loads(_read(os.path.join(project_dir, "package.json")) or "{}")
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue

        declared = _declared_packages(manifest)
        aliased = _has_path_aliases(project_dir)
        automatic_jsx = _automatic_jsx_runtime(project_dir, manifest)
        rel_project = os.path.relpath(project_dir, workspace).replace("\\", "/")
        prefix = "" if rel_project == "." else rel_project + "/"

        for key, package, version in _unpinned_dependencies(manifest):
            issues.append(
                f"frontend contract: {prefix}package.json {key} '{package}' is "
                f"\"{version}\", which is not a version range. npm installs whatever "
                f"major is newest at build time; with \"latest\" the vite 8 tree crashes "
                f"npm 10.8's install (\"Cannot read properties of null (reading "
                f"'edgesOut')\"). Pin a concrete semver range such as \"^18.3.1\".")

        missing_paths: list[tuple[str, str]] = []
        missing_packages: dict[str, str] = {}
        jsx_without_react: list[str] = []

        for folder, names in _walk(project_dir):
            # A nested package.json owns its own subtree.
            if folder != project_dir and "package.json" in names:
                continue
            for name in names:
                # .d.ts files are type-only; nothing loads them at runtime.
                if not name.endswith(_SOURCE_EXT) or name.endswith(".d.ts"):
                    continue
                path = os.path.join(folder, name)
                source, specs = _lexed_source(path)
                rel_file = os.path.relpath(path, project_dir).replace("\\", "/")

                for written in specs:
                    # Vite drops a query suffix (?react, ?raw, ?url, ...) before resolving.
                    spec = written.split("?", 1)[0]
                    if spec.startswith("."):
                        if not _resolves(folder, spec):
                            missing_paths.append((rel_file, written))
                    elif spec.startswith("/") or spec.startswith("~") or spec.startswith("#"):
                        continue  # root/alias-relative: a bundler decides
                    else:
                        package = _package_name(spec)
                        if (package in declared or package in _NODE_BUILTINS
                                or spec in _IMPLICIT_PACKAGES or aliased):
                            continue
                        missing_packages.setdefault(package, rel_file)

                if (not automatic_jsx and name.endswith((".jsx", ".tsx"))
                        and not re.search(r"^\s*import\s+(?:\*\s+as\s+)?React\b", source, re.M)
                        and not re.search(r"^\s*import\s+React\s*,", source, re.M)
                        and re.search(r"<[A-Za-z][\w.]*[\s/>]", source)):
                    jsx_without_react.append(rel_file)

        for rel_file, spec in sorted(set(missing_paths)):
            issues.append(
                f"frontend contract: {prefix}{rel_file} imports '{spec}', which does not "
                f"exist in the generated tree. The page never mounts and the production "
                f"build fails. Write the missing module or drop the import.")
        for package, rel_file in sorted(missing_packages.items()):
            issues.append(
                f"frontend dependency: {prefix}{rel_file} imports the package '{package}', "
                f"which is not declared in {prefix}package.json. Nothing installs it, so "
                f"the import fails at load. Add it to dependencies or stop importing it.")
        if jsx_without_react:
            # One finding per project: the cause is the project's config.
            shown = sorted(set(jsx_without_react))
            listed = ", ".join(prefix + rel for rel in shown[:_JSX_FILES_LISTED])
            if len(shown) > _JSX_FILES_LISTED:
                listed += f" and {len(shown) - _JSX_FILES_LISTED} more"
            issues.append(
                f"frontend contract: {listed} contain JSX but do not import "
                f"React, and this project configures no automatic JSX runtime (react() is not "
                f"registered in vite.config's plugins: [...] - declaring @vitejs/plugin-react "
                f"or mentioning it in a comment does not enable it - and no \"jsx\": "
                f"\"react-jsx\"). The classic transform emits React.createElement and every "
                f"page dies with 'React is not defined'. Import the plugin and add react() "
                f"to plugins, or add 'import React from \"react\"' to each file.")

    return issues


if __name__ == "__main__":  # manual probe: python frontend_resolution.py <dir>
    import sys
    for finding in collect_frontend_resolution_issues(sys.argv[1]):
        print(finding)
