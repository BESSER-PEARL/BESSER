"""Static resolution of a generated frontend: every import must actually exist.

Motivated by a browser sweep of the recorded corpus on 2026-09-21. Four of the
six apps driven in Chrome rendered nothing at all, and every one of them is
labelled WORKING - the label comes from probe_case.py, which boots the backend
and drives HTTP and never renders a page. The three failure shapes were:

    Qwen ... mjpuzh5s  App.jsx imports ./pages/OrderView.jsx, never written
    Qwen ... mz30st3s  App.jsx imports ./pages/FineList.jsx, never written,
                       and api.js imports axios, absent from package.json
    terra s7a5e4er     App.jsx uses JSX without importing React, in a project
    terra i4hedytk     with no vite.config and no @vitejs/plugin-react, so the
                       classic runtime is used and the page dies on mount

All three are decidable from the files on disk with no node_modules, no
install and no shell. ``frontend_build.py`` would also catch them, but it is
gated on ``enable_toolchain_validation and allow_shell``, which the hosted
product turns off, so in the shipping configuration nothing looks.

The checks deliberately stay narrow - an unresolvable path, an undeclared
package, a JSX file with no React in scope and no automatic runtime
configured. Anything needing a bundler's resolution rules (aliases,
conditional exports, monorepo links) is skipped rather than guessed.
"""

from __future__ import annotations

import json
import os
import re

_EXCLUDED_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__",
                  "dist", "build", ".next", "out", "coverage", "target"}
_SOURCE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_RESOLVE_EXT = ("", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json",
                ".css", ".scss", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")

# from/import/require/dynamic-import specifiers
_SPEC_RE = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|\bimport\s+)"""
    r"""(['"])([^'"\n]+)\1""")

_NODE_BUILTINS = {
    "assert", "buffer", "child_process", "cluster", "console", "constants",
    "crypto", "dgram", "dns", "domain", "events", "fs", "http", "http2",
    "https", "module", "net", "os", "path", "perf_hooks", "process",
    "punycode", "querystring", "readline", "repl", "stream", "string_decoder",
    "timers", "tls", "tty", "url", "util", "v8", "vm", "worker_threads", "zlib",
}

# Packages a bundler or framework provides without a package.json entry.
_IMPLICIT_PACKAGES = {"react/jsx-runtime", "react/jsx-dev-runtime", "vite/client"}

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"(?<![:'\"/])//[^\n]*")


def _strip_comments(text: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _walk(root: str):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs
                         if d not in _EXCLUDED_DIRS and not d.startswith(".besser_"))
        yield folder, files


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
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
    # A ".js" specifier that a TS project writes for a ".ts" source.
    stem, ext = os.path.splitext(target)
    if ext in (".js", ".jsx"):
        for alt in (".ts", ".tsx"):
            if os.path.isfile(stem + alt):
                return True
    return False


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


def _automatic_jsx_runtime(project_dir: str, manifest: dict) -> bool:
    """True when JSX compiles without React in lexical scope."""
    declared = _declared_packages(manifest)
    if declared & {"@vitejs/plugin-react", "@vitejs/plugin-react-swc",
                   "react-scripts", "next", "@preact/preset-vite",
                   "@babel/preset-react", "@rsbuild/plugin-react"}:
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
            text = _read(os.path.join(folder, name))
            if "plugin-react" in text or "preset-react" in text:
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
    recorded, not driving the repair loop. The evidence for keeping it at all
    is run mz30st3s, whose blank page was an undeclared ``axios``.
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
                source = _strip_comments(_read(path))
                rel_file = os.path.relpath(path, project_dir).replace("\\", "/")

                for _, spec in _SPEC_RE.findall(source):
                    if spec.startswith("."):
                        if not _resolves(folder, spec):
                            missing_paths.append((rel_file, spec))
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
        for rel_file in sorted(set(jsx_without_react)):
            issues.append(
                f"frontend contract: {prefix}{rel_file} contains JSX but does not import "
                f"React, and this project configures no automatic JSX runtime (no "
                f"@vitejs/plugin-react, no \"jsx\": \"react-jsx\"). The classic transform "
                f"emits React.createElement and the page dies with 'React is not defined'. "
                f"Add 'import React from \"react\"' or configure the automatic runtime.")

    return issues


if __name__ == "__main__":  # manual probe: python frontend_resolution.py <dir>
    import sys
    for finding in collect_frontend_resolution_issues(sys.argv[1]):
        print(finding)
