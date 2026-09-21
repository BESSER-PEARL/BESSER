"""Whether the generated project compiles on its own toolchain.

One collector per toolchain - ruff for Python, ``tsc`` for TypeScript,
``cargo check`` for Rust, ``kotlinc`` for Kotlin - plus the two helpers that
turn what they report into the re-run commands and the reminder Phase 3 puts
in front of the model.

A missing or disabled compiler is an unverified check, never a source defect:
reporting clean because nothing ran is how a frontend that does not compile
passes validation.

Split out of ``orchestrator.py``. The run state each collector read off
``self`` - the workspace, the tool-call log, the two shell/toolchain
permissions, the warn-once flag - is passed in instead. The flag stays an
orchestrator attribute and comes back out with the ruff issue list.
"""

from __future__ import annotations

import logging
import os
import re as _re
import tempfile

from besser.spec_driven_agent.checkpoint import _SNAPSHOT_DIR
from besser.spec_driven_agent.execution.process import _safe_subprocess_env
from besser.spec_driven_agent.validation.issues import (
    ValidationIssue,
    _check_did_not_run,
    required_check_unverified,
    required_dependency_setup,
    _RUFF_BLOCKER_CODES,
    _RUFF_LINE_RE,
    _RUFF_STYLE_CODES,
)

logger = logging.getLogger(__name__)


def _toolchain_commands_for(
    toolchain_blockers: list[ValidationIssue]
) -> list[str]:
    """Pick the right re-run command for each toolchain in the report.

    Looks at the prefix on each blocker message (``tsc [...]:``,
    ``cargo [...]:``, ``kotlinc [...]:``) and returns the matching
    command (with the project sub-path) the LLM should invoke via
    ``run_command`` to verify its fix. The list is deduplicated so
    the same command isn't suggested twice for a multi-error report.
    """
    commands: list[str] = []
    seen: set[str] = set()
    for issue in toolchain_blockers:
        msg = issue.message
        # Extract the bracketed sub-path: ``tsc [frontend]:`` -> ``frontend``
        match = _re.match(r"(tsc|cargo|kotlinc) \[([^\]]+)\]:", msg)
        if not match:
            continue
        tool, path = match.group(1), match.group(2)
        if tool == "tsc":
            # ``npx tsc --noEmit`` works whether or not tsc is on
            # PATH globally — npm projects nearly always have it
            # installed locally as a devDependency.
            cmd = (
                f"run_command: command='npx tsc --noEmit', "
                f"working_dir='{path}'"
            )
        elif tool == "cargo":
            cmd = (
                f"run_command: command='cargo check', "
                f"working_dir='{path}'"
            )
        elif tool == "kotlinc":
            # The .kt files under the module root, compiled to
            # /dev/null. The bench uses kotlinc directly too.
            cmd = (
                f"run_command: command='kotlinc -nowarn "
                f"-d /tmp/out $(find {path} -name \"*.kt\")', "
                f"working_dir='.'"
            )
        else:  # pragma: no cover - defensive
            continue
        if cmd not in seen:
            commands.append(cmd)
            seen.add(cmd)
    return commands


def _build_toolchain_reminder(
    toolchain_blockers: list[ValidationIssue]
) -> str:
    """High-salience reminder text for the toolchain-fix loop.

    Mirrors the shape of ``_build_modify_loop_reminder`` (the
    per-file modify-loop guard the LLM already recognises) so the
    model treats the toolchain errors with the same urgency as a
    rewrite-or-quit warning. The verbatim error lines are quoted
    in the reminder so they sit in the model's working memory
    right before its next response.
    """
    # Cap the reminder body so a runaway report (e.g. 100 type
    # errors after a single missing import) doesn't blow out the
    # context window. The full list is already in the prompt.
    lines = [i.message for i in toolchain_blockers[:8]]
    bulleted = "\n".join(f"  - {ln}" for ln in lines)
    more = (
        f"\n  - (+{len(toolchain_blockers) - 8} more — see prompt for full list)"
        if len(toolchain_blockers) > 8 else ""
    )
    return (
        "<system-reminder>"
        "The generated project does not compile on its own toolchain. "
        "The bench's per-project compile-pass score is currently 0 "
        "for this run because of these errors:\n"
        f"{bulleted}{more}\n\n"
        "You MUST drive these to zero. After EACH edit, invoke "
        "run_command with the appropriate toolchain check (npx tsc "
        "--noEmit / cargo check / kotlinc) and read the output. "
        "Only call end_turn once the toolchain reports zero errors, "
        "or after you have made a clear good-faith attempt that the "
        "remaining errors require dependencies you cannot add."
        "</system-reminder>"
    )


# How many ruff lines Phase 3 reports. Blocker-code lines are always kept
# even past this, then files the LLM edited, then the rest.
_RUFF_MAX_REPORTED = 20
# A real concise-format line always carries a file, a position and a rule.
_CONCISE_FINDING_RE = _re.compile(r"^.+?:\d+:\d+: \S+ ")


def _collect_ruff_issues(
    output_dir: str,
    tool_calls_log: list[dict],
    write_tools: frozenset[str],
    warned_missing: bool,
) -> tuple[list[str], bool]:
    """Run ``ruff check`` across the workspace when available.

    Returns a list of concise issue strings. Times out / unparseable
    output are skipped silently (nice-to-have checks). A MISSING ruff
    binary, however, is reported loudly: ``_classify_issue`` promotes
    ruff's undefined-name findings to blockers ("ships green, boots
    dead"), so when ruff is absent that whole class of defect is
    invisible and a "0 blockers" result is not the verification it
    looks like. Found live 2026-09-10: ruff was never in the hosted
    image (only in CI), so this path returned [] for every pilot run —
    including the two that shipped a backend NameError-ing on import.

    ``warned_missing`` comes in and goes back out with the issue list so the
    warning still fires once per orchestrator, which owns the flag.
    """
    import shutil as _shutil
    import subprocess

    ruff_bin = _shutil.which("ruff")
    if not ruff_bin:
        if not warned_missing:
            warned_missing = True
            logger.warning(
                "Phase 3: ruff is not installed on this host — Python "
                "undefined-name/import checks are SKIPPED, so '0 blockers' "
                "does not cover import-time NameErrors. Install ruff in the "
                "image to enable them."
            )
        # A visible (non-blocking) validation note, deliberately phrased
        # without a ruff rule code so _classify_issue keeps it a warning.
        return [
            "validation: ruff is not installed on this host - Python "
            "undefined-name/import checks were skipped (install ruff to "
            "enable them)"
        ], warned_missing

    try:
        result = subprocess.run(
            [
                ruff_bin, "check",
                "--output-format=concise",
                "--no-cache",
                "--exit-zero",
                "--exclude", _SNAPSHOT_DIR,
                output_dir,
            ],
            capture_output=True, text=True, timeout=30,
            env=_safe_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return [_check_did_not_run("ruff", "timed out after 30s")], warned_missing
    except OSError as exc:
        return [_check_did_not_run("ruff", f"could not be launched: {exc}")], warned_missing

    # --exit-zero means findings alone never set a non-zero status, so a
    # non-zero code here is ruff itself failing (unreadable config, a
    # panic). Without this the run reported clean having checked nothing.
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1][:200] if detail else f"exit code {result.returncode}"
        return [_check_did_not_run("ruff", reason)], warned_missing

    # Concise format is ``<path>:<line>:<col>: <rule> <message>``; ruff also
    # prints its own trailer ("Found 630 errors.", "No fixes available ..."),
    # which carries no rule code, so _classify_issue made each one a warning
    # about nothing. They were invisible only because cosmetic findings used
    # to crowd them out of the cap.
    lines = [line.strip() for line in (result.stdout or "").strip().splitlines()
             if _CONCISE_FINDING_RE.match(line.strip())]
    if not lines:
        return [], warned_missing
    # The cap used to take ruff's first 20 lines, which are sorted by
    # path: on a 585-issue workspace that is always the same scaffold
    # files, and a real F821 late in the alphabet never reached the fix
    # loop at all. Keep every blocker-code line, then spend what is left
    # of the budget on files the LLM actually edited this run.
    # Ranking keys on the SEVERITY each line will be given, not only on the
    # blocker codes. Cosmetic lines used to compete for the same 20 slots as
    # real findings: every generated router star-imports sql_alchemy /
    # pydantic_classes / bal_stdlib, so F403/F405/E402 alone fill the budget
    # (59% of the kept lines across the labelled corpus, on apps that all
    # truncate). Sorting them behind everything else keeps them reported and
    # stops them hiding an actionable finding.
    touched = _llm_edited_paths(output_dir, tool_calls_log, write_tools)
    blockers = []
    edited, rest = [], []                  # actionable
    edited_cosmetic, rest_cosmetic = [], []
    for line in lines:
        match = _RUFF_LINE_RE.search(line)
        code = match.group(1) if match else None
        was_edited = _ruff_line_path(line) in touched
        if code in _RUFF_BLOCKER_CODES:
            blockers.append(line)
        elif code in _RUFF_STYLE_CODES:
            (edited_cosmetic if was_edited else rest_cosmetic).append(line)
        else:
            (edited if was_edited else rest).append(line)
    # Severity first, then whether the agent touched the file - the
    # edited-before-scaffold preference still holds inside each tier.
    ordered = blockers + edited + rest + edited_cosmetic + rest_cosmetic
    kept = ordered[:max(_RUFF_MAX_REPORTED, len(blockers))]
    issues = [f"ruff: {line}" for line in kept]
    if len(ordered) > len(kept):
        issues.append(f"ruff: (+{len(ordered) - len(kept)} more issues truncated)")
    return issues, warned_missing


def _llm_edited_paths(
    output_dir: str, tool_calls_log: list[dict], write_tools: frozenset[str]
) -> set[str]:
    """Absolute, normalised paths the LLM wrote to in this run."""
    edited = set()
    for call in tool_calls_log:
        if call.get("tool") not in write_tools or call.get("success") is False:
            continue
        path = (call.get("input") or {}).get("path")
        if isinstance(path, str) and path.strip():
            edited.add(os.path.normcase(os.path.normpath(
                os.path.join(output_dir, path.replace("\\", "/")))))
    return edited


def _ruff_line_path(line: str) -> str:
    """The file part of a concise ruff line, or "" when unparseable.

    ``<path>:<line>:<col>: <CODE> <message>`` — a Windows drive letter
    puts an extra colon in the path, so split from the right.
    """
    head = line.rsplit(":", 3)
    if len(head) != 4:
        return ""
    return os.path.normcase(os.path.normpath(head[0]))


def _collect_tsc_issues(
    output_dir: str,
    allow_shell_tools: bool,
    enable_toolchain_validation: bool,
    excluded_dirs: set[str],
) -> list[str]:
    """Run ``tsc --noEmit`` for any TypeScript project in the workspace.

    Looks for ``tsconfig.json`` files (skipping the snapshot dir) and
    runs ``tsc --noEmit`` against each project root. A disabled or missing
    compiler is unverified, not a source defect for automatic repair.
    """
    import shutil as _shutil
    import subprocess

    tsconfigs: list[str] = []
    workspace = os.path.realpath(output_dir)
    for root, dirs, files in os.walk(output_dir):
        retained = []
        for directory in dirs:
            if directory in excluded_dirs or directory.startswith(".besser_"):
                continue
            try:
                if os.path.commonpath([workspace, os.path.realpath(os.path.join(root, directory))]) == workspace:
                    retained.append(directory)
            except ValueError:
                continue
        dirs[:] = retained
        rel_root = os.path.relpath(root, output_dir).replace("\\", "/")
        if rel_root.startswith(_SNAPSHOT_DIR):
            continue
        # Skip node_modules — running tsc there is both meaningless
        # and extremely slow.
        if "node_modules" in rel_root.split("/"):
            continue
        if "tsconfig.json" in files:
            try:
                if os.path.commonpath([workspace, os.path.realpath(os.path.join(root, "tsconfig.json"))]) == workspace:
                    tsconfigs.append(root)
            except ValueError:
                continue

    if not tsconfigs:
        return []

    global_tsc = _shutil.which("tsc") or _shutil.which("tsc.cmd")
    issues: list[str] = []
    for project_dir in tsconfigs:
        rel = os.path.relpath(project_dir, output_dir).replace("\\", "/") or "."
        if not enable_toolchain_validation:
            issues.append(required_check_unverified(
                f"tsc [{rel}]", "toolchain validation is disabled"))
            continue
        deps_installed = os.path.isdir(os.path.join(project_dir, "node_modules"))
        tsc_bin = global_tsc
        if allow_shell_tools:
            # Project executables are package-authored code. Only prefer
            # them when shell execution was explicitly authorized.
            local_tsc = os.path.join(project_dir, "node_modules", ".bin",
                                     "tsc.cmd" if os.name == "nt" else "tsc")
            if os.path.isfile(local_tsc):
                tsc_bin = local_tsc
        setup_allowed = (
            not deps_installed and allow_shell_tools
            and os.path.isfile(os.path.join(project_dir, "package.json"))
            and bool(_shutil.which("npm") or _shutil.which("npm.cmd"))
        )
        if setup_allowed:
            issues.append(required_dependency_setup(f"tsc [{rel}]", rel))
        if not tsc_bin:
            if not setup_allowed:
                issues.append(required_check_unverified(f"tsc [{rel}]", "TypeScript compiler is unavailable"))
            continue
        if not deps_installed and not setup_allowed:
            issues.append(required_check_unverified(f"tsc [{rel}]", "dependencies are not installed; only partial source checks are possible"))
        project_arg, cleanup = _tsc_project_arg(project_dir, deps_installed)
        try:
            result = subprocess.run(
                [tsc_bin, "--noEmit", "-p", project_arg],
                capture_output=True, text=True, timeout=60,
                cwd=project_dir,
                env=_safe_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            issues.append(required_check_unverified(f"tsc [{rel}]", "timed out after 60s"))
            continue
        except OSError as exc:
            issues.append(
                required_check_unverified(f"tsc [{rel}]", f"could not be launched: {exc}")
            )
            continue
        finally:
            cleanup()

        # tsc emits errors on stdout (not stderr) in the classic
        # ``file(line,col): error TSxxxx: message`` format.
        output = (result.stdout or "").strip().splitlines()
        err_lines = [ln.strip() for ln in output if ln.strip() and "error" in ln.lower()]
        if not err_lines and result.returncode == 0:
            continue
        if not err_lines:
            # Non-zero exit with nothing parseable — a missing typescript
            # install, a broken tsconfig. Reporting clean here is how a
            # frontend that does not compile passes validation.
            detail = (
                (result.stderr or "").strip().splitlines()
                + (result.stdout or "").strip().splitlines()
            )
            tail = detail[-1][:200] if detail else f"exit code {result.returncode}"
            issues.append(f"tsc [{rel}]: {tail}")
            continue
        if not deps_installed:
            err_lines = _demote_tsc_without_deps(err_lines, rel, issues)
            if not err_lines:
                continue
        # One name repeated 8 times would fill the cap and hide the other
        # 7 distinct names behind "truncated", costing a whole fix round.
        err_lines = _collapse_repeated_names(err_lines)
        for line in err_lines[:10]:
            issues.append(f"tsc [{rel}]: {line}")
        if len(err_lines) > 10:
            issues.append(f"tsc [{rel}]: (+{len(err_lines) - 10} more errors truncated)")
    return issues


def _collapse_repeated_names(err_lines: list[str]) -> list[str]:
    """Keep the first occurrence of each undefined name per file."""
    seen: set[tuple[str, str]] = set()
    kept: list[str] = []
    for line in err_lines:
        match = _TSC_UNDEFINED_NAME_RE.match(line.strip())
        if not match:
            kept.append(line)
            continue
        key = (match.group("file"), match.group("name"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return kept


# A relative import names a file the run was supposed to write; a bare
# one names a package. Only the first is checkable without an install.
_TSC_MISSING_MODULE_RE = _re.compile(
    r"error TS2307:.*?Cannot find module ['\"](?P<spec>[^'\"]+)['\"]")
# TS1000-TS1999 is the syntactic range; no dependency install can change it.
_TSC_SYNTAX_RE = _re.compile(r"error TS1\d{3}:")


# Written next to the real tsconfig so its relative include/exclude/baseUrl
# still resolve, then removed.
_TSC_PROBE_NAME = "tsconfig.besser-probe.json"
# target/moduleResolution are overridden too: TypeScript 7 REMOVED
# `target: es5` and `moduleResolution: node`, which every CRA-era
# scaffold still carries, and a removed option is TS5108 at config
# time -- the same zero-files-checked abort this probe exists to
# avoid. Neither option affects TS2304, the only code promoted here.
_TSC_PROBE_BODY = (
    '{\n  "extends": "./tsconfig.json",\n'
    '  "compilerOptions": {\n'
    '    "types": [], "noEmit": true,\n'
    '    "target": "es2020", "module": "esnext", "moduleResolution": "bundler"\n'
    '  }\n}\n'
)


def _tsc_project_arg(project_dir: str, deps_installed: bool):
    """Return the ``-p`` target for tsc, plus a cleanup callable.

    A ``types`` entry naming an uninstalled package (``"types":
    ["vite/client"]``, which every Vite scaffold carries) makes tsc abort
    at config resolution: it emits TS2688 and type-checks ZERO files. Run
    36e9c8a6 shipped a frontend with 23 real errors whose entire tsc
    output was that one line, so Phase 3 saw a clean frontend.

    Clearing ``types`` costs nothing when deps are missing — those types
    cannot resolve either way — and lets tsc actually read the source.
    There is no CLI equivalent: ``--types ""`` is TS6044 and
    ``--typeRoots`` does not suppress an explicit ``types`` entry.
    """
    if deps_installed:
        return ".", lambda: None

    probe = os.path.join(project_dir, _TSC_PROBE_NAME)
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write(_TSC_PROBE_BODY)
    except OSError:
        return ".", lambda: None

    def cleanup():
        try:
            os.remove(probe)
        except OSError:
            pass

    return _TSC_PROBE_NAME, cleanup


# TS2304 means a name has no binding in scope. That is true whether or not
# packages are installed -- EXCEPT for names a package contributes as a
# global type declaration, which are unresolvable only because of the
# missing install.
_TSC_PACKAGE_GLOBALS = frozenset({
    # test runners (vitest / jest globals)
    "describe", "it", "test", "expect", "vi", "jest", "suite", "assert",
    "beforeEach", "afterEach", "beforeAll", "afterAll", "afterFile",
    # node
    "process", "Buffer", "__dirname", "__filename", "global", "require",
    "module", "exports", "NodeJS", "globalThis",
    # react UMD global / JSX namespace
    "React", "JSX",
})

_TSC_UNDEFINED_NAME_RE = _re.compile(
    r"^(?P<file>[^(]+)\(.*?error TS2304: Cannot find name '(?P<name>[^']+)'"
)


def _is_real_undefined_name(line: str) -> bool:
    """True for a TS2304 that a ``npm install`` would not have fixed."""
    match = _TSC_UNDEFINED_NAME_RE.match(line.strip())
    if not match:
        return False
    if match.group("name") in _TSC_PACKAGE_GLOBALS:
        return False
    path = match.group("file").replace("\\", "/").lower()
    base = path.rsplit("/", 1)[-1]
    # Test files pull in runner globals we cannot enumerate; skip them.
    if "__tests__" in path or "__mocks__" in path:
        return False
    return not any(
        f".{kind}." in base for kind in ("test", "spec", "stories", "cy")
    )


def _demote_tsc_without_deps(
    err_lines: list[str], rel: str, issues: list[str]
) -> list[str]:
    """Keep only the tsc errors that survive a missing ``node_modules``.

    We never run ``npm install`` during validation — it needs network
    the host may not have, and on a proxied corporate network it fails
    outright. So tsc runs against an uninstalled tree, where every
    package import is unresolvable and the type errors cascade from
    there. On the 2026-09-17 hotel run that produced
    ``error TS2688: Cannot find type definition file for 'vite/client'``
    and a "3 blocker-level issues remain — may not run as-is" verdict
    on a frontend that starts and renders perfectly once installed.

    What tsc CAN still tell us truthfully is whether a locally
    referenced file exists: ``import Foo from './components/Foo'``
    where the run never wrote that file is a genuine break either way.
    Everything else is reported, but as advisory — it must not drive
    the Phase-3 fix loop, which would spend its budget "fixing"
    imports that are already correct.
    """
    real: list[str] = []
    demoted = 0
    for line in err_lines:
        match = _TSC_MISSING_MODULE_RE.search(line)
        if match and match.group("spec").startswith("."):
            real.append(line)
        elif _is_real_undefined_name(line):
            real.append(line)
        elif _TSC_SYNTAX_RE.search(line):
            # TS1xxx is the grammar, not the type system: an unparseable file
            # is unparseable installed or not. Demoting these is how a
            # Booking.tsx that no bundler can read reached "workflow_ok".
            real.append(line)
        else:
            demoted += 1
            if demoted <= 5:
                issues.append(f"tsc-advisory [{rel}]: {line}")
    if demoted:
        issues.append(
            f"tsc-advisory [{rel}]: {demoted} type error(s) reported without "
            "node_modules installed — package imports and their types cannot "
            "resolve, so these are advisory, not blockers. Run npm install "
            "before trusting them."
        )
    return real


def _collect_cargo_issues(output_dir: str) -> list[str]:
    """Run ``cargo check`` for any Rust crate in the workspace.

    Mirrors ``_collect_tsc_issues`` for the Rust toolchain. Looks
    for ``Cargo.toml`` files at any depth (skipping the snapshot
    dir and any ``target/`` build output) and runs ``cargo check
    --message-format=short`` per crate. Skips silently if ``cargo``
    is not on PATH — matches the soft-skip pattern the bench uses
    when the toolchain isn't installed on the run host.

    ``cargo check`` is used in preference to ``cargo build``: it
    runs the front-end and type-checker without producing artifacts,
    which is what the per-project compile-pass criterion actually
    cares about and is ~3-5× faster.
    """
    import shutil as _shutil
    import subprocess

    cargo_bin = _shutil.which("cargo") or _shutil.which("cargo.exe")
    if not cargo_bin:
        return []

    crates: list[str] = []
    for root, _, files in os.walk(output_dir):
        rel_root = os.path.relpath(root, output_dir).replace("\\", "/")
        if rel_root.startswith(_SNAPSHOT_DIR):
            continue
        # ``target/`` is the cargo build cache — running cargo
        # inside it is meaningless. Also skip any vendored deps.
        parts = rel_root.split("/")
        if "target" in parts or "vendor" in parts:
            continue
        if "Cargo.toml" in files:
            crates.append(root)

    if not crates:
        return []

    # Redirect cargo's build cache OUT of the user workspace: without
    # this, ``target/`` (thousands of files for a typical axum crate)
    # lands inside the output dir — bloating the download zip — and
    # every check cold-compiles all dependency crates from scratch.
    # A shared per-host cache dir makes repeat checks incremental.
    # Strip secrets from the env handed to cargo (it can execute build.rs /
    # proc-macros from generated crates). Keeps PATH/HOME so cargo still
    # resolves its toolchain + ~/.cargo.
    cargo_env = {**_safe_subprocess_env()}
    cargo_env.setdefault(
        "CARGO_TARGET_DIR",
        os.path.join(tempfile.gettempdir(), "besser_cargo_cache"),
    )

    issues: list[str] = []
    for crate_dir in crates:
        rel = os.path.relpath(crate_dir, output_dir).replace("\\", "/") or "."
        try:
            result = subprocess.run(
                [
                    cargo_bin, "check",
                    "--message-format=short",
                    "--quiet",
                ],
                capture_output=True, text=True, timeout=180,
                cwd=crate_dir,
                env=cargo_env,
            )
        except subprocess.TimeoutExpired:
            issues.append(_check_did_not_run(f"cargo [{rel}]", "timed out after 180s"))
            continue
        except OSError as exc:
            issues.append(
                _check_did_not_run(f"cargo [{rel}]", f"could not be launched: {exc}")
            )
            continue

        # cargo emits diagnostics on stderr in short format like:
        #   src/main.rs:12:5: error[E0308]: mismatched types
        err_lines = []
        for ln in (result.stderr or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith("error") or ": error" in s:
                err_lines.append(s)
        if not err_lines and result.returncode == 0:
            continue
        if not err_lines:
            # Non-zero exit with no parseable error lines (rare —
            # network failure resolving deps, missing rustc, etc.).
            # Surface a single summary line so the LLM can decide
            # whether to address it.
            summary = (result.stderr or "").strip().splitlines()
            tail = summary[-1] if summary else "cargo check failed with no output"
            issues.append(f"cargo [{rel}]: {tail[:200]}")
            continue
        for line in err_lines[:10]:
            issues.append(f"cargo [{rel}]: {line}")
        if len(err_lines) > 10:
            issues.append(
                f"cargo [{rel}]: (+{len(err_lines) - 10} more errors truncated)"
            )
    return issues


def _collect_kotlinc_issues(output_dir: str) -> list[str]:
    """Run ``kotlinc`` against ``.kt`` sources in the workspace.

    Kotlin / Spring projects from Phase 0.5 ship with a Gradle
    build, but invoking the Gradle wrapper would pull the network
    on first run and is far too slow for an inner-loop check. We
    instead run the standalone ``kotlinc`` compiler on the
    ``src/main/kotlin`` tree with no class-path (Spring annotations
    and missing imports still surface as compile errors).

    Limitations (documented for the caller, not bugs):
      - Type references to external Maven deps will show up as
        unresolved-reference errors. That's the right call here —
        it tells the LLM the import / dep listing is wrong, and
        the project will fail Gradle in the same way.
      - We only walk one source root per Kotlin module to keep
        the invocation cheap. Multi-module projects compile one
        module at a time.

    Soft-skips when ``kotlinc`` is not on PATH (no warning in the
    recipe — the bench host either has it or doesn't).
    """
    import shutil as _shutil
    import subprocess

    kotlinc_bin = (
        _shutil.which("kotlinc")
        or _shutil.which("kotlinc.bat")
        or _shutil.which("kotlinc.cmd")
    )
    if not kotlinc_bin:
        return []

    # Locate Kotlin source roots. We look for ``src/main/kotlin``
    # under any directory containing a Gradle build file, which is
    # the convention every Phase 0.5 Kotlin template lands in.
    modules: list[str] = []
    for root, dirs, files in os.walk(output_dir):
        rel_root = os.path.relpath(root, output_dir).replace("\\", "/")
        if rel_root.startswith(_SNAPSHOT_DIR):
            continue
        parts = rel_root.split("/")
        if "build" in parts or ".gradle" in parts:
            # Don't recurse into build output / Gradle caches.
            dirs[:] = []
            continue
        has_gradle = (
            "build.gradle.kts" in files
            or "build.gradle" in files
        )
        if not has_gradle:
            continue
        src_main_kotlin = os.path.join(root, "src", "main", "kotlin")
        if os.path.isdir(src_main_kotlin):
            modules.append(src_main_kotlin)

    if not modules:
        return []

    issues: list[str] = []
    for src_root in modules:
        module_rel = (
            os.path.relpath(src_root, output_dir).replace("\\", "/") or "."
        )
        # Collect every .kt file under the source root. Limited to
        # 200 sources per invocation to keep the command line in
        # bounds on Windows; if a project exceeds that, the rest
        # are skipped (and the LLM still sees the first batch).
        kt_files: list[str] = []
        for kt_root, _, kt_files_in_dir in os.walk(src_root):
            for fname in kt_files_in_dir:
                if fname.endswith(".kt"):
                    kt_files.append(os.path.join(kt_root, fname))
                    if len(kt_files) >= 200:
                        break
            if len(kt_files) >= 200:
                break
        if not kt_files:
            continue

        try:
            result = subprocess.run(
                [kotlinc_bin, "-nowarn", "-d", os.devnull, *kt_files],
                capture_output=True, text=True, timeout=180,
                cwd=output_dir,
                env=_safe_subprocess_env(),
            )
        except (subprocess.TimeoutExpired, OSError):
            continue

        # kotlinc reports diagnostics on stderr as
        #   /abs/path/Foo.kt:12:5: error: unresolved reference: Bar
        err_lines = []
        for ln in (result.stderr or "").splitlines():
            s = ln.strip()
            if not s or ": warning:" in s:
                continue
            if ": error:" in s or s.startswith("error:"):
                # Strip the absolute path prefix so the LLM sees
                # the location relative to the workspace.
                err_lines.append(
                    s.replace(output_dir + os.sep, "")
                     .replace(output_dir + "/", "")
                )
        if not err_lines and result.returncode == 0:
            continue
        if not err_lines:
            tail = (result.stderr or "").strip().splitlines()
            summary = tail[-1] if tail else "kotlinc failed with no output"
            issues.append(f"kotlinc [{module_rel}]: {summary[:200]}")
            continue
        for line in err_lines[:10]:
            issues.append(f"kotlinc [{module_rel}]: {line}")
        if len(err_lines) > 10:
            issues.append(
                f"kotlinc [{module_rel}]: "
                f"(+{len(err_lines) - 10} more errors truncated)"
            )
    return issues
