"""What the lab agent is given before it reads a file: the skill index and an orientation in the repositories.

The skill index lists ``<repo>/.claude/skills/*/SKILL.md`` of every repository (name, path, the one-line
``metadata.when_to_read``); the agent reads a skill with ``read_file`` when its line fits the task. The
orientation holds the ``CLAUDE.md`` of the repositories in scope (in full or as a section index) and a map of
each repository built from ``git ls-tree`` / ``git log`` of its base commit: never from the working tree, so
untracked files and ``node_modules`` play no part, and the same commit always gives the same bytes. Maps are
cached in ``<state_dir>/maps/<repo>-<sha12>.md``. No model call builds any of it.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import yaml

from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_FRONTEND, REPO_MODELING_AGENT, REPO_ORDER, RepoSpec
from besser.spec_driven_agent.lab.kinds import _registry_entries
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError

logger = logging.getLogger(__name__)

#: Per repository: the map is read at the start of every step, so it stays a page, not a listing.
MAP_MAX_CHARS = 12_000
#: The whole orientation: BESSER's CLAUDE.md is 29 kB, the editor's 32 kB, the agent's 26 kB; not all fit.
ORIENTATION_MAX_CHARS = 60_000

# The system prompt's knowledge section is capped at 350 words; the skill index gets about a third.
_MAX_SKILL_INDEX_WORDS = 120
# One line in the index (the skill format's rule for ``when_to_read``).
_MAX_WHEN_TO_READ_CHARS = 160
# Enough history to see what the base branch has been working on, not a changelog.
_RECENT_COMMITS = 15
# The current base and the one before it (a draft's base while an older change is still open).
_MAPS_KEPT_PER_REPO = 2
# A short SHA long enough to stay unique in BESSER's history; fixed, so the map does not depend on git's abbrev.
_SHORT_SHA = 9
_CACHE_SHA = 12

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_ORIENTATION_HEADING = "## Orientation (from the base commits; your edits are not in it)"
_SECTIONS_HEADING = "#### CLAUDE.md sections (line ranges; read the lines you need)"

_REPO_TITLES = {REPO_BESSER: "BESSER", REPO_FRONTEND: "web editor", REPO_MODELING_AGENT: "modeling agent"}
_CONVERTERS = "besser/utilities/web_modeling_editor/backend/services/converters/"


@dataclass(frozen=True)
class _Landmarks:
    """What a repository's map shows besides its header, ``CLAUDE.md`` sections and recent commits."""

    #: ``a/*/b/*``: the folders matched by the last ``*``, each with its file count
    layout: tuple[str, ...]
    #: (title, folder): the files directly in the folder
    files: tuple[tuple[str, str], ...] = ()
    #: the generator registry, read with ``kinds._registry_entries``
    registry: str = ""
    #: (json_to_buml folder, buml_to_json folder): processors and converters paired by diagram
    converters: Optional[tuple[str, str]] = None


_LANDMARKS: dict[str, _Landmarks] = {
    REPO_BESSER: _Landmarks(
        layout=("besser/*", "besser/BUML/metamodel/*", "besser/BUML/notations/*", "besser/generators/*",
                "besser/spec_driven_agent/*", "besser/utilities/*", "besser/utilities/web_modeling_editor/backend/*",
                "besser/utilities/web_modeling_editor/backend/services/*", "tests/*", "docs/source/*"),
        files=(("Backend routers", "besser/utilities/web_modeling_editor/backend/routers/"),),
        registry="besser/generators/registry.py",
        converters=(_CONVERTERS + "json_to_buml/", _CONVERTERS + "buml_to_json/"),
    ),
    REPO_FRONTEND: _Landmarks(
        layout=("packages/*", "packages/*/src/main/*", "packages/editor/src/main/packages/*",
                "packages/webapp/src/main/features/*"),
    ),
    REPO_MODELING_AGENT: _Landmarks(
        layout=("src/*", "tests/*", "docs/source/*"),
        files=(("Diagram handlers", "src/diagram_handlers/types/"),),
    ),
}


# -- skills ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillEntry:
    """One skill of a repository, as the system prompt lists it."""

    repo: str
    name: str
    #: the ``SKILL.md``, relative to the tools' root (the folder that holds the repositories), ``/``-separated
    path: str
    when_to_read: str


class _SkillError(ValueError):
    pass


def _ordered(keys) -> list[str]:
    return sorted(keys, key=lambda key: (REPO_ORDER.index(key) if key in REPO_ORDER else len(REPO_ORDER), key))


def _read_skill(path: str) -> tuple[str, str]:
    """``(name, when_to_read)`` from a ``SKILL.md``'s YAML frontmatter; :class:`_SkillError` says what is wrong."""
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in (line.strip() for line in lines[1:]):
        raise _SkillError("no YAML frontmatter between --- lines")
    end = next(n for n, line in enumerate(lines[1:], 1) if line.strip() == "---")
    try:
        front = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise _SkillError(f"the frontmatter is not valid YAML ({exc.__class__.__name__})") from None
    if not isinstance(front, dict) or not isinstance(front.get("name"), str) or not front["name"].strip():
        raise _SkillError("the frontmatter has no name")
    metadata = front.get("metadata")
    line = metadata.get("when_to_read") if isinstance(metadata, dict) else None
    if not isinstance(line, str) or not line.strip():
        raise _SkillError("the frontmatter has no metadata.when_to_read")
    line = line.strip()
    if "\n" in line or len(line) > _MAX_WHEN_TO_READ_CHARS:
        raise _SkillError(f"metadata.when_to_read must be one line of at most {_MAX_WHEN_TO_READ_CHARS} characters")
    return front["name"].strip(), line


def skill_index(repos: Mapping[str, RepoSpec]) -> list[SkillEntry]:
    """The skills in ``<repo>/.claude/skills/*/SKILL.md`` of every repository's working tree.

    A skill whose frontmatter is malformed or has no one-line ``metadata.when_to_read`` is logged
    at warning level and left out.
    """
    if not repos:
        return []
    # The editor's tree sits inside BESSER's, so the root is the common parent of every tree's parent.
    tools_root = os.path.commonpath([os.path.dirname(os.path.abspath(spec.path)) for spec in repos.values()])
    entries = []
    for key in _ordered(repos):
        skills_dir = os.path.join(repos[key].path, ".claude", "skills")
        if not os.path.isdir(skills_dir):
            continue
        for folder in sorted(os.listdir(skills_dir)):
            path = os.path.join(skills_dir, folder, "SKILL.md")
            if not os.path.isfile(path):
                continue
            try:
                name, line = _read_skill(path)
            except (_SkillError, OSError, UnicodeDecodeError) as exc:
                logger.warning("BESSER Lab: skill %s left out of the skill index: %s", path, exc)
                continue
            rel = os.path.relpath(os.path.abspath(path), tools_root).replace(os.sep, "/")
            entries.append(SkillEntry(key, name, rel, line))
    return entries


def format_skill_index(entries: Sequence[SkillEntry]) -> str:
    """The skill index for the system prompt, at most 120 words; "" without skills.

    Skills past the cap are counted in a last line (and logged), not dropped silently.
    """
    if not entries:
        return ""
    lines = ["Skills (read a skill's file with read_file when its line fits the task):"]
    words = len(lines[0].split())
    for position, entry in enumerate(entries):
        line = f"- {entry.name} (`{entry.path}`): {entry.when_to_read}"
        left = len(entries) - position
        note = f"({left} more skills not listed; each is in its repository's .claude/skills/.)"
        reserve = len(note.split()) if left > 1 else 0
        if words + len(line.split()) + reserve > _MAX_SKILL_INDEX_WORDS:
            lines.append(note)
            logger.warning("BESSER Lab: %d skills left out of the skill index (%d-word cap)", left,
                           _MAX_SKILL_INDEX_WORDS)
            break
        lines.append(line)
        words += len(line.split())
    return "\n".join(lines)


# -- the map of a repository -----------------------------------------------------------------------------------


def _resolve(workspace: LabWorkspace, repo: str, base_sha: str) -> str:
    """The full SHA of ``base_sha`` in ``repo``; only a hex SHA is accepted (it reaches git's argv)."""
    if not isinstance(base_sha, str) or not _SHA_RE.match(base_sha):
        raise ValueError(f"{base_sha!r} is not a commit SHA")
    return workspace.run_git(repo, "rev-parse", "--verify", "--quiet", f"{base_sha}^{{commit}}").strip()


def _blob(workspace: LabWorkspace, repo: str, sha: str, path: str) -> str:
    return workspace.run_git(repo, "cat-file", "blob", f"{sha}:{path}")


def _section_index(text: str) -> list[str]:
    """The ``#`` to ``###`` headings of a Markdown text with their line ranges, code fences skipped."""
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    fence = ""
    for number, line in enumerate(lines, 1):
        marker = line.lstrip()[:3]
        if marker in ("```", "~~~"):
            fence = "" if fence == marker else fence or marker
            continue
        match = None if fence else _HEADING_RE.match(line)
        if match:
            headings.append((number, len(match.group(1)), match.group(2)))
    index = []
    for position, (start, level, title) in enumerate(headings):
        end = next((later - 1 for later, later_level, _ in headings[position + 1:] if later_level <= level),
                   len(lines))
        while end > start and not lines[end - 1].strip():
            end -= 1
        index.append(f"{'  ' * (level - 1)}- {title} (L{start}-{end})")
    return index


def _layout_line(files: Sequence[str], pattern: str) -> Optional[str]:
    """``pattern: a (n), b (m)``: the folders the pattern's last ``*`` matches, with the files under each."""
    segments = pattern.split("/")
    fixed = segments.index("*")
    counts: Counter[str] = Counter()
    for path in files:
        parts = path.split("/")
        if len(parts) > len(segments) and all(fnmatch.fnmatchcase(part, segment)
                                              for part, segment in zip(parts, segments)):
            counts["/".join(parts[fixed:len(segments)])] += 1
    if not counts:
        return None
    return f"`{pattern}`: " + ", ".join(f"{name} ({count})" for name, count in sorted(counts.items()))


def _files_in(files: Sequence[str], folder: str) -> list[str]:
    names = (path[len(folder):] for path in files if path.startswith(folder))
    return sorted(name for name in names if "/" not in name and not name.startswith("_"))


def _generators(workspace: LabWorkspace, repo: str, sha: str, files: Sequence[str], registry: str) -> list[str]:
    try:
        module = ast.parse(_blob(workspace, repo, sha, registry)) if registry in files else None
    except SyntaxError:
        module = None
    entries = _registry_entries(module) if module is not None else []
    if not entries:
        logger.warning("BESSER Lab: the map of %s at %s lists no generator: %s is missing or does not parse",
                       repo, sha[:_SHORT_SHA], registry)
        return [f"{registry} is missing or does not parse at this commit."]
    return [f"- {entry.id}: {entry.module}" + (f" (agent tool {entry.tool})" if entry.tool else "")
            for entry in sorted(entries, key=lambda entry: entry.id)]


def _converter_pairs(files: Sequence[str], to_buml: str, to_json: str) -> list[str]:
    processors = {name[:-len("_processor.py")]: name for name in _files_in(files, to_buml)
                  if name.endswith("_processor.py")}
    converters = {name[:-len("_converter.py")]: name for name in _files_in(files, to_json)
                  if name.endswith("_converter.py")}
    lines = []
    if both := sorted(processors.keys() & converters.keys()):
        lines.append("both ways: " + ", ".join(both))
    if only := sorted(processors.keys() - converters.keys()):
        lines.append("only json_to_buml: " + ", ".join(processors[key] for key in only))
    if only := sorted(converters.keys() - processors.keys()):
        lines.append("only buml_to_json: " + ", ".join(converters[key] for key in only))
    return lines


def _build_map(workspace: LabWorkspace, spec: RepoSpec, sha: str) -> str:
    repo = spec.key
    files = [path for path in workspace.run_git(repo, "ls-tree", "-r", "--name-only", "-z", sha).split("\0") if path]
    log = workspace.run_git(repo, "-c", "log.showSignature=false", "log", "--no-color", f"-{_RECENT_COMMITS}",
                            "--format=%H%x09%cs%x09%s", sha, "--")
    commits = [line.split("\t", 2) for line in log.splitlines() if line.count("\t") >= 2]
    date = commits[0][1] if commits else "unknown date"
    title = _REPO_TITLES.get(repo, repo)
    blocks = [f"### {spec.github_owner}/{spec.github_name}@{spec.base_branch}, base {sha[:_SHORT_SHA]} ({date}): "
              f"the {title} repository (`{repo}`)\nPaths below are relative to this repository's folder."]
    if "CLAUDE.md" in files:
        blocks.append("\n".join([_SECTIONS_HEADING, *_section_index(_blob(workspace, repo, sha, "CLAUDE.md"))]))
    else:
        blocks.append(f"{_SECTIONS_HEADING}\nno CLAUDE.md at this commit")
    landmarks = _LANDMARKS.get(repo, _Landmarks(layout=()))
    layout = [line for line in (_layout_line(files, pattern) for pattern in landmarks.layout) if line]
    if layout:
        blocks.append("\n".join(["#### Layout (folders with their file counts)", *layout]))
    if landmarks.registry:
        blocks.append("\n".join([f"#### Generators (`{landmarks.registry}`: id, module)",
                                 *_generators(workspace, repo, sha, files, landmarks.registry)]))
    if landmarks.converters:
        pairs = _converter_pairs(files, *landmarks.converters)
        if pairs:
            blocks.append("\n".join(["#### Editor converters by diagram (`json_to_buml/<x>_processor.py`, "
                                     "`buml_to_json/<x>_converter.py`)", *pairs]))
    for heading, folder in landmarks.files:
        if names := _files_in(files, folder):
            blocks.append(f"#### {heading} (`{folder}`)\n" + ", ".join(names))
    blocks.append("\n".join([f"#### Recent commits on {spec.base_branch}",
                             *(f"- {full[:_SHORT_SHA]} {day} {subject}" for full, day, subject in commits)]))
    return _cut("\n\n".join(blocks), MAP_MAX_CHARS, f"map of {repo} at {sha[:_SHORT_SHA]}", "map")


def _cut(text: str, cap: int, what: str, label: str) -> str:
    """``text`` cut at a line boundary to fit ``cap`` with a closing note; logged when it is cut."""
    if len(text) <= cap:
        return text
    note = f"\n[{label} cut at {cap:,} characters]"
    head = text[:cap - len(note)]
    head = head[:head.rfind("\n")] if "\n" in head else head
    logger.warning("BESSER Lab: the %s is %d characters, cut at %d", what, len(text), cap)
    return head + note


def _cache_files(folder: str, repo: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(repo)}-[0-9a-f]{{{_CACHE_SHA}}}\.md$")
    return [os.path.join(folder, name) for name in os.listdir(folder) if pattern.match(name)]


def _store(folder: str, repo: str, path: str, text: str) -> None:
    """Write the map atomically, then keep only the newest maps of ``repo``."""
    os.makedirs(folder, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=os.path.basename(path), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    newest_first = sorted(_cache_files(folder, repo), key=lambda name: (os.path.getmtime(name), name), reverse=True)
    for stale in newest_first[_MAPS_KEPT_PER_REPO:]:
        try:
            os.remove(stale)
        except OSError as exc:  # another thread removed or is reading it; the next build tries again
            logger.warning("BESSER Lab: old map %s not removed: %s", stale, exc)


def repo_map(workspace: LabWorkspace, repo: str, base_sha: str) -> str:
    """The map of ``repo`` at commit ``base_sha``: header, ``CLAUDE.md`` sections, layout, landmarks, commits.

    Built from ``git ls-tree`` and ``git log`` of that commit only, at most :data:`MAP_MAX_CHARS`,
    and cached in ``<state_dir>/maps/<repo>-<sha12>.md``; the two most recently used maps of
    each repository are kept. Raises ``ValueError`` for anything but a hex SHA and
    :class:`LabWorkspaceError` for an unknown repository or commit.
    """
    spec = workspace.base_repos.get(repo)
    if spec is None:
        raise LabWorkspaceError(f"unknown repository {repo!r}")
    folder = os.path.join(workspace.state_dir, "maps")
    # A full SHA finds its cached map without asking git; anything else is resolved to one first.
    known = isinstance(base_sha, str) and _FULL_SHA_RE.match(base_sha) is not None
    sha = base_sha if known else _resolve(workspace, repo, base_sha)
    path = os.path.join(folder, f"{repo}-{sha[:_CACHE_SHA]}.md")
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        os.utime(path)  # most recently used: kept by the next eviction
        return text
    except FileNotFoundError:
        pass
    if known:
        sha = _resolve(workspace, repo, sha)  # the commit exists (git refuses a tree or a missing object)
    text = _build_map(workspace, spec, sha)
    _store(folder, repo, path, text)
    return text


# -- the orientation -------------------------------------------------------------------------------------------


def _with_full_claude_md(repo_map_text: str, claude_md: str) -> str:
    """The map with its section index replaced by the full ``CLAUDE.md``, right after the header."""
    header, *blocks = repo_map_text.split("\n\n")
    rest = [block for block in blocks if not block.startswith(_SECTIONS_HEADING)]
    return "\n\n".join([header, "#### CLAUDE.md (full text at the base commit)\n" + claude_md.strip(), *rest])


def _with_note(repo_map_text: str, note: str) -> str:
    header, _, rest = repo_map_text.partition("\n\n")
    return f"{header}\n{note}\n\n{rest}"


def orientation(workspace: LabWorkspace, *, inline: Sequence[str], indexed: Sequence[str],
                base_shas: Mapping[str, str]) -> str:
    """The orientation for a step's first message: per repository its ``CLAUDE.md`` and its map.

    ``inline`` repositories get their full ``CLAUDE.md``, ``indexed`` ones its section index
    (a repository in both is inlined); ``base_shas`` names the base commit of each. Over
    :data:`ORIENTATION_MAX_CHARS`, the last inlined ``CLAUDE.md`` falls back to its index, with a
    note saying so, until it fits; what is still over is cut. Raises ``ValueError`` when a
    repository has no base commit.
    """
    order = list(dict.fromkeys([*inline, *indexed]))
    missing = [repo for repo in order if repo not in base_shas]
    if missing:
        raise ValueError(f"no base commit given for {', '.join(missing)}")
    maps = {repo: repo_map(workspace, repo, base_shas[repo]) for repo in order}
    full: dict[str, str] = {}
    for repo in dict.fromkeys(inline):
        sha = _resolve(workspace, repo, base_shas[repo])
        if "CLAUDE.md" in workspace.run_git(repo, "ls-tree", "--name-only", sha, "--", "CLAUDE.md").split():
            full[repo] = _blob(workspace, repo, sha, "CLAUDE.md")
    demoted: list[str] = []
    while True:
        parts = [_ORIENTATION_HEADING]
        for repo in order:
            if repo in full and repo not in demoted:
                parts.append(_with_full_claude_md(maps[repo], full[repo]))
            elif repo in demoted:
                parts.append(_with_note(maps[repo], "Its full CLAUDE.md did not fit in the orientation: read the "
                                                    "sections you need from the index below."))
            else:
                parts.append(maps[repo])
        text = "\n\n".join(parts)
        remaining = [repo for repo in full if repo not in demoted]
        if len(text) <= ORIENTATION_MAX_CHARS or not remaining:
            break
        demoted.append(remaining[-1])
        logger.warning("BESSER Lab: the CLAUDE.md of %s did not fit in the orientation (%d characters, cap %d); "
                       "given as a section index", remaining[-1], len(text), ORIENTATION_MAX_CHARS)
    return _cut(text, ORIENTATION_MAX_CHARS, "orientation", "orientation")
