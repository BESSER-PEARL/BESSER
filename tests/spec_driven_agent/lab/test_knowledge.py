"""The lab's orientation: the skill index, the repository map built from a base commit, and their assembly."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import pytest

from besser.spec_driven_agent.lab import knowledge
from besser.spec_driven_agent.lab.contracts import REPO_BESSER, REPO_MODELING_AGENT, RepoSpec
from besser.spec_driven_agent.lab.knowledge import (
    MAP_MAX_CHARS,
    ORIENTATION_MAX_CHARS,
    SkillEntry,
    format_skill_index,
    orientation,
    repo_map,
    skill_index,
)
from besser.spec_driven_agent.lab.workspace import LabWorkspace, LabWorkspaceError
from tests.spec_driven_agent.lab.conftest import git, write

# This repository's own checkout: the map of a real BESSER is built from it, read-only.
_CHECKOUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_BESSER_CLAUDE = """# CLAUDE.md

Intro line.

## Essential Commands

```bash
# not a heading: inside a fence
python -m pytest tests/
```

### Testing
Run the tests.

## Where Things Live
The table.
"""

_REGISTRY = '''GENERATORS = (
    GeneratorSpec(id="django", class_path="besser.generators.django:DjangoGenerator", output_type="zip"),
    GeneratorSpec(id="python", class_path="besser.generators.python_classes:PythonGenerator",
                  agent_tool=AgentTool(name="generate_python", label="Python")),
)
'''

_CONVERTERS = "besser/utilities/web_modeling_editor/backend/services/converters/"


def _commit(path: str, message: str) -> str:
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD")


def _tiny_besser(path: str) -> str:
    """A small BESSER-shaped repository; returns its HEAD."""
    files = {
        "CLAUDE.md": _BESSER_CLAUDE,
        "besser/generators/registry.py": _REGISTRY,
        "besser/generators/django/__init__.py": "",
        "besser/generators/django/django_generator.py": "x = 1\n",
        "besser/generators/python_classes/__init__.py": "",
        "besser/BUML/metamodel/structural/structural.py": "",
        _CONVERTERS + "json_to_buml/class_diagram_processor.py": "",
        _CONVERTERS + "json_to_buml/utils.py": "",
        _CONVERTERS + "buml_to_json/class_diagram_converter.py": "",
        _CONVERTERS + "buml_to_json/kg_diagram_converter.py": "",
        "besser/utilities/web_modeling_editor/backend/routers/generation_router.py": "",
        "besser/utilities/web_modeling_editor/backend/routers/__init__.py": "",
        "tests/generators/django/test_django.py": "",
        "docs/source/generators/django.rst": "",
    }
    for rel, text in files.items():
        write(os.path.join(path, *rel.split("/")), text)
    git(path, "init", "-q", "-b", "development")
    return _commit(path, "feat: first commit (#12)")


def _tiny_agent(path: str) -> str:
    # A long body line, so indexing this CLAUDE.md instead of inlining it saves room.
    write(os.path.join(path, "CLAUDE.md"),
          "# Agent\n\nAGENT-BODY-LINE " + "filler " * 300 + "\n\n## Adding (or Auditing) a Diagram Type\nSteps.\n")
    write(os.path.join(path, "src", "diagram_handlers", "types", "class_diagram_handler.py"), "")
    write(os.path.join(path, "src", "routing", "guards.py"), "")
    git(path, "init", "-q", "-b", "develop")
    return _commit(path, "feat: agent")


def _spec(key: str, path: str, branch: str, name: str) -> RepoSpec:
    return RepoSpec(key, path, branch, "BESSER-PEARL", name, os.path.join(path, ".git"))


@pytest.fixture
def besser_repo(tmp_path):
    path = str(tmp_path / "trees" / "besser")
    sha = _tiny_besser(path)
    return path, sha


def _workspace(tmp_path, *specs: RepoSpec, state: str = "state") -> LabWorkspace:
    return LabWorkspace(str(tmp_path / "work"), specs, state_dir=str(tmp_path / state), git_root=str(tmp_path / "git"))


def _besser_workspace(tmp_path, path: str, state: str = "state") -> LabWorkspace:
    return _workspace(tmp_path, _spec(REPO_BESSER, path, "development", "BESSER"), state=state)


# -- repo_map ------------------------------------------------------------------------------------------------


def test_map_holds_header_sections_layout_landmarks_and_commits(tmp_path, besser_repo):
    path, sha = besser_repo
    text = repo_map(_besser_workspace(tmp_path, path), REPO_BESSER, sha)

    assert f"BESSER-PEARL/BESSER@development, base {sha[:9]}" in text
    assert time.strftime("%Y") in text  # the base commit's date
    # The CLAUDE.md section index, with line ranges, headings inside a code fence left out.
    assert "Essential Commands (L5-13)" in text
    assert "Testing (L12-13)" in text
    assert "Where Things Live (L15-16)" in text
    assert "not a heading" not in text
    # Layout: declared source roots with file counts.
    assert "django (2)" in text and "python_classes (1)" in text and "structural (1)" in text
    # Generators from the registry, converters paired by diagram, routers.
    assert "- django: besser.generators.django" in text
    assert "- python: besser.generators.python_classes (agent tool generate_python)" in text
    assert "class_diagram" in text
    assert "only buml_to_json: kg_diagram_converter.py" in text
    assert "generation_router.py" in text and "__init__.py" not in text.split("routers")[-1].split("####")[0]
    assert f"{sha[:9]} " in text and "feat: first commit (#12)" in text


def test_map_is_cached_under_the_state_folder_and_reused(tmp_path, besser_repo, monkeypatch):
    path, sha = besser_repo
    workspace = _besser_workspace(tmp_path, path)
    first = repo_map(workspace, REPO_BESSER, sha)
    cached = tmp_path / "state" / "maps" / f"besser-{sha[:12]}.md"
    assert cached.read_text(encoding="utf-8") == first

    def no_git(*args, **kwargs):
        raise AssertionError("a cached map runs no git")

    monkeypatch.setattr(workspace, "run_git", no_git)
    assert repo_map(workspace, REPO_BESSER, sha) == first


def test_map_is_byte_identical_and_ignores_untracked_files_and_node_modules(tmp_path, besser_repo):
    path, sha = besser_repo
    first = repo_map(_besser_workspace(tmp_path, path, "state-1"), REPO_BESSER, sha)
    write(os.path.join(path, "besser", "generators", "scratch.py"), "untracked\n")
    write(os.path.join(path, "node_modules", "left-pad", "index.js"), "module.exports = 1\n")
    write(os.path.join(path, "besser", "generators", "django", "new_file.py"), "")

    second = repo_map(_besser_workspace(tmp_path, path, "state-2"), REPO_BESSER, sha)

    assert second == first
    assert "node_modules" not in second and "scratch.py" not in second


def test_a_moved_base_rebuilds_the_map_and_keeps_the_two_newest(tmp_path, besser_repo):
    path, first_sha = besser_repo
    workspace = _besser_workspace(tmp_path, path)
    first = repo_map(workspace, REPO_BESSER, first_sha)
    write(os.path.join(path, "besser", "generators", "sql", "__init__.py"), "")
    second_sha = _commit(path, "feat: add the sql generator")
    time.sleep(0.05)  # distinct modification times on coarse file systems
    second = repo_map(workspace, REPO_BESSER, second_sha)
    write(os.path.join(path, "docs", "source", "more.rst"), "")
    third_sha = _commit(path, "docs: more")
    time.sleep(0.05)
    repo_map(workspace, REPO_BESSER, third_sha)

    assert second != first
    assert "feat: add the sql generator" in second and "sql (1)" in second
    kept = sorted(os.listdir(tmp_path / "state" / "maps"))
    assert kept == sorted([f"besser-{second_sha[:12]}.md", f"besser-{third_sha[:12]}.md"])


def test_map_accepts_a_short_sha_and_caches_it_under_the_full_one(tmp_path, besser_repo):
    path, sha = besser_repo
    repo_map(_besser_workspace(tmp_path, path), REPO_BESSER, sha[:7])
    assert (tmp_path / "state" / "maps" / f"besser-{sha[:12]}.md").is_file()


@pytest.mark.parametrize("bad", ["--output=x", "HEAD", "../x", "zzzzzzz", ""])
def test_map_refuses_anything_but_a_commit_sha(tmp_path, besser_repo, bad):
    path, _sha = besser_repo
    with pytest.raises(ValueError):
        repo_map(_besser_workspace(tmp_path, path), REPO_BESSER, bad)


def test_map_refuses_an_unknown_repository(tmp_path, besser_repo):
    path, sha = besser_repo
    with pytest.raises(LabWorkspaceError):
        repo_map(_besser_workspace(tmp_path, path), "elsewhere", sha)


def test_map_over_its_cap_is_cut_with_a_note(tmp_path, besser_repo, monkeypatch, caplog):
    path, sha = besser_repo
    for n in range(60):
        write(os.path.join(path, "besser", "generators", f"generator_number_{n:02d}", "__init__.py"), "")
    sha = _commit(path, "feat: many generators")
    monkeypatch.setattr(knowledge, "MAP_MAX_CHARS", 1_500)

    with caplog.at_level(logging.WARNING):
        text = repo_map(_besser_workspace(tmp_path, path), REPO_BESSER, sha)

    assert len(text) <= 1_500
    assert "map cut at 1,500 characters" in text
    assert "cut" in caplog.text


def test_map_of_a_repository_without_claude_md_or_registry_says_so(tmp_path, caplog):
    path = str(tmp_path / "trees" / "besser")
    write(os.path.join(path, "README.md"), "hi\n")
    git(path, "init", "-q", "-b", "development")
    sha = _commit(path, "init")
    with caplog.at_level(logging.WARNING):
        text = repo_map(_besser_workspace(tmp_path, path), REPO_BESSER, sha)
    assert "no CLAUDE.md at this commit" in text
    assert "besser/generators/registry.py is missing or does not parse" in text
    assert "registry.py" in caplog.text


def test_map_of_the_agent_lists_its_diagram_handlers(tmp_path):
    path = str(tmp_path / "trees" / "modeling-agent")
    sha = _tiny_agent(path)
    workspace = _workspace(tmp_path, _spec(REPO_MODELING_AGENT, path, "develop", "modeling-agent"))
    text = repo_map(workspace, REPO_MODELING_AGENT, sha)
    assert "BESSER-PEARL/modeling-agent@develop" in text
    assert "class_diagram_handler.py" in text
    assert "routing (1)" in text


def _checkout_spec() -> RepoSpec:
    branch = git(_CHECKOUT, "rev-parse", "--abbrev-ref", "HEAD")
    return RepoSpec(REPO_BESSER, _CHECKOUT, branch, "BESSER-PEARL", "BESSER", os.path.join(_CHECKOUT, ".git"))


def _require_checkout() -> None:
    if shutil.which("git") is None or not os.path.exists(os.path.join(_CHECKOUT, ".git")):
        pytest.skip("not a git checkout of BESSER")
    try:
        git(_CHECKOUT, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout of BESSER")


def test_map_of_this_besser_checkout_is_fast_small_and_deterministic(tmp_path):
    _require_checkout()
    spec = _checkout_spec()
    sha = git(_CHECKOUT, "rev-parse", "HEAD")

    started = time.monotonic()
    first = repo_map(_workspace(tmp_path, spec, state="state-1"), REPO_BESSER, sha)
    elapsed = time.monotonic() - started
    second = repo_map(_workspace(tmp_path, spec, state="state-2"), REPO_BESSER, sha)

    assert elapsed < 5.0
    assert len(first) < MAP_MAX_CHARS
    assert second == first
    assert "map cut" not in first
    assert "Essential Commands" in first
    assert "- django: besser.generators.django" in first
    assert "class_diagram" in first and "generation_router.py" in first


# -- skill_index ---------------------------------------------------------------------------------------------


def _skill(root: str, name: str, frontmatter: str) -> None:
    write(os.path.join(root, ".claude", "skills", name, "SKILL.md"), f"---\n{frontmatter}---\n\n# {name}\n")


def test_skill_index_lists_good_skills_and_skips_malformed_ones_with_a_warning(tmp_path, caplog):
    besser = str(tmp_path / "base" / "besser")
    agent = str(tmp_path / "base" / "modeling-agent")
    _skill(besser, "good", 'name: good\ndescription: >\n  long trigger text\nmetadata:\n  when_to_read: "Adding X."\n')
    _skill(besser, "broken-yaml", "name: [unclosed\n")
    _skill(besser, "no-line", "name: no-line\ndescription: x\n")
    _skill(besser, "too-long", f"name: too-long\nmetadata:\n  when_to_read: \"{'a' * 161}\"\n")
    _skill(besser, "two-lines", 'name: two-lines\nmetadata:\n  when_to_read: "one\\ntwo"\n')
    write(os.path.join(besser, ".claude", "skills", "no-frontmatter", "SKILL.md"), "# just text\n")
    _skill(agent, "agent-skill", 'name: agent-skill\nmetadata:\n  when_to_read: "Changing routing."\n')
    repos = {REPO_BESSER: _spec(REPO_BESSER, besser, "development", "BESSER"),
             REPO_MODELING_AGENT: _spec(REPO_MODELING_AGENT, agent, "develop", "modeling-agent")}

    with caplog.at_level(logging.WARNING):
        entries = skill_index(repos)

    assert entries == [
        SkillEntry(REPO_BESSER, "good", "besser/.claude/skills/good/SKILL.md", "Adding X."),
        SkillEntry(REPO_MODELING_AGENT, "agent-skill", "modeling-agent/.claude/skills/agent-skill/SKILL.md",
                   "Changing routing."),
    ]
    for skipped in ("broken-yaml", "no-line", "too-long", "two-lines", "no-frontmatter"):
        assert skipped in caplog.text


def test_skill_index_of_a_repository_without_skills_is_empty(tmp_path):
    path = str(tmp_path / "base" / "besser")
    os.makedirs(path)
    assert skill_index({REPO_BESSER: _spec(REPO_BESSER, path, "development", "BESSER")}) == []


def test_the_contributor_skill_in_this_checkout_is_listed_with_its_when_to_read():
    skill = os.path.join(_CHECKOUT, ".claude", "skills", "besser-dev", "SKILL.md")
    if not os.path.isfile(skill):
        pytest.skip("this checkout has no .claude/skills/besser-dev")
    # The index reads the working tree; the git directory plays no part.
    entries = skill_index({REPO_BESSER: RepoSpec(REPO_BESSER, _CHECKOUT, "development", "BESSER-PEARL", "BESSER",
                                                 os.path.join(_CHECKOUT, ".git"))})
    besser_dev = [entry for entry in entries if entry.name == "besser-dev"]
    assert len(besser_dev) == 1
    assert besser_dev[0].when_to_read.startswith("Adding or changing a generator")
    assert besser_dev[0].path.endswith(".claude/skills/besser-dev/SKILL.md")


def test_format_skill_index_names_path_and_line_within_120_words(caplog):
    entries = [SkillEntry(REPO_BESSER, f"skill-{n}", f"besser/.claude/skills/skill-{n}/SKILL.md",
                          "Adding or changing a generator, a metamodel package, editor converters or docs.")
               for n in range(12)]
    with caplog.at_level(logging.WARNING):
        text = format_skill_index(entries)
    assert len(text.split()) <= 120
    assert "skill-0" in text and "besser/.claude/skills/skill-0/SKILL.md" in text
    assert "more skills not listed" in text
    assert "6 skills left out" in caplog.text
    assert format_skill_index([]) == ""
    one = format_skill_index(entries[:1])
    assert "Adding or changing a generator" in one and "not listed" not in one


# -- orientation ---------------------------------------------------------------------------------------------


@pytest.fixture
def two_repos(tmp_path):
    besser = str(tmp_path / "trees" / "besser")
    agent = str(tmp_path / "trees" / "modeling-agent")
    shas = {REPO_BESSER: _tiny_besser(besser), REPO_MODELING_AGENT: _tiny_agent(agent)}
    workspace = _workspace(tmp_path, _spec(REPO_BESSER, besser, "development", "BESSER"),
                           _spec(REPO_MODELING_AGENT, agent, "develop", "modeling-agent"))
    return workspace, shas


def test_orientation_inlines_claude_md_of_inline_and_indexes_the_others(two_repos):
    workspace, shas = two_repos
    text = orientation(workspace, inline=[REPO_BESSER], indexed=[REPO_MODELING_AGENT], base_shas=shas)

    assert text.startswith("## Orientation (from the base commits; your edits are not in it)")
    assert "Intro line." in text and "python -m pytest tests/" in text  # BESSER's CLAUDE.md in full
    assert "AGENT-BODY-LINE" not in text  # the agent's only as an index
    assert "Adding (or Auditing) a Diagram Type (L5-6)" in text
    assert "- django: besser.generators.django" in text and "class_diagram_handler.py" in text
    # BESSER's section index is not repeated next to its full text.
    assert "Essential Commands (L5-13)" not in text
    assert text.index("BESSER-PEARL/BESSER@") < text.index("BESSER-PEARL/modeling-agent@")


def test_orientation_over_its_cap_indexes_claude_md_instead_and_says_so(two_repos, monkeypatch, caplog):
    workspace, shas = two_repos
    full = orientation(workspace, inline=[REPO_BESSER, REPO_MODELING_AGENT], indexed=[], base_shas=shas)
    assert "AGENT-BODY-LINE" in full
    cap = len(full) - 20
    monkeypatch.setattr(knowledge, "ORIENTATION_MAX_CHARS", cap)

    with caplog.at_level(logging.WARNING):
        text = orientation(workspace, inline=[REPO_BESSER, REPO_MODELING_AGENT], indexed=[], base_shas=shas)

    assert len(text) <= cap
    assert "Intro line." in text  # the first repository keeps its full CLAUDE.md
    assert "AGENT-BODY-LINE" not in text
    assert "did not fit" in text and "Adding (or Auditing) a Diagram Type (L5-6)" in text
    assert "did not fit" in caplog.text
    assert ORIENTATION_MAX_CHARS == 60_000


def test_orientation_needs_a_base_commit_for_every_repository(two_repos):
    workspace, shas = two_repos
    with pytest.raises(ValueError, match="modeling_agent"):
        orientation(workspace, inline=[REPO_BESSER], indexed=[REPO_MODELING_AGENT],
                    base_shas={REPO_BESSER: shas[REPO_BESSER]})
