"""Dockerfile auto-fixes must reach multi-service layouts.

Live 2026-09-11: a generated app had `Dockerfile.frontend` / `Dockerfile.backend`
at the project root with `package.json` under `frontend/`. `docker compose build`
failed twice over, and the user burned FIVE fix runs on it — the last one
mis-diagnosed the problem entirely, declaring `npm ci --silent --no-optional`
"good" and spending 3m40s improving unrelated React error handling.

Both failures were already guarded against in Phase-3 validation. Neither guard
ran, because the guard block was gated on `fname == "Dockerfile"` exactly.
"""

import os

import pytest

from besser.generators.llm.orchestrator import (
    _is_dockerfile,
    _project_has_npm_lockfile,
    _strip_missing_lockfile_copy,
)


# ------------------------------------------------------- which files count


@pytest.mark.parametrize("name", [
    "Dockerfile",
    "dockerfile",
    "Dockerfile.frontend",   # the layout that slipped through
    "Dockerfile.backend",
    "Dockerfile.dev",
    "frontend.Dockerfile",
])
def test_dockerfile_naming_conventions_are_all_recognised(name):
    assert _is_dockerfile(name) is True


@pytest.mark.parametrize("name", [
    "README.md", "docker-compose.yml", "Dockerfileish.txt", "package.json",
])
def test_unrelated_files_are_not_treated_as_dockerfiles(name):
    assert _is_dockerfile(name) is False


# ------------------------------------------------- finding the lockfile


def test_lockfile_is_found_anywhere_in_the_project(tmp_path):
    """A root Dockerfile.frontend COPYs from frontend/, so looking only beside
    the Dockerfile found nothing even when a lockfile existed."""
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package-lock.json").write_text("{}", encoding="utf-8")
    assert _project_has_npm_lockfile(str(tmp_path)) is True


def test_shrinkwrap_also_counts(tmp_path):
    (tmp_path / "npm-shrinkwrap.json").write_text("{}", encoding="utf-8")
    assert _project_has_npm_lockfile(str(tmp_path)) is True


def test_no_lockfile_anywhere_is_detected(tmp_path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    assert _project_has_npm_lockfile(str(tmp_path)) is False


def test_node_modules_is_not_searched(tmp_path):
    """A lockfile belonging to a dependency is not the project's lockfile, and
    walking node_modules is slow enough to matter."""
    nm = tmp_path / "node_modules" / "some-dep"
    nm.mkdir(parents=True)
    (nm / "package-lock.json").write_text("{}", encoding="utf-8")
    assert _project_has_npm_lockfile(str(tmp_path)) is False


# --------------------------------------------- stripping the bad COPY


def test_the_exact_failing_line_is_repaired():
    """Verbatim from the user's build:
        COPY frontend/package.json frontend/package-lock.json ./
        -> "/frontend/package-lock.json": not found
    """
    out = _strip_missing_lockfile_copy(
        "COPY frontend/package.json frontend/package-lock.json ./"
    )
    assert out.strip() == "COPY frontend/package.json ./"
    assert "package-lock" not in out


def test_a_lockfile_only_copy_line_is_dropped_entirely():
    assert _strip_missing_lockfile_copy("COPY package-lock.json ./").strip() == ""


def test_a_glob_copy_is_left_alone():
    """``COPY package*.json ./`` is valid with or without a lockfile — Docker
    matches what exists. Rewriting it would be a regression."""
    line = "COPY package*.json ./"
    assert _strip_missing_lockfile_copy(line) == line


def test_indentation_is_preserved():
    out = _strip_missing_lockfile_copy(
        "    COPY frontend/package.json frontend/package-lock.json ./"
    )
    assert out.startswith("    COPY frontend/package.json")


def test_non_copy_lines_are_untouched():
    line = "RUN npm ci  # needs package-lock.json"
    assert _strip_missing_lockfile_copy(line) == line


def test_other_dockerfile_content_survives():
    content = (
        "FROM node:16-alpine\n"
        "WORKDIR /app\n"
        "COPY frontend/package.json frontend/package-lock.json ./\n"
        "RUN npm ci\n"
        "COPY frontend/ ./\n"
        'CMD ["npm", "run", "dev"]\n'
    )
    out = _strip_missing_lockfile_copy(content)
    assert "FROM node:16-alpine" in out
    assert "WORKDIR /app" in out
    assert "COPY frontend/package.json ./" in out
    assert "package-lock" not in out
    assert "RUN npm ci" in out            # fixed separately, by the npm-ci guard
    assert 'CMD ["npm", "run", "dev"]' in out
    assert out.endswith("\n")
