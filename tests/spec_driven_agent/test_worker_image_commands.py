"""What the run's shell offers: ``ps`` is installed, ``git`` is not and the tool says so.

Without ``ps`` the agent cannot see which servers it left running; without a
word about ``git`` it spends turns on ``git status`` / ``git diff`` in a folder
that is not a repository.
"""
from __future__ import annotations

import re
from pathlib import Path

from besser.spec_driven_agent.agent.tools import EXECUTION_TOOLS


def _worker_apt_packages() -> set[str]:
    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text(encoding="utf-8")
    worker = dockerfile.split("FROM python-deps AS smartgen-worker", 1)[1]
    packages: set[str] = set()
    for block in re.findall(r"apt-get install -y --no-install-recommends((?:[^&]|&(?!&))*)", worker):
        packages |= {word for word in block.replace("\\", " ").split() if not word.startswith("-")}
    return packages


def test_the_worker_image_installs_ps():
    assert "procps" in _worker_apt_packages()


def test_the_worker_image_does_not_install_git():
    assert "git" not in _worker_apt_packages()


def test_run_command_says_git_is_unavailable_and_the_folder_is_not_a_repository():
    description = next(t for t in EXECUTION_TOOLS if t["name"] == "run_command")["description"]
    assert "git is not installed" in description
    assert "not a git repository" in description
