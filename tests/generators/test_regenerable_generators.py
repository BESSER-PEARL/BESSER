"""A generator may opt in to ``regenerate()`` only with output that is stable for one model.

Unstable output would read as a model change on every run. Each run below is a fresh
interpreter with a different ``PYTHONHASHSEED``; ``set`` order is what usually differs.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pygit2")

from besser.generators.registry import GENERATORS
from tests.generators.regenerable_builders import BUILDERS

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("regenerable_builders.py")
SEEDS = ("0", "1")
# Excluded on purpose: output that differs per run, or an LLM step.
NOT_REGENERABLE = {
    "terraform": "its templates loop over sets of regions, services and deployments",
    "supabase": "a timestamp in the migration's file name",
    "flutter": "sql_helper.dart's order follows the hash seed",
    "agent": "personalization calls an LLM and draws random ids",
    "docker_compose": "personalization calls an LLM and draws random ids",
}


def _regenerate_all(top: Path, seed: str) -> tuple[dict, dict[str, bytes]]:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO)}
    result = subprocess.run([sys.executable, str(SCRIPT), str(top)], env=env, cwd=REPO,
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        pytest.fail(f"A generator failed to build or regenerate:\n{result.stderr[-3000:]}")
    files = {str(p.relative_to(top)): p.read_bytes() for p in sorted(top.rglob("*")) if p.is_file()}
    return json.loads(result.stdout.splitlines()[-1]), files


def _without_secret(files: dict[str, bytes]) -> dict[str, bytes]:
    """Django's ``SECRET_KEY`` is stable in one folder, and random between two new ones."""
    return {name: re.sub(rb"(?m)^SECRET_KEY = .*$", b"SECRET_KEY = ...", content)
            if name.endswith("settings.py") else content for name, content in files.items()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return {seed: _regenerate_all(tmp_path_factory.mktemp(f"seed{seed}"), seed) for seed in SEEDS}


def test_every_generator_is_either_covered_here_or_excluded_with_a_reason():
    ids = {spec.id for spec in GENERATORS}
    assert set(NOT_REGENERABLE) | {*BUILDERS, "django"} == ids


@pytest.mark.parametrize("spec", GENERATORS, ids=lambda spec: spec.id)
def test_regenerable_matches_the_list_above(spec):
    try:
        generator_class = spec.load_class()
    except ImportError as exc:
        pytest.skip(f"{spec.id} is not installed: {exc}")
    assert generator_class.regenerable is (spec.id not in NOT_REGENERABLE)


@pytest.mark.skipif(importlib.util.find_spec("django") is None, reason="Django is not covered: needs Django")
def test_django_is_among_the_generators_checked(runs):
    assert "django" in runs[SEEDS[0]][0]


def test_a_second_run_with_the_same_model_changes_nothing(runs):
    reports, _ = runs[SEEDS[0]]
    changed = {gid: report for gid, report in reports.items()
               if any(report[key] for key in ("written", "merged", "conflicted", "deleted",
                                              "kept_removed_from_model", "not_restored", "not_merged"))}
    assert not changed


def test_the_output_is_identical_across_hash_seeds(runs):
    first, second = (_without_secret(runs[seed][1]) for seed in SEEDS)
    assert first.keys() == second.keys()
    assert not [name for name in first if first[name] != second[name]]
