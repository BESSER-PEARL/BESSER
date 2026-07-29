"""The conversion must be byte-for-byte reproducible.

Generators in this repo are expected to produce identical output for identical
input. That is not free here: the pipeline reads rdflib result *sets*, and the
reference implementation this port is based on was measurably nondeterministic
— three runs over the same input produced three different models, differing in
which member of an ``owl:inverseOf`` pair became the association's source role.
BIBO declares two such pairs (``cites``/``citedBy`` and
``presents``/``presentedAt``), so the fixture still reproduces the original
failure mode.

Set iteration order depends on ``PYTHONHASHSEED``, which is fixed for the life
of a process, so re-running in-process proves nothing. These tests fork
subprocesses with different seeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "bibo"
REPO_ROOT = Path(__file__).resolve().parents[4]

_SCRIPT = textwrap.dedent(
    """
    import hashlib, json, sys
    from rdflib import Graph
    from besser.utilities.owl_to_buml import owl_file_to_knowledge_graph
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram
    from besser.BUML.metamodel.structural import Class

    schema, shapes, combined = sys.argv[1], sys.argv[2], sys.argv[3]
    graph = Graph()
    graph.parse(schema, format="turtle")
    graph.parse(shapes, format="turtle")
    graph.serialize(destination=combined, format="turtle")

    model = kg_to_class_diagram(
        owl_file_to_knowledge_graph(combined), model_name="Bibo"
    ).domain_model
    payload = {
        "classes": sorted(t.name for t in model.types if isinstance(t, Class)),
        "generalizations": sorted(
            [g.specific.name, g.general.name] for g in model.generalizations
        ),
        "associations": sorted(
            [a.name] + sorted(f"{e.name}:{e.type.name}" for e in a.ends)
            for a in model.associations
        ),
        "constraints": sorted(c.expression for c in model.constraints),
    }
    print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    """
)


def _digest_with_seed(seed: str, tmp_path: Path) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    combined = tmp_path / f"combined-{seed}.ttl"
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(FIXTURES / "bibo.ttl"),
         str(FIXTURES / "bibo-shapes.ttl"), str(combined)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_conversion_is_independent_of_hash_seed(tmp_path: Path):
    digests = {seed: _digest_with_seed(seed, tmp_path) for seed in ("0", "1", "12345")}
    assert len(set(digests.values())) == 1, (
        "conversion output varies with PYTHONHASHSEED — an rdflib result set is "
        f"being iterated unsorted somewhere: {digests}"
    )
