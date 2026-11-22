"""JSON Schema output must be valid JSON and identical across runs (#679).

The regular schema was hand-built JSON in a template: a class with
attributes and subclasses but no navigable association end lost the comma
before its first subclass, so the file did not parse. Both modes looped over
attribute, literal, association and end sets, whose iteration order changes
between interpreters. Each run is a fresh interpreter with its own
``PYTHONHASHSEED``.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

GENERATE = textwrap.dedent('''
    import os, sys
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, BooleanType, Class, DateType, DomainModel, Enumeration,
        EnumerationLiteral, FloatType, Generalization, IntegerType, Multiplicity,
        Property, StringType,
    )
    from besser.generators.json import JSONSchemaGenerator

    def end(name, cls, lo, hi):
        return Property(name=name, type=cls, multiplicity=Multiplicity(lo, hi))

    status = Enumeration(name="Status", literals={
        EnumerationLiteral(name=n) for n in ("OPEN", "CLOSED", "PENDING", "ARCHIVED", "DRAFT")})
    # Person has attributes and subclasses but no association end.
    person = Class(name="Person", attributes={
        Property(name=n, type=t) for n, t in (
            ("first_name", StringType), ("last_name", StringType), ("email", StringType),
            ("age", IntegerType), ("height", FloatType), ("active", BooleanType))})
    member = Class(name="Member", attributes={
        Property(name="joined", type=DateType), Property(name="level", type=IntegerType),
        Property(name="status", type=status),
        Property(name="tags", type=StringType, multiplicity=Multiplicity(0, "*"))})
    clerk = Class(name="Clerk", attributes={Property(name="desk", type=StringType)})
    library = Class(name="Library", attributes={
        Property(name=n, type=StringType) for n in ("name", "city", "street", "zip")})
    book = Class(name="Book", attributes={
        Property(name=n, type=StringType) for n in ("title", "isbn", "summary")})
    model = DomainModel(
        name="Lib", types={person, member, clerk, library, book, status},
        associations={
            BinaryAssociation(name="holds", ends={end("library", library, 1, 1), end("books", book, 0, "*")}),
            BinaryAssociation(name="visits", ends={end("libraries", library, 0, "*"), end("members", member, 0, "*")}),
            BinaryAssociation(name="manages", ends={end("manager", member, 0, 1), end("reports", member, 0, "*")}),
        },
        generalizations={Generalization(general=person, specific=member),
                         Generalization(general=person, specific=clerk)})
    JSONSchemaGenerator(model, output_dir=os.path.join(sys.argv[1], "regular")).generate()
    JSONSchemaGenerator(model, output_dir=os.path.join(sys.argv[1], "smart_data"), mode="smart_data").generate()
''')

SEEDS = ("0", "1", "2", "3")


def _generate(out: Path, seed: str) -> dict[str, bytes]:
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)}
    subprocess.run([sys.executable, "-c", GENERATE, str(out)], env=env, check=True, cwd=repo,
                   capture_output=True)
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_every_output_is_valid_json(tmp_path):
    files = _generate(tmp_path / "out", "0")
    assert "regular/json_schema.json" in files
    for name, data in files.items():
        if name.endswith((".json", ".jsonld")):
            json.loads(data)  # raises on invalid JSON, naming the position

    schema = json.loads(files["regular/json_schema.json"])
    person = schema["definitions"]["Person"]["allOf"][-1]["properties"]
    assert {"first_name", "Member", "Clerk"} <= person.keys()
    member = schema["definitions"]["Member"]["allOf"]
    assert member[0] == {"$ref": "#/definitions/Person"}
    assert member[-1]["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert member[-1]["properties"]["reports"] == {"type": "array", "items": {"$ref": "#/definitions/Member"}}
    assert schema["definitions"]["Status"]["enum"] == ["OPEN", "CLOSED", "PENDING", "ARCHIVED", "DRAFT"]


def test_json_schema_is_identical_across_hash_seeds(tmp_path):
    runs = {seed: _generate(tmp_path / seed, seed) for seed in SEEDS}
    reference = runs[SEEDS[0]]
    for seed in SEEDS[1:]:
        assert runs[seed].keys() == reference.keys()
        changed = [name for name in reference if runs[seed][name] != reference[name]]
        assert not changed, f"PYTHONHASHSEED={seed} changed: {changed}"
