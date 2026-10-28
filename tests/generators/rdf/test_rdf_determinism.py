"""The RDF vocabulary must be byte-for-byte identical across runs (#679).

Attributes, literals, parents, associations and association ends are sets
whose iteration order changes between interpreters, so the template emitted
them in a different order on every run, and for an association it could even
describe the other end. Each run is a fresh interpreter with its own
``PYTHONHASHSEED``.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

GENERATE = textwrap.dedent('''
    import sys
    from besser.BUML.metamodel.structural import (
        BinaryAssociation, BooleanType, Class, DateType, DomainModel, Enumeration,
        EnumerationLiteral, FloatType, Generalization, IntegerType, Multiplicity,
        Property, StringType,
    )
    from besser.generators.rdf import RDFGenerator

    def end(name, cls, lo, hi):
        return Property(name=name, type=cls, multiplicity=Multiplicity(lo, hi))

    status = Enumeration(name="Status", literals={
        EnumerationLiteral(name=n) for n in ("OPEN", "CLOSED", "PENDING", "ARCHIVED", "DRAFT")})
    person = Class(name="Person", attributes={
        Property(name=n, type=t) for n, t in (
            ("first_name", StringType), ("last_name", StringType), ("email", StringType),
            ("age", IntegerType), ("height", FloatType), ("active", BooleanType))})
    agent = Class(name="Agent", attributes={Property(name="code", type=StringType)})
    member = Class(name="Member", attributes={
        Property(name="joined", type=DateType), Property(name="level", type=IntegerType),
        Property(name="status", type=status)})
    library = Class(name="Library", attributes={
        Property(name=n, type=StringType) for n in ("name", "city", "street", "zip")})
    book = Class(name="Book", attributes={
        Property(name=n, type=StringType) for n in ("title", "isbn", "summary")})
    model = DomainModel(
        name="Lib", types={person, agent, member, library, book, status},
        associations={
            BinaryAssociation(name="holds", ends={end("library", library, 1, 1), end("books", book, 0, "*")}),
            BinaryAssociation(name="visits", ends={end("libraries", library, 0, "*"), end("members", member, 0, "*")}),
            BinaryAssociation(name="flagship", ends={end("home", library, 0, 1), end("featured", book, 0, 1)}),
            BinaryAssociation(name="manages", ends={end("manager", member, 0, 1), end("reports", member, 0, "*")}),
        },
        generalizations={Generalization(general=person, specific=member),
                         Generalization(general=agent, specific=member)})
    RDFGenerator(model, output_dir=sys.argv[1]).generate()
''')

SEEDS = ("0", "1", "2", "3")


def _generate(out: Path, seed: str) -> bytes:
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)}
    subprocess.run([sys.executable, "-c", GENERATE, str(out)], env=env, check=True, cwd=repo,
                   capture_output=True)
    return (out / "vocabulary.ttl").read_bytes()


def test_rdf_vocabulary_is_identical_across_hash_seeds(tmp_path):
    runs = {seed: _generate(tmp_path / seed, seed) for seed in SEEDS}
    changed = [seed for seed in SEEDS[1:] if runs[seed] != runs[SEEDS[0]]]
    assert not changed, f"PYTHONHASHSEED {changed} changed vocabulary.ttl"
