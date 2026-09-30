"""Generated code must be byte-for-byte identical across runs.

Attributes, literals and association ends live in sets whose iteration order
depends on object addresses and string hashes, so templates that loop over
them unsorted emitted ``__str__`` fields, Pydantic/SQLAlchemy columns, enum
literals and router parameters in a different order on every run -- noisy
diffs on every regeneration and a pushed repo that churns for no reason.
Each run below is a fresh interpreter with a different ``PYTHONHASHSEED``.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

GENERATE = textwrap.dedent('''
    import itertools, os, sys
    from datetime import datetime, timedelta
    from besser.BUML.metamodel.structural import (
        AssociationClass, BinaryAssociation, BooleanType, Class, DateType,
        DomainModel, Enumeration, EnumerationLiteral, FloatType, Generalization,
        IntegerType, Method, Multiplicity, Property, StringType,
    )
    from besser.generators.backend import BackendGenerator
    from besser.generators.django import DjangoGenerator
    from besser.generators.pydantic_classes import PydanticGenerator
    from besser.generators.python_classes import PythonGenerator
    from besser.generators.sql_alchemy import SQLAlchemyGenerator

    out = sys.argv[1]
    # Explicit, distinct timestamps: templates that order by timestamp
    # (declaration order) are then well-defined, so any difference between
    # runs comes from iterating a set.
    _tick = itertools.count()
    def ts():
        return datetime(2026, 1, 1) + timedelta(milliseconds=next(_tick))
    def attrs(*spec):
        return {Property(name=n, type=t, timestamp=ts()) for n, t in spec}
    def end(name, cls, lo, hi):
        return Property(name=name, type=cls, multiplicity=Multiplicity(lo, hi), timestamp=ts())
    def methods(*names):
        return {Method(name=n, timestamp=ts()) for n in names}
    status = Enumeration(name="Status", timestamp=ts(), literals={
        EnumerationLiteral(name=n, timestamp=ts())
        for n in ("OPEN", "CLOSED", "PENDING", "ARCHIVED", "DRAFT")})
    person = Class(name="Person", timestamp=ts(), attributes=attrs(
        ("first_name", StringType), ("last_name", StringType), ("email", StringType),
        ("age", IntegerType), ("height", FloatType), ("active", BooleanType)),
        methods=methods("__str__", "get_email", "is_active"))
    member = Class(name="Member", timestamp=ts(), attributes=attrs(
        ("joined", DateType), ("level", IntegerType), ("nickname", StringType),
        ("notes", StringType)), methods=methods("__str__"))
    book = Class(name="Book", timestamp=ts(), attributes=attrs(
        ("title", StringType), ("isbn", StringType), ("pages", IntegerType),
        ("price", FloatType), ("summary", StringType))
        | {Property(name="status", type=status, timestamp=ts())},
        methods=methods("__str__"))
    library = Class(name="Library", timestamp=ts(), attributes=attrs(
        ("name", StringType), ("city", StringType), ("street", StringType), ("zip", StringType)))
    holds = BinaryAssociation(name="holds", timestamp=ts(), ends={
        end("library", library, 1, 1), end("books", book, 0, "*")})
    visits = BinaryAssociation(name="visits", timestamp=ts(), ends={
        end("libraries", library, 0, "*"), end("members", member, 0, "*")})
    loan_assoc = BinaryAssociation(name="Loan_assoc", timestamp=ts(), ends={
        end("borrowers", member, 0, "*"), end("loans", book, 0, "*")})
    loan = AssociationClass(name="Loan", timestamp=ts(), attributes=attrs(
        ("start", DateType), ("due", DateType), ("renewals", IntegerType), ("fee", FloatType)),
        association=loan_assoc)
    model = DomainModel(name="Lib", types={person, member, book, library, loan, status},
                        associations={holds, visits, loan_assoc},
                        generalizations={Generalization(general=person, specific=member, timestamp=ts())})

    PythonGenerator(model, output_dir=os.path.join(out, "python")).generate()
    PydanticGenerator(model, backend=True, output_dir=os.path.join(out, "pydantic")).generate()
    SQLAlchemyGenerator(model, output_dir=os.path.join(out, "sqla")).generate()
    BackendGenerator(model, output_dir=os.path.join(out, "backend")).generate()
    django = DjangoGenerator(model, project_name="proj", app_name="app",
                             output_dir=os.path.join(out, "django"))
    os.makedirs(django._app_dir(), exist_ok=True)
    django.generate_models()
''')

SEEDS = ("0", "1", "2", "3")


def _generate(out: Path, seed: str) -> dict[str, bytes]:
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(repo)}
    subprocess.run([sys.executable, "-c", GENERATE, str(out)], env=env, check=True,
                   cwd=repo, capture_output=True)
    return {
        str(p.relative_to(out)): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".txt"}
    }


def test_generated_code_is_identical_across_hash_seeds(tmp_path):
    runs = {seed: _generate(tmp_path / seed, seed) for seed in SEEDS}
    reference = runs[SEEDS[0]]
    assert reference, "nothing was generated"
    for seed in SEEDS[1:]:
        assert runs[seed].keys() == reference.keys()
        differing = [name for name in reference if runs[seed][name] != reference[name]]
        assert not differing, f"PYTHONHASHSEED={seed} changed: {differing}"


def test_str_lists_fields_sorted_by_name(tmp_path):
    files = _generate(tmp_path / "out", "0")
    python = files[os.path.join("python", "classes.py")].decode()
    assert ('return f"Person(active={self.active}, age={self.age}, email={self.email}, '
            'first_name={self.first_name}, height={self.height}, last_name={self.last_name})"'
            in python)
    # Inherited attributes are listed, in Python and Django alike.
    assert 'return f"Member(active={self.active}, age={self.age}, email={self.email}' in python
    django = files[os.path.join("django", "proj", "app", "models.py")].decode()
    assert 'return f"Member(active={self.active}, age={self.age}, email={self.email}' in django
