"""An abstract class nothing specializes yet must not stop the generated ORM from working.

It was written as ``AbstractConcreteBase``, whose polymorphic union over no table raises
``IndexError`` when the mappers are configured: every request of a backend generated from such a
model failed, for a model that only earns a validation warning (and that ``besser model
set-abstract`` or ``remove-parent`` leaves behind on the way to the next change).
"""
import subprocess
import sys

from besser.BUML.metamodel.structural import Class, DomainModel, Generalization, Property, StringType
from besser.generators.sql_alchemy import SQLAlchemyGenerator

_CONFIGURE = "import sql_alchemy\nfrom sqlalchemy.orm import configure_mappers\nconfigure_mappers()\n"


def _model(with_subclass: bool) -> DomainModel:
    party = Class(name="Party", attributes={Property(name="code", type=StringType)}, is_abstract=True)
    book = Class(name="Book", attributes={Property(name="title", type=StringType)})
    types = {party, book}
    generalizations = set()
    if with_subclass:
        person = Class(name="Person", attributes={Property(name="name", type=StringType)})
        types.add(person)
        generalizations.add(Generalization(general=party, specific=person))
    return DomainModel(name="Shop", types=types, generalizations=generalizations)


def _configure(tmp_path, with_subclass: bool) -> tuple[str, subprocess.CompletedProcess]:
    SQLAlchemyGenerator(model=_model(with_subclass), output_dir=str(tmp_path)).generate()
    run = subprocess.run([sys.executable, "-c", _CONFIGURE], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8"), run


def test_the_mappers_configure_when_no_class_specializes_an_abstract_class(tmp_path):
    code, run = _configure(tmp_path, with_subclass=False)

    assert run.returncode == 0, run.stderr
    assert "class Party(Base):" in code and "__abstract__ = True" in code


def test_an_abstract_class_with_a_subclass_is_written_as_before(tmp_path):
    code, run = _configure(tmp_path, with_subclass=True)

    assert run.returncode == 0, run.stderr
    assert "class Party(AbstractConcreteBase, Base):" in code and "__abstract__" not in code
