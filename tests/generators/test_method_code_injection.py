"""``SQLGenerator`` must not run the code of the model's methods.

It prints the tables by running the module ``SQLAlchemyGenerator`` writes, in a
subprocess. A method's code comes from the model (an editor JSON export as much
as a ``.py`` file) and was written into the class body, so a statement after the
method's ``def`` ran at generation time. ``SQLGenerator`` now asks for the module
without the methods; the SQLAlchemy output itself keeps them.
"""
from besser.BUML.metamodel.structural import Class, DomainModel, Method, Property, StringType
from besser.generators.sql import SQLGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _model_whose_method_writes(marker) -> DomainModel:
    code = (
        "def summary(self):\n"
        "    return self.title\n"
        f"open({str(marker)!r}, 'w').write('ran')\n"
    )
    book = Class(name="Book", attributes={Property(name="title", type=StringType)},
                 methods={Method(name="summary", code=code)})
    return DomainModel(name="Library", types={book})


def test_sql_generation_does_not_run_method_code(tmp_path):
    marker = tmp_path / "ran.txt"

    SQLGenerator(_model_whose_method_writes(marker), output_dir=str(tmp_path / "out")).generate()

    assert not marker.exists()
    ddl = (tmp_path / "out" / "tables_sqlite.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE book" in ddl


def test_the_sqlalchemy_module_keeps_the_methods(tmp_path):
    marker = tmp_path / "ran.txt"

    SQLAlchemyGenerator(_model_whose_method_writes(marker), output_dir=str(tmp_path)).generate()

    assert "def summary(self):" in (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    assert not marker.exists()
