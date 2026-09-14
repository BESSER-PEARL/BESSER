"""A generator default must be a value that generator actually accepts.

Live 2026-09-14: ``DEFAULT_SQL_DIALECT`` was ``"standard"``. ``SQLGenerator``
passes ``sql_dialect`` straight through as ``SQLAlchemyGenerator.generate(dbms=)``,
whose ``VALID_DBMS`` does not contain it, so every ``POST /generate-output`` for
``sql`` without an explicit ``config.dialect`` raised::

    ValueError: Invalid DBMS. Valid options are postgresql, mariadb, mysql,
    sqlite, mssql, oracle.

i.e. SQL generation was broken for anyone who did not pick a dialect.
"""

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, IntegerType, Property, StringType,
)
from besser.generators.sql import SQLGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    DEFAULT_DBMS,
    DEFAULT_SQL_DIALECT,
)


@pytest.fixture
def tiny_model():
    task = Class(name="Task", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="title", type=StringType),
    })
    return DomainModel(name="Tiny", types={task})


def test_the_default_sql_dialect_is_one_the_generator_accepts():
    assert DEFAULT_SQL_DIALECT in SQLAlchemyGenerator.VALID_DBMS


def test_the_default_dbms_is_one_the_generator_accepts():
    assert DEFAULT_DBMS in SQLAlchemyGenerator.VALID_DBMS


def test_sql_generation_works_with_the_default_dialect(tiny_model, tmp_path):
    """The exact path /generate-output takes when config.dialect is absent."""
    SQLGenerator(model=tiny_model, output_dir=str(tmp_path),
                 sql_dialect=DEFAULT_SQL_DIALECT).generate()
    produced = list(tmp_path.glob("*.sql"))
    assert produced, "no .sql file was produced"


def test_an_invalid_dialect_is_still_rejected(tiny_model, tmp_path):
    """The guard itself must stay — the bug was the default, not the check."""
    with pytest.raises(ValueError, match="Invalid DBMS"):
        SQLGenerator(model=tiny_model, output_dir=str(tmp_path),
                     sql_dialect="standard").generate()
