"""The agent's guide to indexing a generated column names the str type that can be indexed.

A generated ``str`` column is ``Text``, which MySQL, SQL Server and Oracle refuse to index.
"""

import pytest

from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor


@pytest.mark.parametrize("tool", ["generate_sqlalchemy", "generate_fastapi_backend"])
def test_the_guide_says_to_index_a_str_column_as_a_bounded_string(tool):
    guide = ToolExecutor._MODIFICATION_GUIDES[tool]
    assert "index=True" in guide and "unique=True" in guide
    assert SQLAlchemyGenerator.KEY_STRING_TYPE in guide
