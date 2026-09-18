"""
Example 4: Generate Pydantic + SQLAlchemy models and enhance them.

Demonstrates using multiple generators together, then customizing.
The LLM will:
1. Call generate_pydantic() → Pydantic validation models
2. Call generate_sqlalchemy() → SQLAlchemy ORM with PostgreSQL
3. Add Alembic migration setup
4. Add custom Pydantic validators (email format, price range)
5. Add SQLAlchemy indexes and constraints
6. Verify syntax

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python example_4_pydantic_sqlalchemy.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Constraint,
)
from besser.generators.llm import LLMGenerator

# ── Simple inventory model ───────────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
FloatType = PrimitiveDataType("float")
BooleanType = PrimitiveDataType("bool")

warehouse = Class(name="Warehouse")
warehouse.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="location", type=StringType),
    Property(name="capacity", type=IntegerType),
}

item = Class(name="Item")
item.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="sku", type=StringType),
    Property(name="name", type=StringType),
    Property(name="price", type=FloatType),
    Property(name="quantity", type=IntegerType),
    Property(name="is_fragile", type=BooleanType),
}

warehouse_item = BinaryAssociation(name="Warehouse_Item", ends={
    Property(name="warehouse", type=warehouse, multiplicity=Multiplicity(1, 1)),
    Property(name="items", type=item, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

inventory_model = DomainModel(
    name="InventorySystem",
    types={warehouse, item},
    associations={warehouse_item},
)

# ── Generate and enhance ─────────────────────────────────────────────

from run_helper import run_generator

run_generator(
    model=inventory_model,
    output_dir="./output/inventory_models",
    instructions="""
    Generate data models for this inventory system:

    1. Generate Pydantic models (use the generator)
    2. Generate SQLAlchemy models with PostgreSQL (use the generator)
    3. Enhance the Pydantic models:
       - Add a validator that ensures price > 0
       - Add a validator that ensures quantity >= 0
       - Add a validator that ensures SKU matches pattern 'XXX-NNN'
    4. Enhance the SQLAlchemy models:
       - Add a unique constraint on Item.sku
       - Add an index on Item.name
       - Add a check constraint: price > 0
    5. Set up Alembic migration:
       - Create alembic.ini and alembic/ directory
       - Write the initial migration script
    6. Create a simple test script that imports both modules to verify they work
    """,
)
