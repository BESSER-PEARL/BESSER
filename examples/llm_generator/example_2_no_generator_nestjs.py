"""
Example 2: Generate a NestJS backend (no BESSER generator exists).

BESSER has no NestJS generator — the LLM writes everything from scratch
using the domain model as specification. It knows the exact entities,
attributes, types, and relationships to produce correct TypeScript code.

The LLM will:
1. See that no NestJS generator is available
2. Write NestJS modules, controllers, services, DTOs from scratch
3. Wire up TypeORM entities matching the domain model
4. Add Swagger/OpenAPI documentation
5. Test the build

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python example_2_no_generator_nestjs.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)
from besser.generators.llm import LLMGenerator

# ── Define a simple e-commerce model ─────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
FloatType = PrimitiveDataType("float")
BooleanType = PrimitiveDataType("bool")

order_status = Enumeration(name="OrderStatus", literals={
    EnumerationLiteral(name="PENDING"),
    EnumerationLiteral(name="CONFIRMED"),
    EnumerationLiteral(name="SHIPPED"),
    EnumerationLiteral(name="DELIVERED"),
    EnumerationLiteral(name="CANCELLED"),
})

customer = Class(name="Customer")
customer.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="email", type=StringType),
    Property(name="first_name", type=StringType),
    Property(name="last_name", type=StringType),
    Property(name="phone", type=StringType, is_optional=True),
}

product = Class(name="Product")
product.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="description", type=StringType),
    Property(name="price", type=FloatType),
    Property(name="stock", type=IntegerType),
    Property(name="is_available", type=BooleanType),
}

order = Class(name="Order")
order.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="total", type=FloatType),
    Property(name="quantity", type=IntegerType),
}

# Associations
customer_order = BinaryAssociation(name="Customer_Order", ends={
    Property(name="customer", type=customer, multiplicity=Multiplicity(1, 1)),
    Property(name="orders", type=order, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

order_product = BinaryAssociation(name="Order_Product", ends={
    Property(name="order", type=order, multiplicity=Multiplicity(1, 1)),
    Property(name="product", type=product, multiplicity=Multiplicity(1, 1)),
})

ecommerce_model = DomainModel(
    name="ECommerce",
    types={customer, product, order, order_status},
    associations={customer_order, order_product},
)

# ── Generate NestJS backend (no generator exists — LLM writes from scratch) ──

from run_helper import run_generator

run_generator(
    model=ecommerce_model,
    output_dir="./output/ecommerce_nestjs",
    instructions="""
    Build a NestJS backend for this e-commerce platform:

    1. Use NestJS with TypeScript
    2. Use TypeORM with PostgreSQL
    3. Create modules, controllers, services, and DTOs for each entity
    4. Add Swagger/OpenAPI documentation with @nestjs/swagger
    5. Add input validation with class-validator
    6. Add pagination on list endpoints
    7. Calculate order total automatically from product price * quantity
    8. Include package.json, tsconfig.json, and a Dockerfile
    9. Include a README with setup instructions
    """,
)
