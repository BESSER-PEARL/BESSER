"""
Example 7: Generate multiple outputs from the same model.

Shows how to use the LLM to call several generators and combine their
output into a coherent project with migration scripts, test data, and docs.

The LLM will:
1. Call generate_sql() → SQL table definitions
2. Call generate_json_schema() → JSON Schema for validation
3. Call generate_pydantic() → Pydantic models
4. Write migration scripts
5. Write seed data (JSON fixtures)
6. Write a Python script that validates fixtures against the schema

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python example_7_json_schema_and_sql.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)
from besser.generators.llm import LLMGenerator

# ── Library management model ─────────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
BooleanType = PrimitiveDataType("bool")
DateType = PrimitiveDataType("date")

genre = Enumeration(name="Genre", literals={
    EnumerationLiteral(name="FICTION"),
    EnumerationLiteral(name="NON_FICTION"),
    EnumerationLiteral(name="SCIENCE"),
    EnumerationLiteral(name="HISTORY"),
    EnumerationLiteral(name="TECHNOLOGY"),
})

author = Class(name="Author")
author.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="nationality", type=StringType, is_optional=True),
}

book = Class(name="Book")
book.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="isbn", type=StringType),
    Property(name="title", type=StringType),
    Property(name="year", type=IntegerType),
    Property(name="available", type=BooleanType),
}

member = Class(name="Member")
member.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="email", type=StringType),
    Property(name="membership_date", type=DateType),
}

loan = Class(name="Loan")
loan.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="loan_date", type=DateType),
    Property(name="return_date", type=DateType, is_optional=True),
}

author_book = BinaryAssociation(name="Author_Book", ends={
    Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

member_loan = BinaryAssociation(name="Member_Loan", ends={
    Property(name="member", type=member, multiplicity=Multiplicity(1, 1)),
    Property(name="loans", type=loan, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

book_loan = BinaryAssociation(name="Book_Loan", ends={
    Property(name="book", type=book, multiplicity=Multiplicity(1, 1)),
    Property(name="loans", type=loan, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

library_model = DomainModel(
    name="LibrarySystem",
    types={author, book, member, loan, genre},
    associations={author_book, member_loan, book_loan},
)

# ── Generate multi-format output ─────────────────────────────────────

from run_helper import run_generator

run_generator(
    model=library_model,
    output_dir="./output/library_data",
    instructions="""
    Generate a complete data layer for this library system:

    1. Use generate_sql to create SQL table definitions
    2. Use generate_json_schema to create JSON Schema
    3. Use generate_pydantic to create Pydantic validation models
    4. Write seed data: create fixtures/sample_data.json with 3 authors,
       5 books, 2 members, and 3 loans (realistic data)
    5. Write a validate_fixtures.py script that:
       - Loads the JSON schema
       - Loads the sample data
       - Validates each fixture against its schema
       - Prints results
    6. Write a README explaining the project structure
    """,
)
