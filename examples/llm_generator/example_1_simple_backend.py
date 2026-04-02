"""
Example 1: Generate a FastAPI backend with JWT authentication.

The LLM will:
1. Call generate_fastapi_backend() to get base CRUD endpoints
2. Add JWT auth middleware, login/register endpoints
3. Switch from SQLite to PostgreSQL
4. Test the code, fix any errors
5. Generate a README with setup instructions

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY="sk-ant-..."
    # Or for gateway:
    export ANTHROPIC_BASE_URL="https://gateway.pia.private.list.lu"
    export ANTHROPIC_AUTH_TOKEN="your-token"

    python example_1_simple_backend.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)
from besser.generators.llm import LLMGenerator

# ── Define the domain model ──────────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
BooleanType = PrimitiveDataType("bool")
DateTimeType = PrimitiveDataType("datetime")

# Enumerations
role_enum = Enumeration(name="UserRole", literals={
    EnumerationLiteral(name="ADMIN"),
    EnumerationLiteral(name="USER"),
    EnumerationLiteral(name="MODERATOR"),
})

post_status = Enumeration(name="PostStatus", literals={
    EnumerationLiteral(name="DRAFT"),
    EnumerationLiteral(name="PUBLISHED"),
    EnumerationLiteral(name="ARCHIVED"),
})

# Classes
user = Class(name="User")
user.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="email", type=StringType),
    Property(name="username", type=StringType),
    Property(name="password_hash", type=StringType),
    Property(name="bio", type=StringType, is_optional=True),
    Property(name="is_active", type=BooleanType),
}

post = Class(name="Post")
post.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="title", type=StringType),
    Property(name="content", type=StringType),
    Property(name="created_at", type=DateTimeType),
}

comment = Class(name="Comment")
comment.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="text", type=StringType),
    Property(name="created_at", type=DateTimeType),
}

# Associations
user_post = BinaryAssociation(name="User_Post", ends={
    Property(name="author", type=user, multiplicity=Multiplicity(1, 1)),
    Property(name="posts", type=post, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

post_comment = BinaryAssociation(name="Post_Comment", ends={
    Property(name="post", type=post, multiplicity=Multiplicity(1, 1)),
    Property(name="comments", type=comment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

user_comment = BinaryAssociation(name="User_Comment", ends={
    Property(name="commenter", type=user, multiplicity=Multiplicity(1, 1)),
    Property(name="comments", type=comment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

# Domain model
blog_model = DomainModel(
    name="BlogPlatform",
    types={user, post, comment, role_enum, post_status},
    associations={user_post, post_comment, user_comment},
)

# ── Generate with LLM ────────────────────────────────────────────────

from run_helper import run_generator

run_generator(
    model=blog_model,
    output_dir="./output/blog_backend",
    instructions="""
    Build a production-ready FastAPI backend for this blog platform:

    1. Use PostgreSQL as the database
    2. Add JWT authentication (login with email + password, register endpoint)
    3. Add pagination to all list endpoints (limit/offset)
    4. Add a search endpoint for posts (search by title)
    5. Only post authors can edit/delete their own posts
    6. Include requirements.txt with all dependencies
    7. Include a Dockerfile for deployment
    """,
)
