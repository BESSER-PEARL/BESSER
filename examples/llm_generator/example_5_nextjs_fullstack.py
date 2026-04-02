"""
Example 5: Generate a Next.js full-stack app (no generator exists).

Demonstrates the LLM writing a complete modern web app from scratch
using only the domain model as specification.

The LLM will:
1. See that no Next.js generator exists in BESSER
2. Write the entire Next.js app from scratch:
   - App Router with TypeScript
   - Prisma ORM for database
   - Server Actions for mutations
   - React Server Components
3. Use the model to ensure correct entities and relationships
4. Test the build

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python example_5_nextjs_fullstack.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)
from besser.generators.llm import LLMGenerator

# ── Task management model ────────────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
BooleanType = PrimitiveDataType("bool")
DateTimeType = PrimitiveDataType("datetime")

priority = Enumeration(name="Priority", literals={
    EnumerationLiteral(name="LOW"),
    EnumerationLiteral(name="MEDIUM"),
    EnumerationLiteral(name="HIGH"),
    EnumerationLiteral(name="URGENT"),
})

task_status = Enumeration(name="TaskStatus", literals={
    EnumerationLiteral(name="TODO"),
    EnumerationLiteral(name="IN_PROGRESS"),
    EnumerationLiteral(name="IN_REVIEW"),
    EnumerationLiteral(name="DONE"),
})

team = Class(name="Team")
team.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="description", type=StringType, is_optional=True),
}

member = Class(name="Member")
member.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="email", type=StringType),
    Property(name="avatar_url", type=StringType, is_optional=True),
}

task = Class(name="Task")
task.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="title", type=StringType),
    Property(name="description", type=StringType, is_optional=True),
    Property(name="due_date", type=DateTimeType, is_optional=True),
    Property(name="is_completed", type=BooleanType),
}

# Associations
team_member = BinaryAssociation(name="Team_Member", ends={
    Property(name="team", type=team, multiplicity=Multiplicity(1, 1)),
    Property(name="members", type=member, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

member_task = BinaryAssociation(name="Member_Task", ends={
    Property(name="assignee", type=member, multiplicity=Multiplicity(0, 1)),
    Property(name="tasks", type=task, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

team_task = BinaryAssociation(name="Team_Task", ends={
    Property(name="team", type=team, multiplicity=Multiplicity(1, 1)),
    Property(name="tasks", type=task, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

taskmanager_model = DomainModel(
    name="TaskManager",
    types={team, member, task, priority, task_status},
    associations={team_member, member_task, team_task},
)

# ── Generate Next.js app (from scratch — no generator) ───────────────

from run_helper import run_generator

run_generator(
    model=taskmanager_model,
    output_dir="./output/taskmanager_nextjs",
    instructions="""
    Build a modern Next.js full-stack task management app:

    1. Use Next.js 14+ with App Router and TypeScript
    2. Use Prisma ORM with SQLite (for easy local dev)
    3. Create pages:
       - /teams — list all teams
       - /teams/[id] — team detail with members and tasks
       - /tasks — kanban board grouped by TaskStatus
    4. Use Server Actions for creating/updating tasks and members
    5. Use Tailwind CSS for styling
    6. Add a dark mode toggle
    7. Include package.json, tsconfig.json, prisma/schema.prisma
    8. Include a README with: npm install && npx prisma db push && npm run dev
    """,
)
