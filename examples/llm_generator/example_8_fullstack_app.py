"""
Example 8: Full-stack web app — FastAPI backend + React frontend + Docker.

The LLM will:
1. Call generate_fastapi_backend() for the API
2. Write a React TypeScript frontend from scratch (Vite + React Router + Tailwind)
3. Wire the frontend to the backend API
4. Add JWT authentication on both sides
5. Write Docker setup (docker-compose + Dockerfiles)
6. Test both backend and frontend builds

Usage:
    $env:ANTHROPIC_BASE_URL = "https://gateway.pia.private.list.lu"
    $env:ANTHROPIC_AUTH_TOKEN = "your-token"
    python example_8_fullstack_app.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)

# ── Domain model: Project management app ─────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
BooleanType = PrimitiveDataType("bool")
DateTimeType = PrimitiveDataType("datetime")

priority = Enumeration(name="Priority", literals={
    EnumerationLiteral(name="LOW"),
    EnumerationLiteral(name="MEDIUM"),
    EnumerationLiteral(name="HIGH"),
    EnumerationLiteral(name="CRITICAL"),
})

status = Enumeration(name="TaskStatus", literals={
    EnumerationLiteral(name="TODO"),
    EnumerationLiteral(name="IN_PROGRESS"),
    EnumerationLiteral(name="DONE"),
})

user = Class(name="User")
user.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="email", type=StringType),
    Property(name="name", type=StringType),
    Property(name="password_hash", type=StringType),
}

project = Class(name="Project")
project.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
    Property(name="description", type=StringType, is_optional=True),
}

task = Class(name="Task")
task.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="title", type=StringType),
    Property(name="description", type=StringType, is_optional=True),
    Property(name="completed", type=BooleanType),
    Property(name="created_at", type=DateTimeType),
}

# Associations
user_project = BinaryAssociation(name="User_Project", ends={
    Property(name="owner", type=user, multiplicity=Multiplicity(1, 1)),
    Property(name="projects", type=project, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

project_task = BinaryAssociation(name="Project_Task", ends={
    Property(name="project", type=project, multiplicity=Multiplicity(1, 1)),
    Property(name="tasks", type=task, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

user_task = BinaryAssociation(name="User_Task", ends={
    Property(name="assignee", type=user, multiplicity=Multiplicity(0, 1)),
    Property(name="assigned_tasks", type=task, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

model = DomainModel(
    name="ProjectManager",
    types={user, project, task, priority, status},
    associations={user_project, project_task, user_task},
)

# ── Generate full-stack app ──────────────────────────────────────────

from run_helper import run_generator

run_generator(
    model=model,
    output_dir="./output/projectmanager_fullstack",
    instructions="""
    Build a full-stack project management web app:

    ## Backend (use the FastAPI generator, then customize)
    1. Use generate_fastapi_backend for base CRUD API
    2. Add JWT authentication (register + login endpoints)
    3. Add PostgreSQL support (change from SQLite)
    4. Add pagination on all list endpoints
    5. Only project owners can edit/delete their projects
    6. Only task assignees or project owners can update tasks

    ## Frontend (write from scratch — no React generator available without GUI model)
    1. Create a React + TypeScript app with Vite
    2. Use Tailwind CSS for styling
    3. Pages:
       - /login and /register (auth forms)
       - /projects (list user's projects)
       - /projects/:id (project detail with task list)
       - /tasks/:id (task detail/edit)
    4. Use React Router for navigation
    5. Store JWT token in localStorage
    6. Add Authorization header to all API calls
    7. Create a simple, clean UI (not fancy, just functional)

    ## Docker
    1. docker-compose.yml with 3 services: postgres, backend, frontend
    2. Backend Dockerfile (Python)
    3. Frontend Dockerfile (Node, build with Vite, serve with nginx)
    4. .env.example with all config vars

    ## README
    Include setup instructions: docker compose up --build
    """,
    max_turns=80,
)
