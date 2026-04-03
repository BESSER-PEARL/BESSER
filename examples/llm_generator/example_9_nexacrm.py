"""
Example 9: NexaCRM — Full-stack CRM app from real BUML project.

Uses a real-world CRM domain model (11 classes, 19 associations, 7 enums)
with a GUI model containing multiple screens (Dashboard, Contacts, Companies,
Opportunities, Tasks, etc.).

The LLM will:
1. Use generate_fastapi_backend for the API
2. Use generate_web_app if GUI model is available, or write React from scratch
3. Add JWT authentication
4. Add CRM-specific features (lead scoring, pipeline view, search)
5. Docker deployment setup

Usage:
    $env:ANTHROPIC_BASE_URL = "https://gateway.pia.private.list.lu"
    $env:ANTHROPIC_AUTH_TOKEN = "your-token"
    python example_9_nexacrm.py
"""

import sys
import os

# Execute the BUML project file to get the models
print("Loading NexaCRM BUML model...")
project_file = os.path.join(os.path.dirname(__file__), "class_diagram_project.py")
namespace = {}
with open(project_file, "r", encoding="utf-8") as f:
    code = f.read()

# The structural model is named "user_model" but the GUI code references "domain_model"
# Inject the alias before executing
code = code.replace(
    "###############\n#  GUI MODEL  #",
    "domain_model = user_model\n\n###############\n#  GUI MODEL  #",
    1,
)
exec(code, namespace)

domain_model = namespace["user_model"]
gui_model = namespace.get("gui_model")

print(f"  Domain model: {domain_model.name}")
print(f"  Classes: {', '.join(c.name for c in domain_model.get_classes())}")
print(f"  Enumerations: {', '.join(e.name for e in domain_model.get_enumerations())}")
print(f"  Associations: {len(domain_model.associations)}")
print(f"  GUI model: {'Yes' if gui_model else 'No'}")
print()

# ── Generate full-stack CRM ──────────────────────────────────────────

from run_helper import run_generator

run_generator(
    model=domain_model,
    gui_model=gui_model,
    output_dir="./output/nexacrm_fullstack",
    max_turns=80,
    instructions="""
    Build a production-ready full-stack CRM web application (NexaCRM):

    ## Backend (use the FastAPI generator, then customize)
    1. Use generate_fastapi_backend for base CRUD endpoints
    2. Add JWT authentication with User model (register + login)
    3. Use PostgreSQL
    4. Add pagination on all list endpoints
    5. Add search endpoint for contacts (by name, email, company)
    6. Add search endpoint for companies (by name, industry)
    7. Add filtering on opportunities by stage
    8. Role-based access: ADMIN can do everything, SALES_REP can only manage their own contacts/opportunities

    ## Frontend (write React + TypeScript + Vite from scratch)
    1. Create a modern React SPA with:
       - Sidebar navigation (Dashboard, Contacts, Companies, Opportunities, Tasks)
       - Dashboard page: summary cards (total contacts, open opportunities, pipeline value)
       - Contacts page: searchable table with lead score badges
       - Companies page: filterable list by industry
       - Opportunities page: kanban-style pipeline view grouped by stage
       - Tasks page: list with due dates and assignment
    2. Use Tailwind CSS for styling
    3. Use React Router for navigation
    4. JWT auth: login page, token in localStorage, protected routes
    5. API client with Authorization header on all requests

    ## Docker
    1. docker-compose.yml with: postgres, backend (FastAPI), frontend (nginx)
    2. Dockerfiles for both backend and frontend
    3. .env.example with all config variables

    ## README with setup instructions
    """,
)
