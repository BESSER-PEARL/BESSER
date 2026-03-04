# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BESSER is a low-code platform for building software through model-driven engineering. It consists of:
- **B-UML**: A Python-based metamodel for describing domain models, state machines, GUI designs, agents, quantum circuits, and more
- **Code Generators**: Transform B-UML models into executable code (Django, FastAPI, SQLAlchemy, Flutter, React, etc.)
- **Web Modeling Editor Backend**: FastAPI services powering the online visual editor at https://editor.besser-pearl.org
- **Frontend Submodule**: TypeScript/React UI at `besser/utilities/web_modeling_editor/frontend` (maintained separately)

## Essential Commands

### Setup and Installation
```bash
# Create virtual environment and install dependencies
python -m venv venv
venv/Script/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .

# Install documentation dependencies (optional)
pip install -r docs/requirements.txt
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test module
python -m pytest tests/generators -k sqlalchemy

# Run targeted tests
python -m pytest tests/BUML/metamodel/structural -k library

# Run a standalone example to verify setup
python tests/BUML/metamodel/structural/library/library.py
```

### Documentation
```bash
# Build documentation locally
cd docs
make html  # Windows: make.bat html

# View built docs
# Open docs/build/html/index.html in browser
```

### Local Stack Deployment
```bash
# Run full stack with Docker Compose
docker-compose up --build
```

## Architecture Overview

### Core Architecture Pattern: Metamodel → Conversion → Generation

```
Frontend JSON (visual editor)
    ↕ (json_to_buml / buml_to_json converters)
BUML Metamodel (Python objects)
    ↕ (notations parsers / code builders)
Generated Code (Django, React, SQL, etc.)
```

### Major Components

#### 1. B-UML Metamodel (`besser/BUML/metamodel/`)

The metamodel defines abstract syntax for all domain concepts:

- **`structural/`**: Core object-oriented modeling
  - `DomainModel` - container for all types
  - `Class`, `Property`, `Method` - OOP constructs
  - `Association` types: `BinaryAssociation`, `AssociationClass`
  - `Generalization` - inheritance relationships
  - `PrimitiveDataType`, `Enumeration` - data types
  - Base classes: `Element` → `NamedElement` → domain concepts

- **`state_machine/`**: Behavioral state modeling
  - `StateMachine`, `State`, `Transition`

- **`gui/`**: UI modeling
  - `GUIModel` - top-level container
  - `Screen`, `Module` - logical organization
  - `ViewComponent`, `ViewContainer` - UI building blocks
  - `Style`, `Binding`, `EventsActions` - styling/interactivity

- **`action_language/`**: Business Action Language (BAL) for behavior specification

- **Other metamodels**: `feature_model/`, `nn/` (neural networks), `quantum/`, `deployment/`, `object/` (instances)

**Key Pattern**: Private properties with getter/setter validation. Base classes define interfaces, subclasses add domain-specific behavior.

#### 2. Notations (`besser/BUML/notations/`)

Parse concrete syntaxes into metamodel instances:

- **`structuralPlantUML/`**: ANTLR-based PlantUML parser for class diagrams
- **`objectPlantUML/`**: Object diagram notation
- **`ocl/`**: Object Constraint Language support (`BOCLParser`, `BOCLLexer`)
- **`mockup_to_buml/`**: LLM-assisted UI mockup → GUI model conversion
- **`nn/`**, **`deployment/`**: Specialized notation parsers

**Pattern**: Grammar-based parsing (ANTLR) + listener pattern for AST traversal.

#### 3. Generators (`besser/generators/`)

Transform B-UML models into executable artifacts. All implement `GeneratorInterface`:

```python
class GeneratorInterface(ABC):
    def __init__(self, model: Model, output_dir: str): ...
    def generate(self): ...
```

**Generator Categories** (see `utilities/web_modeling_editor/backend/config/generators.py`):

- **Object-Oriented**: `PythonGenerator`, `JavaGenerator`, `PydanticGenerator`
- **Web Frameworks**: `DjangoGenerator`, `BackendGenerator` (FastAPI), `WebAppGenerator` (full-stack)
- **Databases**: `SQLGenerator`, `SQLAlchemyGenerator`
- **Data Formats**: `JSONSchemaGenerator`
- **Frontend**: `ReactGenerator`, `FlutterGenerator`
- **AI/Agents**: `BAFGenerator` (BESSER Agent Framework)
- **Specialized**: `RESTAPIGenerator`, `NNCodeGenerator`, `QiskitGenerator`, `TerraformGenerator`

**Key Pattern**: Template-based generation with Jinja2. Templates live in `generators/[type]/templates/`.

#### 4. Web Modeling Editor Backend (`besser/utilities/web_modeling_editor/backend/`)

FastAPI service with multi-service gateway pattern. Main entry point: `backend.py`.

**Core Services**:

- **Conversion Services** (`services/json_to_buml/`, `services/buml_to_json/`):
  - Bidirectional transformations between frontend JSON and B-UML metamodel
  - 7 processors: class diagrams, state machines, agents, objects, GUI, quantum circuits, projects
  - Detailed parsers for attributes, methods, multiplicity, OCL constraints

- **Validation Services** (`services/validators/`):
  - 3-level validation: construction (setters), metamodel (`.validate()`), OCL constraints
  - Unified validation endpoint: `/validate-diagram`

- **Deployment Services** (`services/deployment/`):
  - Docker Compose orchestration (`docker_deployment.py`)
  - GitHub integration (`github_service.py`, `github_oauth.py`, `github_deploy_api.py`)

- **Reverse Engineering** (`services/reverse_engineering/`):
  - CSV → domain model (`csv_reverse.py`)
  - Image → class diagram (OpenAI integration)

**Key API Endpoints**:
- `POST /generate-output` - Single diagram → code
- `POST /generate-output-from-project` - Multi-diagram project → code (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)
- `POST /export-buml` - Diagram JSON → BUML Python code
- `POST /get-json-model` - BUML file → JSON (auto-detects diagram type)
- `POST /get-json-model-from-image` - Image → ClassDiagram JSON (via OpenAI)
- `POST /validate-diagram` - Unified validation
- `POST /deploy-app` - Docker Compose deployment

**Configuration Layer** (`backend/config/`):
- `generators.py` - Centralized generator registry with metadata (`GeneratorInfo` NamedTuple)

**Models Layer** (`backend/models/`):
- Pydantic schemas: `DiagramInput`, `ProjectInput`, `FeedbackSubmission`

#### 5. BUML Code Builders (`besser/utilities/buml_code_builder/`)

Generate executable Python code from B-UML metamodel instances:
- `domain_model_builder.py` - DomainModel → Python code
- `agent_model_builder.py` - AgentModel → Python code
- `gui_model_builder.py` - GUIModel → Python code
- `project_builder.py` - Project → Python code
- `quantum_model_builder.py` - QuantumCircuit → Python code

**Pattern**: Generated code can be `exec()`'d to recreate the metamodel instance.

### Multi-Diagram Projects

Some generators require multiple diagram types:
- **WebAppGenerator**: Needs `ClassDiagram` (backend) + `GUINoCodeDiagram` (frontend) + optional `AgentDiagram`
- Projects use `ProjectInput` with `diagrams` dict containing typed diagram data

**Flow Example**:
```
1. Frontend sends ProjectInput with ClassDiagram + GUINoCodeDiagram
2. /generate-output-from-project endpoint
3. ClassDiagram JSON → process_class_diagram → DomainModel
4. GUINoCodeDiagram JSON → process_gui_diagram → GUIModel
5. WebAppGenerator(domain_model, gui_model, agent_model)
6. Templates rendered → React/TypeScript + FastAPI backend
7. ZIP streamed to frontend
```

## Important Conventions

### Code Style
- PEP 8 with 4-space indentation, 120-character line limit (configured in `pyproject.toml`)
- Type hints for public APIs, descriptive docstrings
- Naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- Import order: standard library, third-party, local modules

### Validation Strategy
Three layers:
1. **Construction validation**: Setter constraints in metamodel (e.g., multiplicity bounds)
2. **Metamodel validation**: `.validate()` method checks structural rules
3. **Constraint validation**: OCL constraints evaluated on models

All validation errors should be collected (not thrown) for unified reporting.

### Generator Development
To add a new generator:
1. Create package in `besser/generators/[name]/`
2. Implement `GeneratorInterface` with `__init__(model, output_dir)` and `generate()`
3. Add templates in `generators/[name]/templates/`
4. Register in `utilities/web_modeling_editor/backend/config/generators.py`
5. Add tests in `tests/generators/[name]/`
6. Document in `docs/source/generators.rst`

### Resource Management
- Temp directories use UUID prefixes for uniqueness
- Always use try-finally blocks for cleanup
- ZIP streaming for large outputs to avoid memory issues

## Frontend Integration

The frontend lives at `besser/utilities/web_modeling_editor/frontend` (git submodule).

**Important**:
- Frontend code can be modified when necessary (features, bug fixes, improvements)
- Keep TypeScript/React tooling consistent with upstream (`npm install`, `npm test`)
- Update OpenAPI schemas when changing backend API contracts

## Testing Approach

- Place tests in `tests/` mirroring source structure
- Name test files `test_*.py`
- Add tests for behavioral changes, especially metamodel and generator logic
- Use fixtures under `tests/fixtures/` or inline in test files
- Validate both structure (class names, endpoints) and content (business logic)

## Documentation Sync

Keep docs in `docs/source/` synchronized with code changes:
- `buml_language.rst` - Metamodel additions
- `generators.rst` - New generator documentation
- `web_editor.rst` - API endpoint changes
- `utilities.rst` - New utility documentation
- `contributor_guide.rst`, `ai_assistant_guide.rst` - Workflow changes

Build locally before committing: `cd docs && make html`

## Commit Conventions

Recent history uses Conventional Commits style:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code restructuring
- `docs:` - Documentation updates
- `test:` - Test additions/modifications

Keep subjects short and imperative. Use topic branches (`feature/add-generator`).

## Related Files

- **`.github/copilot-instructions.md`**: Comprehensive AI assistant guidelines (also in `.cursorrules`)
- **`CONTRIBUTING.md`**: Contribution workflow and expectations
- **`AGENTS.md`**: Custom Claude Code agent configuration for this repository
- **`README.md`**: Project overview and quick start
- **`GOVERNANCE.md`**: Project governance and decision-making

## Key Technical Patterns

### Multiplicity Constant
```python
UNLIMITED_MAX_MULTIPLICITY = 9999  # Used throughout for "many" relationships
```

### Metamodel Base Hierarchy
```
Element (base) → NamedElement → {Class, Property, Method, Association, ...}
```

### Generator Registry Pattern
Generators registered in `config/generators.py` with metadata:
```python
GeneratorInfo(
    generator_class=DjangoGenerator,
    output_type="application/zip",
    file_extension=".zip",
    category="web"
)
```

### Bidirectional Converters
Always maintain symmetry:
- `json_to_buml/process_class_diagram.py` ↔ `buml_to_json/class_to_json.py`
- Same features supported in both directions

### Template Rendering
```python
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates/'))
template = env.get_template('model.py.j2')
output = template.render(model=domain_model, config=config)
```

## Common Pitfalls to Avoid

1. **Don't duplicate logic**: Shared helpers belong in `besser/utilities`, not in individual generators
2. **Maintain determinism**: Generators should produce identical output for identical input (avoid timestamps in file names)
3. **Validate early**: Use construction validation in setters to fail fast
4. **Clean up resources**: Always use try-finally for temp directories and file handles
5. **Keep converters symmetric**: If JSON→BUML supports a feature, BUML→JSON must too
6. **Test round-trips**: Especially for converters (JSON→BUML→JSON should be identity)
7. **Update docs**: Backend changes often require `docs/source/` updates

## Debugging Tips

### Running Individual Examples
```bash
cd tests/BUML/metamodel/structural/library
python library.py
```

### Inspecting Generated Output
Check `generated/` directory (git-ignored build output).

### Validation Debugging
Enable verbose OCL validation in `/validate-diagram` endpoint responses.

### Frontend-Backend Integration
Use browser DevTools Network tab to inspect API payloads. Backend returns detailed error messages.

## Support Resources

- Documentation: https://besser.readthedocs.io/
- Online Editor: https://editor.besser-pearl.org/
- Examples: https://github.com/BESSER-PEARL/BESSER-examples
- Contributor Guide: `docs/source/contributor_guide.rst`
- AI Assistant Guide: `docs/source/ai_assistant_guide.rst`
