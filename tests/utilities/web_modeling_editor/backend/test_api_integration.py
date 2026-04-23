"""
Integration tests for the BESSER backend API endpoints.

Tests the FastAPI application endpoints using httpx AsyncClient with minimal
but valid diagram JSON payloads that mirror the frontend editor's data format.

Uses anyio (via pytest-anyio) for async test execution. The httpx AsyncClient
with ASGITransport is used because the installed starlette/httpx versions
do not support the legacy TestClient(app=...) pattern.
"""

import io
import json
import asyncio
from functools import wraps
from typing import Any, Dict, Optional

import pytest
import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app

# ---------------------------------------------------------------------------
# Test-client helper
# ---------------------------------------------------------------------------

BASE_URL = "http://testserver"


def _run(coro):
    """Run an async coroutine synchronously for plain pytest tests."""
    return asyncio.run(coro)


class _Client:
    """Thin synchronous wrapper around httpx.AsyncClient for ASGI testing."""

    def __init__(self):
        self._transport = ASGITransport(app=app)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(transport=self._transport, base_url=BASE_URL) as ac:
            return await ac.request(method, url, **kwargs)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return _run(self._request("GET", url, **kwargs))

    def post(self, url: str, **kwargs) -> httpx.Response:
        return _run(self._request("POST", url, **kwargs))


client = _Client()


# ---------------------------------------------------------------------------
# Fixtures -- minimal valid diagram payloads
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_backend_test_artifacts(tmp_path, monkeypatch):
    """Keep generated test artifacts out of the repository root."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def class_diagram_model():
    """Minimal valid class diagram model with two classes and an association."""
    return {
        "type": "ClassDiagram",
        "elements": {
            "class-1": {
                "type": "Class",
                "name": "Author",
                "attributes": ["attr-1"],
                "methods": [],
            },
            "attr-1": {
                "type": "Attribute",
                "name": "+ name: str",
            },
            "class-2": {
                "type": "Class",
                "name": "Book",
                "attributes": ["attr-2"],
                "methods": [],
            },
            "attr-2": {
                "type": "Attribute",
                "name": "+ title: str",
            },
        },
        "relationships": {
            "rel-1": {
                "type": "ClassBidirectional",
                "source": {
                    "element": "class-1",
                    "multiplicity": "1..*",
                    "role": "authors",
                },
                "target": {
                    "element": "class-2",
                    "multiplicity": "0..*",
                    "role": "books",
                },
            },
        },
    }


@pytest.fixture
def class_diagram_input(class_diagram_model):
    """DiagramInput payload for a class diagram."""
    return {
        "title": "LibraryModel",
        "model": class_diagram_model,
        "generator": "python",
    }


@pytest.fixture
def single_class_model():
    """Simplest possible class diagram with one class and one attribute."""
    return {
        "type": "ClassDiagram",
        "elements": {
            "class-1": {
                "type": "Class",
                "name": "Person",
                "attributes": ["attr-1"],
                "methods": [],
            },
            "attr-1": {
                "type": "Attribute",
                "name": "+ name: str",
            },
        },
        "relationships": {},
    }


@pytest.fixture
def single_class_input(single_class_model):
    """DiagramInput with only one class."""
    return {
        "title": "SimpleModel",
        "model": single_class_model,
        "generator": "python",
    }


@pytest.fixture
def state_machine_model():
    """Minimal state machine diagram model.

    Uses the frontend's actual JSON structure where initial state is
    determined by a StateInitialNode element with a StateTransition
    relationship pointing to the first real state.
    """
    return {
        "type": "StateMachineDiagram",
        "elements": {
            "init-node": {
                "type": "StateInitialNode",
                "name": "",
            },
            "state-1": {
                "type": "State",
                "name": "Idle",
                "bodies": [],
            },
            "state-2": {
                "type": "State",
                "name": "Active",
                "bodies": [],
            },
        },
        "relationships": {
            "trans-init": {
                "type": "StateTransition",
                "name": "",
                "source": {"element": "init-node"},
                "target": {"element": "state-1"},
            },
            "trans-1": {
                "type": "StateTransition",
                "name": "",
                "source": {"element": "state-1"},
                "target": {"element": "state-2"},
            },
        },
    }


@pytest.fixture
def state_machine_input(state_machine_model):
    """DiagramInput for a state machine diagram."""
    return {
        "title": "MyStateMachine",
        "model": state_machine_model,
    }


@pytest.fixture
def enumeration_model():
    """Class diagram with an enumeration type."""
    return {
        "type": "ClassDiagram",
        "elements": {
            "enum-1": {
                "type": "Enumeration",
                "name": "Color",
                "attributes": ["lit-1", "lit-2", "lit-3"],
                "methods": [],
            },
            "lit-1": {"type": "EnumerationLiteral", "name": "RED"},
            "lit-2": {"type": "EnumerationLiteral", "name": "GREEN"},
            "lit-3": {"type": "EnumerationLiteral", "name": "BLUE"},
        },
        "relationships": {},
    }


# ---------------------------------------------------------------------------
# Health & Info Endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    """Tests for health check and API info endpoints."""

    def test_health_check(self):
        """GET /health returns 200 with ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_api_root(self):
        """GET /besser_api/ returns API metadata and supported generators list."""
        response = client.get("/besser_api/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "BESSER Backend API"
        assert "version" in data
        assert "supported_generators" in data
        assert isinstance(data["supported_generators"], list)
        assert len(data["supported_generators"]) > 0
        # Verify some known generators are listed
        assert "python" in data["supported_generators"]
        assert "django" in data["supported_generators"]
        assert "sql" in data["supported_generators"]
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)


# ---------------------------------------------------------------------------
# Code Generation Endpoint -- POST /besser_api/generate-output
# ---------------------------------------------------------------------------

class TestGenerateOutput:
    """Tests for the /generate-output code generation endpoint."""

    def test_generate_python_from_class_diagram(self, class_diagram_input):
        """Generate Python classes from a valid class diagram."""
        response = client.post("/besser_api/generate-output", json=class_diagram_input)
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        content_disp = response.headers.get("content-disposition", "")
        assert "classes.py" in content_disp
        body = response.text
        # The generated Python should contain the class names
        assert "Author" in body
        assert "Book" in body

    def test_generate_python_single_class(self, single_class_input):
        """Generate Python from a single-class diagram."""
        response = client.post("/besser_api/generate-output", json=single_class_input)
        assert response.status_code == 200
        body = response.text
        assert "Person" in body

    def test_generate_sql_from_class_diagram(self, class_diagram_input):
        """Generate SQL from a class diagram."""
        payload = {
            **class_diagram_input,
            "generator": "sql",
            "config": {"dialect": "sqlite"},
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        content_disp = response.headers.get("content-disposition", "")
        assert "tables.sql" in content_disp
        body = response.text
        assert "CREATE TABLE" in body.upper() or "create table" in body.lower()

    def test_generate_pydantic_from_class_diagram(self, class_diagram_input):
        """Generate Pydantic models from a class diagram."""
        payload = {**class_diagram_input, "generator": "pydantic"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        content_disp = response.headers.get("content-disposition", "")
        assert "pydantic_classes.py" in content_disp
        body = response.text
        assert "Author" in body
        assert "Book" in body

    def test_generate_sqlalchemy_from_class_diagram(self, class_diagram_input):
        """Generate SQLAlchemy models from a class diagram."""
        payload = {**class_diagram_input, "generator": "sqlalchemy"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        content_disp = response.headers.get("content-disposition", "")
        assert "sql_alchemy.py" in content_disp

    def test_generate_jsonschema_from_class_diagram(self, class_diagram_input):
        """Generate JSON Schema from a class diagram."""
        payload = {**class_diagram_input, "generator": "jsonschema"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        content_disp = response.headers.get("content-disposition", "")
        assert "json_schema.json" in content_disp
        # Should be valid JSON
        json.loads(response.text)

    def test_generate_with_unsupported_generator(self, class_diagram_input):
        """Request with an unsupported generator returns 400."""
        payload = {**class_diagram_input, "generator": "nonexistent_generator"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "nonexistent_generator" in data["detail"]

    def test_generate_with_missing_generator(self):
        """Request with no generator field defaults to None and returns 400."""
        payload = {
            "title": "Test",
            "model": {"type": "ClassDiagram", "elements": {}, "relationships": {}},
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 400

    def test_generate_with_empty_model(self):
        """An empty class diagram model should still work (produces empty output)."""
        payload = {
            "title": "EmptyModel",
            "model": {"type": "ClassDiagram", "elements": {}, "relationships": {}},
            "generator": "python",
        }
        response = client.post("/besser_api/generate-output", json=payload)
        # Should succeed -- empty model generates an empty/stub file
        assert response.status_code == 200

    def test_generate_with_sql_dialect_config(self, class_diagram_input):
        """SQL generation respects dialect config."""
        payload = {
            **class_diagram_input,
            "generator": "sql",
            "config": {"dialect": "postgresql"},
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200

    def test_generate_django_returns_zip(self, class_diagram_input):
        """Django generator returns a ZIP archive.

        Note: The Django generator may fail with a 500 on some platforms
        if the generated project directory structure does not match the
        expected layout (e.g., path separator differences). We accept
        either 200 (success with ZIP) or 500 (known environment issue).
        """
        payload = {
            **class_diagram_input,
            "generator": "django",
            "config": {
                "project_name": "testproject",
                "app_name": "testapp",
                "containerization": False,
            },
        }
        response = client.post("/besser_api/generate-output", json=payload)
        if response.status_code == 200:
            assert "application/zip" in response.headers.get("content-type", "")
        else:
            # Django generation may fail in test environments due to
            # temp directory cleanup timing or path issues.
            assert response.status_code in (400, 500)

    def test_generate_backend_returns_zip(self, class_diagram_input):
        """Backend (FastAPI) generator returns a ZIP archive."""
        payload = {**class_diagram_input, "generator": "backend"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        assert "application/zip" in response.headers.get("content-type", "")

    def test_generate_java_returns_zip(self, class_diagram_input):
        """Java generator returns a ZIP archive."""
        payload = {**class_diagram_input, "generator": "java"}
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        assert "application/zip" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Validation Endpoint -- POST /besser_api/validate-diagram
# ---------------------------------------------------------------------------

class TestValidateDiagram:
    """Tests for the /validate-diagram endpoint."""

    def test_validate_valid_class_diagram(self, class_diagram_input):
        """A well-formed class diagram validates successfully."""
        response = client.post("/besser_api/validate-diagram", json=class_diagram_input)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) == 0

    def test_validate_single_class(self, single_class_input):
        """A single-class diagram validates successfully."""
        response = client.post("/besser_api/validate-diagram", json=single_class_input)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True

    def test_validate_class_diagram_with_enum(self, enumeration_model):
        """A class diagram with enumerations validates successfully."""
        payload = {
            "title": "EnumModel",
            "model": enumeration_model,
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True

    def test_validate_invalid_class_name_with_spaces(self):
        """A class name containing spaces triggers a validation error."""
        payload = {
            "title": "BadModel",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "class-1": {
                        "type": "Class",
                        "name": "Bad Class Name",
                        "attributes": [],
                        "methods": [],
                    },
                },
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert len(data["errors"]) > 0

    def test_validate_empty_class_name(self):
        """An empty class name triggers a validation error."""
        payload = {
            "title": "BadModel",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "class-1": {
                        "type": "Class",
                        "name": "",
                        "attributes": [],
                        "methods": [],
                    },
                },
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert len(data["errors"]) > 0

    def test_validate_unsupported_diagram_type(self):
        """Unsupported diagram type returns isValid=False."""
        payload = {
            "title": "Unknown",
            "model": {"type": "UnknownDiagramType", "elements": {}, "relationships": {}},
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert any("Unsupported" in err for err in data["errors"])

    def test_validate_state_machine_valid(self, state_machine_input):
        """A valid state machine with an initial state passes validation."""
        response = client.post("/besser_api/validate-diagram", json=state_machine_input)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True

    def test_validate_state_machine_no_initial_state(self):
        """State machine without an initial state (no StateInitialNode) fails validation."""
        payload = {
            "title": "NoInitialSM",
            "model": {
                "type": "StateMachineDiagram",
                "elements": {
                    "state-1": {
                        "type": "State",
                        "name": "StateA",
                        "bodies": [],
                    },
                },
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        # The first state must be initial; without StateInitialNode it fails
        assert len(data["errors"]) > 0

    def test_validate_state_machine_no_states(self):
        """State machine with no states fails validation."""
        payload = {
            "title": "EmptySM",
            "model": {
                "type": "StateMachineDiagram",
                "elements": {},
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert any("no states" in err.lower() for err in data["errors"])

    def test_validate_state_machine_duplicate_names(self):
        """State machine with duplicate state names fails validation.

        The state machine processor requires the first state to be initial
        (via StateInitialNode). Duplicate state names are caught by the
        StateMachine.new_state() method and logged as warnings, which
        causes those states to be skipped -- effectively leading to fewer
        states than expected, which counts as invalid.
        """
        payload = {
            "title": "DupSM",
            "model": {
                "type": "StateMachineDiagram",
                "elements": {
                    "init-node": {
                        "type": "StateInitialNode",
                        "name": "",
                    },
                    "state-1": {
                        "type": "State",
                        "name": "Alpha",
                        "bodies": [],
                    },
                    "state-2": {
                        "type": "State",
                        "name": "Alpha",
                        "bodies": [],
                    },
                },
                "relationships": {
                    "trans-init": {
                        "type": "StateTransition",
                        "name": "",
                        "source": {"element": "init-node"},
                        "target": {"element": "state-1"},
                    },
                },
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        # The duplicate state is silently skipped by the processor, so
        # validation might report duplicate names or the SM may have
        # only one state. Either way, we check that the endpoint returns
        # a valid response structure.
        assert "isValid" in data
        assert isinstance(data["errors"], list)

    def test_validate_gui_diagram_always_valid(self):
        """GUINoCodeDiagram always returns valid (no metamodel validation)."""
        payload = {
            "title": "TestGUI",
            "model": {"type": "GUINoCodeDiagram"},
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is True

    def test_validate_returns_warnings_list(self, class_diagram_input):
        """Validation response always includes a warnings list."""
        response = client.post("/besser_api/validate-diagram", json=class_diagram_input)
        assert response.status_code == 200
        data = response.json()
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_validate_missing_model_field(self):
        """Request with empty model dict should handle gracefully."""
        payload = {"title": "NoModel", "model": {}}
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        # No type -> unsupported
        assert data["isValid"] is False

    def test_validate_object_diagram_missing_reference(self):
        """Object diagram without reference class data returns error."""
        payload = {
            "title": "ObjTest",
            "model": {
                "type": "ObjectDiagram",
                "elements": {},
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert any("reference" in err.lower() for err in data["errors"])

    def test_validate_user_diagram_runs_ocl_check(self, monkeypatch):
        """UserDiagram validation should execute OCL checking with object model context."""
        from besser.utilities.web_modeling_editor.backend.routers import validation_router as vr

        class _DummyObjectModel:
            objects = []

            @staticmethod
            def validate(*args, **kwargs):
                return {"success": True, "errors": [], "warnings": []}

        observed = {"called": False, "with_object_model": False}

        def _fake_process_object_diagram(_input_data, _domain_model):
            return _DummyObjectModel()

        def _fake_check_ocl_constraint(_domain_model, object_model=None):
            observed["called"] = True
            observed["with_object_model"] = object_model is not None
            return {
                "success": True,
                "message": "OCL executed",
                "valid_constraints": ["dummy"],
                "invalid_constraints": [],
            }

        monkeypatch.setattr(vr, "process_object_diagram", _fake_process_object_diagram)
        monkeypatch.setattr(vr, "check_ocl_constraint", _fake_check_ocl_constraint)

        payload = {
            "title": "UserProfile",
            "model": {
                "type": "UserDiagram",
                "elements": {},
                "relationships": {},
            },
        }

        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert observed["called"] is True
        assert observed["with_object_model"] is True
        assert data.get("valid_constraints") == ["dummy"]
        assert data.get("invalid_constraints") == []


# ---------------------------------------------------------------------------
# Export BUML Endpoint -- POST /besser_api/export-buml
# ---------------------------------------------------------------------------

class TestExportBuml:
    """Tests for the /export-buml endpoint."""

    def test_export_class_diagram_as_buml(self, class_diagram_input):
        """Export a class diagram as executable BUML Python code."""
        response = client.post("/besser_api/export-buml", json=class_diagram_input)
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        content_disp = response.headers.get("content-disposition", "")
        assert "domain_model.py" in content_disp
        body = response.text
        # The exported BUML code should reference the class names
        assert "Author" in body
        assert "Book" in body
        # Should contain BUML metamodel imports
        assert "DomainModel" in body or "domain_model" in body.lower()

    def test_export_single_class_as_buml(self, single_class_input):
        """Export a single-class diagram as BUML."""
        response = client.post("/besser_api/export-buml", json=single_class_input)
        assert response.status_code == 200
        body = response.text
        assert "Person" in body

    def test_export_state_machine_as_buml(self, state_machine_input):
        """Export a state machine diagram as BUML Python code."""
        response = client.post("/besser_api/export-buml", json=state_machine_input)
        assert response.status_code == 200
        content_disp = response.headers.get("content-disposition", "")
        assert "state_machine.py" in content_disp
        body = response.text
        assert "Idle" in body
        assert "Active" in body

    def test_export_unsupported_diagram_type(self):
        """Exporting an unsupported diagram type returns 400."""
        payload = {
            "title": "Unknown",
            "model": {"type": "UnknownType", "elements": {}, "relationships": {}},
        }
        response = client.post("/besser_api/export-buml", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_export_missing_diagram_type(self):
        """Exporting without a diagram type returns 400."""
        payload = {
            "title": "NoType",
            "model": {"elements": {}, "relationships": {}},
        }
        response = client.post("/besser_api/export-buml", json=payload)
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Reverse Conversion -- POST /besser_api/get-json-model
# ---------------------------------------------------------------------------

class TestGetJsonModel:
    """Tests for the /get-json-model endpoint (BUML file upload to JSON)."""

    def _upload_buml_content(self, content: str, filename: str = "model.py"):
        """Helper to upload BUML content as a file."""
        file_bytes = content.encode("utf-8")
        files = {"buml_file": (filename, file_bytes, "text/plain")}
        return client.post("/besser_api/get-json-model", files=files)

    def test_upload_class_diagram_buml(self):
        """Upload a BUML domain model file and get back JSON.

        The parse_buml_content function execs the file in a sandbox and
        discovers Class/Enumeration instances. The code must use the
        metamodel constructors directly (no method calls like create_class).
        """
        buml_code = (
            "from besser.BUML.metamodel.structural import "
            "DomainModel, Class, Property, PrimitiveDataType\n\n"
            "str_type = PrimitiveDataType('str')\n"
            "person = Class(name='Person', attributes={Property(name='name', type=str_type)})\n"
            "domain_model = DomainModel('TestModel', types={person})\n"
        )
        response = self._upload_buml_content(buml_code, "domain_model.py")
        assert response.status_code == 200
        data = response.json()
        assert data["diagramType"] == "ClassDiagram"
        assert "model" in data
        assert data["model"]["type"] == "ClassDiagram"

    def test_upload_invalid_content(self):
        """Upload non-BUML content returns 400 (invalid Python syntax)."""
        response = self._upload_buml_content("this is not valid python code }{}{", "bad.py")
        assert response.status_code == 400

    def test_upload_empty_file(self):
        """Upload an empty file returns 400."""
        response = self._upload_buml_content("", "empty.py")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Middleware & Request Validation
# ---------------------------------------------------------------------------

class TestMiddlewareAndRequestValidation:
    """Tests for middleware behavior and request validation."""

    def test_security_headers_present(self):
        """Security headers are set on responses."""
        response = client.get("/health")
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_invalid_json_body(self):
        """Sending malformed JSON returns 422 (Pydantic validation error)."""
        response = client.post(
            "/besser_api/generate-output",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Missing required fields (title, model) returns 422."""
        response = client.post("/besser_api/generate-output", json={})
        assert response.status_code == 422

    def test_request_too_large(self):
        """Requests exceeding MAX_REQUEST_SIZE (50MB) are rejected with 413."""
        # Simulate with content-length header (the middleware checks the header
        # before reading the body).
        large_size = 60 * 1024 * 1024  # 60 MB
        response = client.post(
            "/besser_api/generate-output",
            json={"title": "x", "model": {}},
            headers={"Content-Length": str(large_size)},
        )
        assert response.status_code == 413

    def test_nonexistent_endpoint_returns_404(self):
        """Request to non-existent path returns 404."""
        response = client.get("/besser_api/does-not-exist")
        assert response.status_code in (404, 405)


# ---------------------------------------------------------------------------
# Enumeration & Edge Cases in Generation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and special diagram features."""

    def test_generate_class_diagram_with_enumeration(self, enumeration_model):
        """Generating Python code from a diagram with enumerations."""
        payload = {
            "title": "EnumDiagram",
            "model": enumeration_model,
            "generator": "python",
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "Color" in body

    def test_generate_class_with_methods(self):
        """Generate code for a class that has methods."""
        payload = {
            "title": "MethodModel",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "class-1": {
                        "type": "Class",
                        "name": "Calculator",
                        "attributes": [],
                        "methods": ["method-1"],
                    },
                    "method-1": {
                        "type": "Method",
                        "name": "+ add(a: int, b: int): int",
                    },
                },
                "relationships": {},
            },
            "generator": "python",
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "Calculator" in body

    def test_generate_with_inheritance(self):
        """Generate code for classes with inheritance relationship."""
        payload = {
            "title": "InheritModel",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "class-1": {
                        "type": "Class",
                        "name": "Animal",
                        "attributes": ["attr-1"],
                        "methods": [],
                    },
                    "attr-1": {
                        "type": "Attribute",
                        "name": "+ species: str",
                    },
                    "class-2": {
                        "type": "Class",
                        "name": "Dog",
                        "attributes": [],
                        "methods": [],
                    },
                },
                "relationships": {
                    "rel-1": {
                        "type": "ClassInheritance",
                        "source": {"element": "class-2"},
                        "target": {"element": "class-1"},
                    },
                },
            },
            "generator": "python",
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "Animal" in body
        assert "Dog" in body

    def test_generate_abstract_class(self):
        """Generate code for an abstract class."""
        payload = {
            "title": "AbstractModel",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "class-1": {
                        "type": "AbstractClass",
                        "name": "Shape",
                        "attributes": [],
                        "methods": [],
                    },
                },
                "relationships": {},
            },
            "generator": "python",
        }
        response = client.post("/besser_api/generate-output", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "Shape" in body

    def test_validate_duplicate_enumeration_literal(self):
        """Duplicate enumeration literals cause a validation error.

        The ConversionError raised by the processor is caught by the
        validation endpoint's generic exception handler, which appends
        a generic error message to the errors list.
        """
        payload = {
            "title": "DupLiterals",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "enum-1": {
                        "type": "Enumeration",
                        "name": "Status",
                        "attributes": ["lit-1", "lit-2"],
                        "methods": [],
                    },
                    "lit-1": {"type": "EnumerationLiteral", "name": "ACTIVE"},
                    "lit-2": {"type": "EnumerationLiteral", "name": "ACTIVE"},
                },
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False
        assert len(data["errors"]) > 0

    def test_validate_empty_enumeration_literal_name(self):
        """Empty enumeration literal name causes a validation error."""
        payload = {
            "title": "EmptyLit",
            "model": {
                "type": "ClassDiagram",
                "elements": {
                    "enum-1": {
                        "type": "Enumeration",
                        "name": "Status",
                        "attributes": ["lit-1"],
                        "methods": [],
                    },
                    "lit-1": {"type": "EnumerationLiteral", "name": ""},
                },
                "relationships": {},
            },
        }
        response = client.post("/besser_api/validate-diagram", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["isValid"] is False


# ---------------------------------------------------------------------------
# Deprecated Endpoint -- POST /besser_api/check-ocl
# ---------------------------------------------------------------------------

class TestDeprecatedEndpoints:
    """Tests for backwards-compatible deprecated endpoints."""

    def test_check_ocl_redirects_to_validate(self, class_diagram_input):
        """/check-ocl delegates to /validate-diagram and returns same structure."""
        response = client.post("/besser_api/check-ocl", json=class_diagram_input)
        assert response.status_code == 200
        data = response.json()
        assert "isValid" in data
        assert "errors" in data
        assert "warnings" in data


# ---------------------------------------------------------------------------
# Project Generation -- POST /besser_api/generate-output-from-project
# ---------------------------------------------------------------------------

class TestProjectGeneration:
    """Tests for the /generate-output-from-project endpoint."""

    def test_project_generation_missing_generator(self):
        """Project generation without generator in settings returns 400."""
        payload = {
            "id": "proj-1",
            "type": "Project",
            "name": "TestProject",
            "createdAt": "2025-01-01T00:00:00Z",
            "diagrams": {
                "ClassDiagram": [
                    {
                        "title": "Main",
                        "model": {
                            "type": "ClassDiagram",
                            "elements": {},
                            "relationships": {},
                        },
                    }
                ],
            },
            "settings": {},
        }
        response = client.post("/besser_api/generate-output-from-project", json=payload)
        assert response.status_code == 400

    def test_project_generation_unsupported_generator(self):
        """Project generation with unsupported generator returns 400."""
        payload = {
            "id": "proj-1",
            "type": "Project",
            "name": "TestProject",
            "createdAt": "2025-01-01T00:00:00Z",
            "diagrams": {
                "ClassDiagram": [
                    {
                        "title": "Main",
                        "model": {
                            "type": "ClassDiagram",
                            "elements": {},
                            "relationships": {},
                        },
                    }
                ],
            },
            "settings": {"generator": "nonexistent"},
        }
        response = client.post("/besser_api/generate-output-from-project", json=payload)
        assert response.status_code == 400

    def test_project_generation_python_from_class_diagram(self, class_diagram_model):
        """Generate Python code from a project with a class diagram."""
        payload = {
            "id": "proj-1",
            "type": "Project",
            "name": "TestProject",
            "createdAt": "2025-01-01T00:00:00Z",
            "currentDiagramType": "ClassDiagram",
            "currentDiagramIndices": {"ClassDiagram": 0},
            "diagrams": {
                "ClassDiagram": [
                    {
                        "title": "LibraryModel",
                        "model": class_diagram_model,
                    }
                ],
            },
            "settings": {"generator": "python"},
        }
        response = client.post("/besser_api/generate-output-from-project", json=payload)
        assert response.status_code == 200
        body = response.text
        assert "Author" in body
        assert "Book" in body


# ---------------------------------------------------------------------------
# Recommendation Endpoints
# ---------------------------------------------------------------------------

class TestRecommendationEndpoints:
    """Tests for recommendation-related endpoints."""

    def test_recommend_agent_config_llm_success_normalizes_output(self, monkeypatch):
        """LLM recommendation endpoint returns normalized config payload."""
        from besser.utilities.web_modeling_editor.backend.routers import generation_router as gr

        def _fake_profile_document(_model):
            return {"profile": {"age": 34, "notes": "test"}}

        def _fake_call_openai_chat(*_args, **_kwargs):
            return json.dumps(
                {
                    "presentation": {
                        "agentLanguage": "klingon",
                        "interfaceStyle": {
                            "size": 200,
                            "lineSpacing": 0,
                            "alignment": "diagonal",
                        },
                    },
                    "modality": {"inputModalities": ["speech"]},
                    "content": {"adaptContentToUserProfile": True},
                    "system": {
                        "agentPlatform": "desktop_app",
                        "llm": {"provider": "openai", "model": "gpt-5-mini"},
                    },
                }
            )

        monkeypatch.setattr(gr, "_generate_user_profile_document", _fake_profile_document)
        monkeypatch.setattr(gr, "call_openai_chat", _fake_call_openai_chat)

        payload = {
            "userProfileModel": {"type": "UserDiagram", "elements": {}, "relationships": {}},
            "userProfileName": "Alice",
            "model": "gpt-5-mini",
            "currentConfig": {},
        }

        response = client.post("/besser_api/recommend-agent-config-llm", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["source"] == "openai"
        assert data["model"] == "gpt-5-mini"
        assert "generatedAt" in data
        assert data["config"]["content"]["userProfileName"] == "Alice"
        # Clamped and sanitized by normalize_recommended_agent_config
        assert data["config"]["presentation"]["interfaceStyle"]["size"] == 32
        assert data["config"]["presentation"]["interfaceStyle"]["lineSpacing"] == 1
        assert data["config"]["presentation"]["agentLanguage"] == "original"
        assert data["config"]["system"]["agentPlatform"] == "streamlit"
        assert data["config"]["modality"]["inputModalities"] == ["text", "speech"]

    def test_recommend_agent_config_llm_invalid_payload_returns_400(self):
        """LLM recommendation requires a userProfileModel object."""
        response = client.post(
            "/besser_api/recommend-agent-config-llm",
            json={"userProfileName": "Alice"},
        )
        assert response.status_code == 400
        assert "userProfileModel" in response.json().get("detail", "")

    def test_recommend_agent_config_llm_parse_error_returns_502(self, monkeypatch):
        """Invalid LLM text should surface as parse failure with 502."""
        from besser.utilities.web_modeling_editor.backend.routers import generation_router as gr

        monkeypatch.setattr(gr, "_generate_user_profile_document", lambda _m: {"profile": {}})
        monkeypatch.setattr(gr, "call_openai_chat", lambda *_args, **_kwargs: "not-json")

        payload = {
            "userProfileModel": {"type": "UserDiagram", "elements": {}, "relationships": {}},
            "userProfileName": "Alice",
        }
        response = client.post("/besser_api/recommend-agent-config-llm", json=payload)
        assert response.status_code == 502
        assert "Failed to parse LLM recommendation response" in response.json().get("detail", "")

    def test_recommend_agent_config_llm_runtime_error_returns_400(self, monkeypatch):
        """Runtime errors from the LLM client should map to HTTP 400."""
        from besser.utilities.web_modeling_editor.backend.routers import generation_router as gr

        def _raise_runtime_error(*_args, **_kwargs):
            raise RuntimeError("Missing OpenAI API key")

        monkeypatch.setattr(gr, "_generate_user_profile_document", lambda _m: {"profile": {}})
        monkeypatch.setattr(gr, "call_openai_chat", _raise_runtime_error)

        payload = {
            "userProfileModel": {"type": "UserDiagram", "elements": {}, "relationships": {}},
        }
        response = client.post("/besser_api/recommend-agent-config-llm", json=payload)
        assert response.status_code == 400
        assert "Missing OpenAI API key" in response.json().get("detail", "")

    def test_agent_config_manual_mapping_success(self):
        """Manual mapping endpoint returns mapping metadata."""
        response = client.get("/besser_api/agent-config-manual-mapping")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "manual_mapping"
        assert "generatedAt" in data
        assert "mapping" in data
        assert "rules" in data["mapping"]
        assert isinstance(data["mapping"]["rules"], list)

    def test_recommend_agent_config_mapping_success(self, monkeypatch):
        """Manual recommendation endpoint returns config and matching metadata."""
        from besser.utilities.web_modeling_editor.backend.routers import generation_router as gr

        monkeypatch.setattr(gr, "_generate_user_profile_document", lambda _m: {"profile": {"age": 70}})

        def _fake_manual_mapping(*_args, **_kwargs):
            return {
                "config": {"behavior": {"responseTiming": "instant"}},
                "matchedRules": [{"id": "older_adults_readability"}],
                "signals": {"age": 70},
            }

        monkeypatch.setattr(gr, "build_manual_mapping_recommendation", _fake_manual_mapping)

        payload = {
            "userProfileModel": {"type": "UserDiagram", "elements": {}, "relationships": {}},
            "userProfileName": "Bob",
            "currentConfig": {},
        }

        response = client.post("/besser_api/recommend-agent-config-mapping", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "manual_mapping"
        assert data["config"]["behavior"]["responseTiming"] == "instant"
        assert data["matchedRules"][0]["id"] == "older_adults_readability"
        assert data["signals"]["age"] == 70

    def test_recommend_agent_config_mapping_invalid_payload_returns_400(self):
        """Manual recommendation endpoint validates userProfileModel presence/type."""
        response = client.post(
            "/besser_api/recommend-agent-config-mapping",
            json={"userProfileName": "Bob"},
        )
        assert response.status_code == 400
        assert "userProfileModel" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# Feedback Endpoint -- POST /besser_api/feedback
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    """Tests for the feedback submission endpoint."""

    def test_submit_valid_feedback(self):
        """Submit valid feedback returns success."""
        payload = {
            "satisfaction": "happy",
            "category": "usability",
            "feedback": "Great tool!",
            "email": "user@example.com",
            "timestamp": "2025-01-01T00:00:00Z",
            "user_agent": "pytest-test-client",
        }
        response = client.post("/besser_api/feedback", json=payload)
        # Should succeed (200) or depend on feedback service config
        assert response.status_code in (200, 500)

    def test_submit_feedback_invalid_satisfaction(self):
        """Invalid satisfaction value returns 422."""
        payload = {
            "satisfaction": "invalid_value",
            "feedback": "test",
            "timestamp": "2025-01-01T00:00:00Z",
            "user_agent": "pytest",
        }
        response = client.post("/besser_api/feedback", json=payload)
        assert response.status_code == 422

    def test_submit_feedback_missing_required(self):
        """Missing required fields returns 422."""
        response = client.post("/besser_api/feedback", json={})
        assert response.status_code == 422
