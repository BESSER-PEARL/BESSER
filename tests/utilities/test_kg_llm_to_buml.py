"""Tests for the LLM-assisted KG conversion, ``besser.utilities.kg_llm_to_buml``.

The OpenAI transport is mocked (``requests.post``); the model's answer is
turned into a real :class:`DomainModel`.
"""

import importlib
import json
import sys
import warnings

import pytest
from unittest.mock import patch, mock_open, MagicMock

from besser.BUML.metamodel.structural import DomainModel, IntegerType, StringType
from besser.utilities.llm_utils import LLMUpstreamError
from besser.utilities.kg_llm_to_buml import (
    LLMAnswerError,
    kg_to_buml,
    kg_to_plantuml,
    clean_plantuml_response,
    clean_json_response,
    parse_json_safely,
    convert_spec_json_to_buml,
    spec_to_domain_model,
)

def _mock_plantuml_response():
    """Returns a mock PlantUML response from OpenAI"""
    return """```plantuml
@startuml
class Person {
  - name: str
  - age: int
  + getName(): str
}

class Car {
  - model: str
  - year: int
}

Person "1" -- "*" Car : owns
@enduml
```"""

def _mock_gpt_json_response():
    """Returns a mock JSON response from OpenAI"""
    return """```json
{
  "systemName": "KnowledgeGraphSystem",
  "classes": [
    {
      "className": "Person",
      "attributes": [
        {"name": "name", "type": "str", "visibility": "private"},
        {"name": "age", "type": "int", "visibility": "private"}
      ],
      "methods": [
        {
          "name": "getName",
          "returnType": "str",
          "visibility": "public",
          "parameters": []
        }
      ]
    },
    {
      "className": "Car",
      "attributes": [
        {"name": "model", "type": "str", "visibility": "private"},
        {"name": "year", "type": "int", "visibility": "private"}
      ],
      "methods": []
    }
  ],
  "relationships": [
    {
      "type": "Association",
      "source": "Person",
      "target": "Car",
      "sourceMultiplicity": "1",
      "targetMultiplicity": "*",
      "name": "owns"
    }
  ]
}
```"""

def _mock_system_spec():
    """Returns a mock system specification dictionary"""
    return {
        "systemName": "KnowledgeGraphSystem",
        "classes": [
            {
                "className": "Person",
                "attributes": [
                    {"name": "name", "type": "str", "visibility": "private"},
                    {"name": "age", "type": "int", "visibility": "private"}
                ],
                "methods": [
                    {
                        "name": "getName",
                        "returnType": "str",
                        "visibility": "public",
                        "parameters": []
                    }
                ]
            },
            {
                "className": "Car",
                "attributes": [
                    {"name": "model", "type": "str", "visibility": "private"},
                    {"name": "year", "type": "int", "visibility": "private"}
                ],
                "methods": []
            }
        ],
        "relationships": [
            {
                "type": "Association",
                "source": "Person",
                "target": "Car",
                "sourceMultiplicity": "1",
                "targetMultiplicity": "*",
                "name": "owns"
            }
        ]
    }

def _mock_openai_api_response(content):
    """Creates a mock response object from OpenAI API"""
    return {
        "choices": [
            {
                "message": {
                    "content": content
                }
            }
        ]
    }

def test_clean_plantuml_response_removes_markdown():
    raw = """Some text before
@startuml
class Test
@enduml
Some text after"""
    cleaned = clean_plantuml_response(raw)
    assert cleaned == "@startuml\nclass Test\n@enduml"

def test_clean_json_response_strips_markdown_fence():
    fenced = """```json
{"className": "User"}
```"""
    cleaned = clean_json_response(fenced)
    assert cleaned == '{"className": "User"}'
    
def test_parse_json_safely_valid_json():
    result = parse_json_safely('{"key": "value"}')
    assert result == {"key": "value"}
    

def test_parse_json_safely_invalid_json():
    result = parse_json_safely('not json at all')
    assert result is None


# --- Tests for convert_spec_json_to_buml ---
def test_convert_spec_json_to_buml_creates_valid_structure():
    """Test that convert_spec_json_to_buml creates proper Apollon/BUML JSON"""
    system_spec = _mock_system_spec()
    result = convert_spec_json_to_buml(system_spec)
    
    # Verify top-level structure
    assert "title" in result
    assert "model" in result
    assert "elements" in result["model"]
    assert "relationships" in result["model"]
    
    # Verify classes are created
    elements = result["model"]["elements"]
    class_elements = {k: v for k, v in elements.items() if v.get("type") == "Class"}
    assert len(class_elements) >= 2  # Person and Car
    
    # Verify attributes are created
    attr_elements = {k: v for k, v in elements.items() if v.get("type") == "ClassAttribute"}
    assert len(attr_elements) >= 4  # name, age for Person; model, year for Car
    
    # Verify methods are created
    method_elements = {k: v for k, v in elements.items() if v.get("type") == "ClassMethod"}
    assert len(method_elements) >= 1  # getName for Person


def test_convert_spec_json_to_buml_handles_void_return_type():
    """Test that void return types are handled correctly"""
    system_spec = {
        "systemName": "TestSystem",
        "classes": [
            {
                "className": "TestClass",
                "attributes": [],
                "methods": [
                    {
                        "name": "doSomething",
                        "returnType": "void",
                        "visibility": "public",
                        "parameters": []
                    }
                ]
            }
        ],
        "relationships": []
    }
    
    result = convert_spec_json_to_buml(system_spec)
    elements = result["model"]["elements"]
    
    # Find the method element
    method_elements = {k: v for k, v in elements.items() if v.get("type") == "ClassMethod"}
    assert len(method_elements) == 1
    
    method = list(method_elements.values())[0]
    # Void return type should result in empty string at the end
    assert method["name"].endswith(": ")


def test_convert_spec_json_to_buml_creates_relationships():
    """Test that relationships are properly converted"""
    system_spec = _mock_system_spec()
    result = convert_spec_json_to_buml(system_spec)
    
    relationships = result["model"]["relationships"]
    assert len(relationships) >= 1
    
    # Check first relationship structure
    rel = list(relationships.values())[0]
    assert "id" in rel
    assert "type" in rel
    assert "source" in rel
    assert "target" in rel
    assert rel["name"] == "owns"


def test_convert_spec_json_to_buml_handles_custom_title():
    """Test that custom title is applied"""
    system_spec = _mock_system_spec()
    result = convert_spec_json_to_buml(system_spec, title="My Custom Diagram")
    
    assert result["title"] == "My_Custom_Diagram"

# --- Tests for kg_to_plantuml ---
@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_plantuml_returns_cleaned_diagram(mock_post, mock_file):
    """Test that kg_to_plantuml successfully transforms KG to PlantUML"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_plantuml_response())
    mock_post.return_value = mock_response
    
    result = kg_to_plantuml('fake_path.ttl', 'fake_token')
    
    # Verify file was read
    mock_file.assert_called_once_with('fake_path.ttl', 'r', encoding='utf-8')
    
    # Verify API was called
    assert mock_post.called
    
    # Verify result is cleaned PlantUML
    assert result.startswith('@startuml')
    assert result.endswith('@enduml')
    assert 'class Person' in result

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_plantuml_handles_api_error(mock_post, mock_file):
    """Test that kg_to_plantuml raises error on API failure"""
    # Mock an error response
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 401
    mock_response.text = json.dumps({
        'error': {
            'message': 'Invalid API key sk-secret'
        }
    })
    mock_post.return_value = mock_response
    
    # The upstream error carries the status, never the provider's response body
    with pytest.raises(LLMUpstreamError, match="OpenAI API call failed: HTTP 401") as excinfo:
        kg_to_plantuml('fake_path.ttl', 'fake_token')
    assert "Invalid API key" not in str(excinfo.value)
    assert "sk-secret" not in str(excinfo.value)

@patch('builtins.open', side_effect=FileNotFoundError())
def test_kg_to_plantuml_raises_on_file_not_found(mock_file):
    """Test that kg_to_plantuml raises error when file doesn't exist"""
    with pytest.raises(RuntimeError, match="Failed to read KG file"):
        kg_to_plantuml('nonexistent.ttl', 'fake_token')

# --- Tests for kg_to_buml ---
@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_returns_domain_model(mock_post, mock_file):
    """The model's JSON answer becomes a DomainModel, without the web backend."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_gpt_json_response())
    mock_post.return_value = mock_response

    result = kg_to_buml('fake_path.json', 'fake_token')

    mock_file.assert_called_once_with('fake_path.json', 'r', encoding='utf-8')
    assert isinstance(result, DomainModel)
    person = result.get_class_by_name("Person")
    assert {(a.name, a.type, a.visibility) for a in person.attributes} == {
        ("name", StringType, "private"), ("age", IntegerType, "private"),
    }
    (get_name,) = person.methods
    assert (get_name.name, get_name.type) == ("getName", StringType)
    (owns,) = result.associations
    assert owns.name == "owns"
    ends = {end.type.name: end for end in owns.ends}
    assert (ends["Person"].multiplicity.min, ends["Person"].multiplicity.max) == (1, 1)
    assert ends["Car"].multiplicity.max == 9999
    assert result.validate(raise_exception=False)["success"]


def test_the_library_does_not_import_the_web_backend():
    module = importlib.import_module("besser.utilities.kg_llm_to_buml")
    with open(module.__file__, encoding="utf-8") as source:
        assert "web_modeling_editor" not in source.read()


def test_spec_to_domain_model_builds_generalizations_and_compositions():
    model = spec_to_domain_model({
        "classes": [{"className": "Order"}, {"className": "Order Line"}, {"className": "Rush-Order"}],
        "relationships": [
            {"type": "Composition", "source": "Order Line", "target": "Order", "name": "has lines",
             "sourceMultiplicity": "1..*", "targetMultiplicity": "1"},
            {"type": "Inheritance", "source": "Rush-Order", "target": "Order"},
            {"type": "Association", "source": "Order", "target": "Ghost"},
        ],
    })

    assert {c.name for c in model.get_classes()} == {"Order", "Order_Line", "Rush_Order"}
    (generalization,) = model.generalizations
    assert (generalization.specific.name, generalization.general.name) == ("Rush_Order", "Order")
    (composition,) = model.associations
    assert composition.name == "has_lines"
    ends = {end.type.name: end for end in composition.ends}
    assert ends["Order"].is_composite and not ends["Order_Line"].is_composite
    assert (ends["Order_Line"].multiplicity.min, ends["Order_Line"].multiplicity.max) == (1, 9999)


def test_spec_to_domain_model_keeps_end_and_association_names_unique():
    model = spec_to_domain_model({
        "classes": [{"className": "Loan"}, {"className": "Member"}],
        "relationships": [
            {"type": "Association", "source": "Loan", "target": "Member", "name": "borrower"},
            {"type": "Association", "source": "Loan", "target": "Member", "name": "borrower"},
        ],
    })
    assert sorted(a.name for a in model.associations) == ["borrower", "borrower_1"]
    assert model.validate(raise_exception=False)["success"]


@pytest.mark.parametrize("spec, message", [
    ({"classes": [{"className": "A", "attributes": [{"name": "x", "type": "List<String>"}]}]}, "Unknown type"),
    ({"classes": [{"className": "A"}, {"className": "B"}],
      "relationships": [{"type": "Association", "source": "A", "target": "B", "targetMultiplicity": "many"}]},
     "Invalid multiplicity"),
    ({"classes": [{"attributes": []}]}, "could not be built"),
    (["not", "an", "object"], "not a JSON object"),
])
def test_an_unusable_specification_is_an_llm_answer_error(spec, message):
    with pytest.raises(LLMAnswerError, match=message):
        spec_to_domain_model(spec)


def test_the_old_module_name_is_a_deprecated_alias():
    sys.modules.pop("besser.utilities.kg_to_buml", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = importlib.import_module("besser.utilities.kg_to_buml")
    assert any(issubclass(w.category, DeprecationWarning) and "kg_llm_to_buml" in str(w.message)
               for w in caught)
    assert alias.kg_to_buml is kg_to_buml
    assert alias.kg_to_plantuml is kg_to_plantuml


@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_handles_empty_response(mock_post, mock_file):
    """Test that kg_to_buml handles empty API responses"""
    # Mock an empty response
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": ""
                }
            }
        ]
    }
    mock_post.return_value = mock_response
    
    with pytest.raises(LLMUpstreamError, match="empty completion"):
        kg_to_buml('fake_path.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_handles_invalid_json(mock_post, mock_file):
    """Test that kg_to_buml handles invalid JSON responses"""
    # Mock a response with invalid JSON
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response("not valid json")
    mock_post.return_value = mock_response
    
    with pytest.raises(LLMAnswerError, match="Failed to parse JSON"):
        kg_to_buml('fake_path.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_handles_api_failure(mock_post, mock_file):
    """Test that kg_to_buml handles API call failures"""
    # Mock a failed API response
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response
    
    with pytest.raises(LLMUpstreamError, match="OpenAI API call failed: HTTP 401") as excinfo:
        kg_to_buml('fake_path.json', 'fake_token')
    assert "Unauthorized" not in str(excinfo.value)


@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_plantuml_always_sends_a_timeout(mock_post, mock_file):
    """The legacy path goes through the shared transport, which never posts without a timeout."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_plantuml_response())
    mock_post.return_value = mock_response

    kg_to_plantuml('fake_path.ttl', 'fake_token', timeout=7)
    assert mock_post.call_args.kwargs["timeout"] == 7

@patch('builtins.open', side_effect=IOError("Permission denied"))
def test_kg_to_buml_raises_on_file_read_error(mock_file):
    """Test that kg_to_buml raises error when file can't be read"""
    with pytest.raises(RuntimeError, match="Failed to read KG file"):
        kg_to_buml('protected.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_uses_specified_model(mock_post, mock_file):
    """Test that kg_to_buml uses the specified OpenAI model"""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_gpt_json_response())
    mock_post.return_value = mock_response
    
    kg_to_buml('fake_path.json', 'fake_token', openai_model='gpt-3.5-turbo')
    
    # Verify the model parameter in the API call
    call_args = mock_post.call_args
    payload = call_args[1]['json']
    assert payload['model'] == 'gpt-3.5-turbo'

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('requests.post')
def test_kg_to_buml_handles_conversion_failure(mock_post, mock_file):
    """An answer that is JSON but not a usable class diagram is an LLMAnswerError."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(json.dumps({
        "classes": [{"className": "A", "attributes": [{"name": "x", "type": "Map<K,V>"}]}],
    }))
    mock_post.return_value = mock_response

    with pytest.raises(LLMAnswerError, match="Unknown type") as excinfo:
        kg_to_buml('fake_path.json', 'fake_token')
    assert isinstance(excinfo.value, RuntimeError)  # what the released API raised
