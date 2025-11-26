import json
import pytest
from unittest.mock import patch, mock_open, MagicMock

from besser.utilities.kg_to_buml import (
    kg_to_buml,
    kg_to_plantuml,
    clean_plantuml_response,
    clean_json_response,
    parse_json_safely,
    convert_spec_json_to_buml
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
@patch('besser.utilities.requests.post')
def test_kg_to_plantuml_returns_cleaned_diagram(mock_post, mock_file):
    """Test that kg_to_plantuml successfully transforms KG to PlantUML"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.text = json.dumps(_mock_openai_api_response(_mock_plantuml_response()))
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
@patch('besser.utilities.requests.post')
def test_kg_to_plantuml_handles_api_error(mock_post, mock_file):
    """Test that kg_to_plantuml raises error on API failure"""
    # Mock an error response
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status = 401
    mock_response.text = json.dumps({
        'error': {
            'message': 'Invalid API key'
        }
    })
    mock_post.return_value = mock_response
    
    # Function should raise RuntimeError on API error
    with pytest.raises(RuntimeError, match="Failed to process the KG to PlantUML conversion"):
        kg_to_plantuml('fake_path.ttl', 'fake_token')

@patch('builtins.open', side_effect=FileNotFoundError())
def test_kg_to_plantuml_raises_on_file_not_found(mock_file):
    """Test that kg_to_plantuml raises error when file doesn't exist"""
    with pytest.raises(RuntimeError, match="Failed to read KG file"):
        kg_to_plantuml('nonexistent.ttl', 'fake_token')

# --- Tests for kg_to_buml ---
@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('besser.utilities.kg_to_buml.requests.post')
@patch('besser.utilities.kg_to_buml.process_class_diagram')
def test_kg_to_buml_returns_domain_model(mock_process, mock_post, mock_file):
    """Test that kg_to_buml successfully transforms KG to DomainModel"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_gpt_json_response())
    mock_post.return_value = mock_response
    
    # Mock the process_class_diagram to return a mock DomainModel
    mock_domain_model = MagicMock()
    mock_domain_model.name = "KnowledgeGraphSystem"
    mock_process.return_value = mock_domain_model
    
    result = kg_to_buml('fake_path.json', 'fake_token')
    
    # Verify file was read
    mock_file.assert_called_once_with('fake_path.json', 'r', encoding='utf-8')
    
    # Verify API was called
    assert mock_post.called
    
    # Verify process_class_diagram was called
    assert mock_process.called
    
    # Verify result is a DomainModel
    assert result == mock_domain_model
    assert result.name == "KnowledgeGraphSystem"


@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('besser.utilities.requests.post')
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
    
    with pytest.raises(RuntimeError, match="Empty response from OpenAI"):
        kg_to_buml('fake_path.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('besser.utilities.requests.post')
def test_kg_to_buml_handles_invalid_json(mock_post, mock_file):
    """Test that kg_to_buml handles invalid JSON responses"""
    # Mock a response with invalid JSON
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response("not valid json")
    mock_post.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Failed to parse JSON"):
        kg_to_buml('fake_path.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('besser.utilities.requests.post')
def test_kg_to_buml_handles_api_failure(mock_post, mock_file):
    """Test that kg_to_buml handles API call failures"""
    # Mock a failed API response
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status = 401
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="OpenAI API call failed"):
        kg_to_buml('fake_path.json', 'fake_token')

@patch('builtins.open', side_effect=IOError("Permission denied"))
def test_kg_to_buml_raises_on_file_read_error(mock_file):
    """Test that kg_to_buml raises error when file can't be read"""
    with pytest.raises(RuntimeError, match="Failed to read KG file"):
        kg_to_buml('protected.json', 'fake_token')

@patch('builtins.open', new_callable=mock_open, read_data='mock kg data')
@patch('besser.utilities.requests.post')
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
@patch('besser.utilities.kg_to_buml.requests.post')
@patch('besser.utilities.kg_to_buml.process_class_diagram')
def test_kg_to_buml_handles_conversion_failure(mock_process, mock_post, mock_file):
    """Test that kg_to_buml handles failure in convert_spec_json_to_buml"""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = _mock_openai_api_response(_mock_gpt_json_response())
    mock_post.return_value = mock_response
    
    # Mock process_class_diagram to raise an exception
    mock_process.side_effect = Exception("Conversion failed")
    
    with pytest.raises(RuntimeError, match="Failed to process the KG to B-UML conversion"):
        kg_to_buml('fake_path.json', 'fake_token')

#Test kg_to_buml
test_kg_to_buml_handles_conversion_failure()
test_kg_to_buml_uses_specified_model()
test_kg_to_buml_raises_on_file_read_error()       
test_kg_to_buml_handles_api_failure()
test_kg_to_buml_handles_invalid_json()
test_kg_to_buml_handles_empty_response()
test_kg_to_buml_returns_domain_model()

#Test kg_to_plantuml
test_kg_to_plantuml_raises_on_file_not_found()
test_kg_to_plantuml_handles_api_error()
test_kg_to_plantuml_returns_cleaned_diagram()

#Test utilities
test_convert_spec_json_to_buml_handles_custom_title()
test_convert_spec_json_to_buml_creates_relationships()
test_convert_spec_json_to_buml_handles_void_return_type()
test_convert_spec_json_to_buml_creates_valid_structure()
test_parse_json_safely_invalid_json()
test_parse_json_safely_valid_json()
test_clean_json_response_strips_markdown_fence()
test_clean_plantuml_response_removes_markdown()

