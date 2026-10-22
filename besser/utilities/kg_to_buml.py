from typing import Any, Callable, Optional

from besser.BUML.metamodel.structural import data_types
from besser.BUML.metamodel.structural import DomainModel
from besser.utilities.llm_utils import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    LLMUpstreamError,
    clean_json_response,
    parse_json_safely,
    post_openai_chat,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import process_class_diagram

# Output budget of the single completion each conversion asks for.
_MAX_COMPLETION_TOKENS = 10000


def _ask_openai(
    openai_token: str,
    openai_model: str,
    prompt: str,
    kg: str,
    *,
    timeout: float,
    http_post: Optional[Callable[..., Any]],
) -> str:
    """Send the conversion prompt plus the raw KG text; return the non-empty reply.

    Goes through :func:`~besser.utilities.llm_utils.post_openai_chat`, so the
    request has a timeout and a failure is an :class:`LLMUpstreamError` whose
    message never carries the provider's response body.
    """
    payload = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": kg},
                ],
            }
        ],
        "max_completion_tokens": _MAX_COMPLETION_TOKENS,
    }
    content = post_openai_chat(openai_token, payload, http_post=http_post, timeout=timeout)
    if not content:
        raise LLMUpstreamError("The model returned an empty completion.")
    return content


def clean_plantuml_response(response: str) -> str:
    """Clean PlantUML response from LLM (remove markdown formatting)"""
    start_position = response.find("@startuml")
    end_position = response.find("@enduml", start_position)
    return response[start_position:end_position + len("@enduml")]

def convert_spec_json_to_buml(system_spec, title="KG Imported Diagram"):
    """
    Convert class specifications from GPT-generated JSON to Apollon/BUML format.
    """

    elements = {}
    relationships = {}

    # ID generation helpers
    def make_id(prefix, counter):
        return f"{prefix}-{counter}"

    class_map = {}
    class_counter = 1
    attr_counter = 1
    method_counter = 1
    rel_counter = 1

    # === Classes and their attributes/methods ===
    for cls in system_spec.get("classes", []):
        class_id = make_id("class", class_counter)
        class_map[cls["className"]] = class_id
        class_counter += 1

        # Create class element
        class_element = {
            "id": class_id,
            "type": "Class",
            "name": cls["className"],
            "attributes": [],
            "methods": []
        }

        # --- Attributes ---
        for attr in cls.get("attributes", []):
            attr_id = make_id("attr", attr_counter)
            attr_counter += 1

            visibility = attr.get("visibility", "public")
            visibility_symbol = (
                "+" if visibility == "public" else
                "-" if visibility == "private" else
                "#"
            )
            name_str = f"{visibility_symbol} {attr['name']}: {attr['type']}"

            attr_element = {
                "id": attr_id,
                "type": "ClassAttribute",
                "owner": class_id,
                "name": name_str
            }

            class_element["attributes"].append(attr_id)
            elements[attr_id] = attr_element

        # --- Methods ---
        for method in cls.get("methods", []):
            method_id = make_id("method", method_counter)
            method_counter += 1

            visibility = method.get("visibility", "public")
            visibility_symbol = (
                "+" if visibility == "public" else
                "-" if visibility == "private" else
                "#"
            )

            params = method.get("parameters", [])
            param_str = ", ".join([f"{p['name']}: {p['type']}" for p in params])

            return_type = method.get("returnType", "")
            if return_type.lower() == "void":
                return_type = ""   # 🔧 fix here

            name_str = f"{visibility_symbol} {method['name']}({param_str}): {return_type}"

            method_element = {
                "id": method_id,
                "type": "ClassMethod",
                "owner": class_id,
                "name": name_str
            }

            class_element["methods"].append(method_id)
            elements[method_id] = method_element

        elements[class_id] = class_element

    # === Relationships ===
    for rel in system_spec.get("relationships", []):
        rel_id = make_id("rel", rel_counter)
        rel_counter += 1

        rel_type = rel.get("type", "").lower()
        if rel_type in ["inheritance", "generalization"]:
            converted_type = "ClassInheritance"
        elif rel_type == "composition":
            converted_type = "ClassComposition"
        elif rel_type == "aggregation":
            converted_type = "ClassAggregation"
        else:
            converted_type = "ClassBidirectional"

        source_class_id = class_map.get(rel.get("sourceClass") or rel.get("source"))
        target_class_id = class_map.get(rel.get("targetClass") or rel.get("target"))

        if not source_class_id or not target_class_id:
            continue  # skip invalid relationships

        relationship_obj = {
            "id": rel_id,
            "type": converted_type,
            "source": {
                "element": source_class_id,
                "multiplicity": rel.get("sourceMultiplicity", "1"),
                "role": rel.get("sourceRole", "")
            },
            "target": {
                "element": target_class_id,
                "multiplicity": rel.get("targetMultiplicity", "1"),
                "role": rel.get("name", "")
            },
            "name": rel.get("name", "")
        }

        relationships[rel_id] = relationship_obj

    # === Final structure ===
    apollon_buml_json = {
        "title": title.replace(" ", "_"),
        "model": {
            "elements": elements,
            "relationships": relationships
        }
    }

    return apollon_buml_json


def kg_to_plantuml(
    kg_path: str,
    openai_token: str,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    http_post: Optional[Callable[..., Any]] = None,
) -> str:
    """
    Transform a Knowledge Graph (KG) into a PlantUML class diagram.

    **Supported inputs:**

    - Turtle files (.ttl)
    - RDF KGs (.rdf files)
    - Neo4j graphs (JSON) exported using the Cypher command below::

        CALL apoc.export.json.all(null, {stream:true, pretty:true})
        YIELD data
        RETURN data AS json

    The function extracts classes, attributes, methods, and associations from the KG
    and asks an LLM to produce a complete PlantUML class diagram using only the
    data types defined in `besser.BUML.metamodel.structural`.

    Parameters
    ----------
    kg_path : str
        Path to the input Knowledge Graph file (raw JSON, TTL, RDF).
    openai_token : str
        Your OpenAI API token.
    openai_model : str, optional
        Model name to use, by default :data:`~besser.utilities.llm_utils.DEFAULT_OPENAI_MODEL`.
    timeout : float, optional
        Seconds before the OpenAI request is abandoned.
    http_post : callable, optional
        Transport override for tests; ``None`` uses :func:`requests.post`.

    Returns
    -------
    str
        A cleaned PlantUML diagram ready to render.

    Raises
    ------
    LLMUpstreamError
        The OpenAI call failed or returned nothing (the message never contains
        the provider's response body).
    RuntimeError
        The KG file could not be read.
    """


    kg = ""

    try:
        with open(kg_path, "r", encoding="utf-8") as kg_file:
            kg = kg_file.read()  # plain text
    except Exception as e:
        raise RuntimeError(f"Failed to read KG file at {kg_path}: {e}") from e

    available_types = [d_type.name for d_type in data_types]

    prompt = f"""
    Prompt for AI Agent:

    Task: Transform a knowledge graph into a class diagram.

    Instructions:
    Input format: You will receive a list of nodes and relations. Each node may represent either a class or an instance. Relations connect nodes and represent associations, attributes, hierarchies, or possible behaviors (methods).

    Instance handling: Nodes that are instances (e.g., Alice in (Alice, Person)) should not appear in the class diagram. Instead, identify their class (Person) even if not explicity in the graph and ensure the instance would be an object of that class in an implementation.

    Class identification: If a node represents a type or category (e.g., Person, Car, Book), treat it as a class in the class diagram. If multiple classes share the same attributes, further confirm they are not or should be considered instances of the same super class.

    Relation handling: For each relation between nodes:
    If it connects two classes (e.g., Person --owns--> Car), create an association between the classes. Include cardinality if it can be inferred.
    If it represents an attribute (e.g., (Person, hasAge, Integer)), create an attribute in the corresponding class.
    If it represents a behavior or action (e.g., (Person, canDrive, Car)), create a method in the corresponding class. The method name can be derived from the relation, with parameters inferred from the target node if applicable.
    If it connects an instance to a class, ignore it.

    Output format: Produce a complete PlantUML class diagram representing:
    All identified classes
    Attributes and methods for each class
    Associations between classes with cardinality where derivable
    Instances should not appear
    Only use data types among the ones listed here: {str(available_types)}
    All parameters from methods must be named and typed, as follows 'method(parm_name : int)'
    All associations must have different names
    If associations are named the same in graph, modify their name so they are different in the generated model

    Goal: Generate a class diagram that can fully represent the structure and behavior implied by the knowledge graph, suitable for PlantUML rendering.

    IMPORTANT: Output ONLY the PlantUML code. Do not include explanations or metadata.
    """

    message = _ask_openai(openai_token, openai_model, prompt, kg, timeout=timeout, http_post=http_post)

    plant_uml_chunk = clean_plantuml_response(message)
    return plant_uml_chunk

def kg_to_buml(
    kg_path: str,
    openai_token: str,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    *,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    http_post: Optional[Callable[..., Any]] = None,
) -> DomainModel:
    """
    Convert a Knowledge Graph (KG) into a B-UML class diagram.

    **Supported inputs**

    - Turtle files (.ttl)
    - RDF KGs (.rdf files)
    - Neo4j graphs (JSON) exported using the Cypher command below::

        CALL apoc.export.json.all(null, {stream:true, pretty:true})
        YIELD data
        RETURN data AS json

    The function reads the KG (TTL/RDF or Neo4j JSON export), extracts classes,
    attributes, methods, and associations, and uses an LLM to generate a DomainModel object
    defined in `besser.BUML.metamodel.structural`.

    Parameters
    ----------
    kg_path : str
        Path to the input Knowledge Graph file.
    openai_token : str
        Your OpenAI API token.
    openai_model : str, optional
        Model name to use, by default :data:`~besser.utilities.llm_utils.DEFAULT_OPENAI_MODEL`.
    timeout : float, optional
        Seconds before the OpenAI request is abandoned.
    http_post : callable, optional
        Transport override for tests; ``None`` uses :func:`requests.post`.

    Returns
    -------
    domain (DomainModel): the B-UML model object.

    Raises
    ------
    LLMUpstreamError
        The OpenAI call failed or returned nothing (the message never contains
        the provider's response body).
    RuntimeError
        The KG file could not be read, or the model's answer could not be
        turned into a class diagram.
    """


    kg = ""

    try:
        with open(kg_path, "r", encoding="utf-8") as kg_file:
            kg = kg_file.read()  # plain text
    except Exception as e:
        raise RuntimeError(f"Failed to read KG file at {kg_path}: {e}") from e

    available_types = [d_type.name for d_type in data_types]

    prompt = f"""
    Prompt for AI Agent:

    Task: Transform a knowledge graph into a class diagram.

    Instructions:
    Input format: You will receive a list of nodes and relations. Each node may represent either a class or an instance. Relations connect nodes and represent associations, attributes, hierarchies, or possible behaviors (methods).

    Instance handling: Nodes that are instances (e.g., Alice in (Alice, Person)) should not appear in the class diagram. Instead, identify their class (Person) even if not explicity in the graph and ensure the instance would be an object of that class in an implementation.

    Class identification: If a node represents a type or category (e.g., Person, Car, Book), treat it as a class in the class diagram. If multiple classes share the same attributes, further confirm they are not or should be considered instances of the same super class.

    Relation handling: For each relation between nodes:
    If it connects two classes (e.g., Person --owns--> Car), create an association between the classes. Include cardinality if it can be inferred.
    If it represents an attribute (e.g., (Person, hasAge, Integer)), create an attribute in the corresponding class.
    If it represents a behavior or action (e.g., (Person, canDrive, Car)), create a method in the corresponding class. The method name can be derived from the relation, with parameters inferred from the target node if applicable.
    If it connects an instance to a class, ignore it.

    Output format:
    Return ONLY a JSON object with this structure:
        {{
        "systemName": "SystemName",
        "classes": [
            {{
            "className": "ClassName",
            "attributes": [
                {{"name": "attr", "type": "String", "visibility": "public"}}
            ],
            "methods": [
                {{"name": "method", "returnType": "void", "visibility": "public", "parameters": [
                {{"name": "param", "type": "String"}}
                ]}}
            ]
            }}
        ],
        "relationships": [
            {{
            "type": "Association",
            "source": "ClassName1",
            "target": "ClassName2",
            "sourceMultiplicity": "1",
            "targetMultiplicity": "*",
            "name": "relationshipName"
            }}
        ]
        }}


    - All identified classes
    - Attributes and methods for each class
    - Associations between classes with cardinality where derivable
    - Instances should not appear
    - visibility: "public", "private", "protected", or "package" (default: public for attributes, public for methods)
    - Only use data types among the ones listed here: {str(available_types)}

    Goal: Generate a class diagram that can fully represent the structure and behavior implied by the knowledge graph, using the JSON structure provided.

    IMPORTANT: Return ONLY the JSON, no explanations.
    """

    raw_content = _ask_openai(openai_token, openai_model, prompt, kg, timeout=timeout, http_post=http_post)

    try:
        # --- Clean and parse JSON ---
        json_text = clean_json_response(raw_content)
        system_spec = parse_json_safely(json_text)

        if not system_spec:
            raise ValueError("Failed to parse JSON from OpenAI response.")

        buml_json = convert_spec_json_to_buml(system_spec)

        if not buml_json:
            raise ValueError("Failed to convert OpenAI generated json to B-UML format.")

        domain_model: DomainModel = process_class_diagram(buml_json)

        return domain_model

    except Exception as e:
        raise RuntimeError(f"Failed to process the KG to B-UML conversion: {e}") from e
