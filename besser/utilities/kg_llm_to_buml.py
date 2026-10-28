"""LLM-assisted conversion of an arbitrary knowledge graph into B-UML.

The model reads the raw KG text (Turtle, RDF/XML, or a Neo4j JSON export) and
proposes a class diagram, returned as PlantUML (:func:`kg_to_plantuml`) or as a
B-UML :class:`~besser.BUML.metamodel.structural.DomainModel`
(:func:`kg_to_buml`). It needs an OpenAI key, and its output is a best-effort
interpretation.

For a proper OWL/SHACL ontology, the deterministic, offline conversion is
:func:`besser.BUML.notations.kg_to_buml.kg_to_class_diagram`.

This module used to be ``besser.utilities.kg_to_buml``, which is kept as a
deprecated alias of it.
"""

from typing import Any, Callable, Dict, Optional, Set

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Method,
    Multiplicity,
    Parameter,
    PrimitiveDataType,
    Property,
    Type,
    UNLIMITED_MAX_MULTIPLICITY,
    data_types,
)
from besser.utilities.llm_utils import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    LLMUpstreamError,
    clean_json_response,
    parse_json_safely,
    post_openai_chat,
)

__all__ = [
    "LLMAnswerError",
    "clean_json_response",
    "clean_plantuml_response",
    "convert_spec_json_to_buml",
    "kg_to_buml",
    "kg_to_plantuml",
    "parse_json_safely",
    "spec_to_domain_model",
]

# Output budget of the single completion each conversion asks for.
_MAX_COMPLETION_TOKENS = 10000


class LLMAnswerError(RuntimeError):
    """The model answered, but its answer is not a usable class diagram.

    Distinct from :class:`~besser.utilities.llm_utils.LLMUpstreamError` (the
    call itself failed) and from a plain :class:`RuntimeError` (the KG file
    could not be read). The message may quote the model's answer, so it is meant
    for logs rather than an end user. Subclasses :class:`RuntimeError`, which is
    what :func:`kg_to_buml` raised for this case before the class existed.
    """


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
    Convert class specifications from GPT-generated JSON to the web editor's
    class-diagram JSON (``{"title", "model": {"nodes", "edges"}}``).

    :func:`kg_to_buml` does not go through this JSON: it builds the model with
    :func:`spec_to_domain_model`. Kept for callers of the released API.
    """

    nodes = []
    edges = []

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

        # Create class node (v4 shape); attribute/method rows carry the
        # "+ name: type" signature, parsed by the class diagram processor.
        class_data = {
            "name": cls["className"],
            "stereotype": None,
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

            class_data["attributes"].append({"id": attr_id, "name": name_str})

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

            class_data["methods"].append({"id": method_id, "name": name_str})

        nodes.append({
            "id": class_id,
            "type": "class",
            "position": {"x": 0, "y": 0},
            "data": class_data
        })

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

        edges.append({
            "id": rel_id,
            "type": converted_type,
            "source": source_class_id,
            "target": target_class_id,
            "data": {
                "name": rel.get("name", ""),
                "sourceMultiplicity": rel.get("sourceMultiplicity", "1"),
                "sourceRole": rel.get("sourceRole", ""),
                "targetMultiplicity": rel.get("targetMultiplicity", "1"),
                "targetRole": rel.get("name", "")
            }
        })

    # === Final structure ===
    besser_buml_json = {
        "title": title.replace(" ", "_"),
        "model": {
            "nodes": nodes,
            "edges": edges
        }
    }

    return besser_buml_json


_VISIBILITIES = {"public", "private", "protected", "package"}
_PRIMITIVES: Dict[str, PrimitiveDataType] = {t.name: t for t in data_types}
# Spellings models use for the primitives, beyond BESSER's own names.
_PRIMITIVE_ALIASES = {
    "string": "str", "integer": "int", "long": "int", "double": "float",
    "decimal": "float", "boolean": "bool",
}


def _identifier(text: Any) -> str:
    """``text`` as a B-UML name: trimmed, with spaces and hyphens made ``_``."""
    return "_".join(str(text).replace("-", " ").split())


def _visibility(value: Any) -> str:
    return value if value in _VISIBILITIES else "public"


def _resolve_type(name: Any, classes: Dict[str, Class]) -> Type:
    """A class of the model, else a primitive (case-insensitive, common aliases)."""
    text = str(name or "").strip()
    if text in classes:
        return classes[text]
    key = _PRIMITIVE_ALIASES.get(text.lower(), text.lower())
    if key in _PRIMITIVES:
        return _PRIMITIVES[key]
    raise LLMAnswerError(f"Unknown type '{text}': neither a class of the diagram nor a primitive type.")


def _multiplicity(text: Any) -> Multiplicity:
    """Parse ``"1"``, ``"0..1"``, ``"1..*"`` or ``"*"``; an empty value is ``1``."""
    value = str(text or "1").strip()
    if value == "*":
        return Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)
    low, dots, high = value.partition("..")
    try:
        minimum = int(low) if low and low != "*" else 0
        if not dots:
            return Multiplicity(minimum, minimum)
        maximum = UNLIMITED_MAX_MULTIPLICITY if high in ("", "*") else int(high)
    except ValueError as exc:
        raise LLMAnswerError(f"Invalid multiplicity '{value}'.") from exc
    return Multiplicity(minimum, maximum)


def _unique(name: str, taken: Set[str]) -> str:
    """``name``, suffixed ``_1``, ``_2``... until it is not in ``taken``; then taken."""
    candidate, counter = name, 1
    while candidate in taken:
        candidate = f"{name}_{counter}"
        counter += 1
    taken.add(candidate)
    return candidate


def spec_to_domain_model(system_spec: Dict[str, Any], name: str = "KG_Imported_Diagram") -> DomainModel:
    """Build a :class:`DomainModel` from the JSON class specification the LLM returns.

    The specification is the one :func:`kg_to_buml` asks for: ``classes`` (each
    with ``className``, ``attributes`` and ``methods``) and ``relationships``
    (``type`` association / composition / aggregation / inheritance,
    ``source``, ``target``, the two multiplicities and a ``name``). Names are
    made valid B-UML names (spaces and hyphens become ``_``), duplicate
    association and end names are suffixed, and a relationship that names a
    class the specification does not define is skipped. A composition's
    ``target`` is the whole, as in the editor's class diagrams; an inheritance
    goes from the specific ``source`` to the general ``target``.

    Raises:
        LLMAnswerError: the specification is not a JSON object, or uses an
            unknown type, an invalid multiplicity, or a name B-UML refuses.
    """
    if not isinstance(system_spec, dict):
        raise LLMAnswerError("The class specification is not a JSON object.")
    try:
        return _build_domain_model(system_spec, name)
    except LLMAnswerError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LLMAnswerError(f"The class specification could not be built into a model: {exc}") from exc


def _build_domain_model(system_spec: Dict[str, Any], name: str) -> DomainModel:
    class_specs = [c for c in system_spec.get("classes") or [] if isinstance(c, dict)]
    # Keyed by the name as written and as sanitised, so either spelling resolves.
    classes: Dict[str, Class] = {}
    for spec in class_specs:
        cls = Class(name=_identifier(spec["className"]))
        classes[spec["className"]] = classes[cls.name] = cls

    for spec in class_specs:
        cls = classes[spec["className"]]
        for attr in spec.get("attributes") or []:
            cls.add_attribute(Property(
                name=_identifier(attr["name"]),
                type=_resolve_type(attr.get("type") or "str", classes),
                visibility=_visibility(attr.get("visibility")),
            ))
        for method in spec.get("methods") or []:
            return_type = str(method.get("returnType") or "").strip()
            cls.add_method(Method(
                name=_identifier(method["name"]),
                visibility=_visibility(method.get("visibility")),
                parameters=[
                    Parameter(name=_identifier(p["name"]), type=_resolve_type(p.get("type") or "str", classes))
                    for p in method.get("parameters") or []
                ],
                type=None if return_type.lower() in ("", "void", "none") else _resolve_type(return_type, classes),
            ))

    associations: Set[BinaryAssociation] = set()
    generalizations: Set[Generalization] = set()
    association_names: Set[str] = set()
    # End names already used per owning class (the class on the other side).
    end_names: Dict[str, Set[str]] = {}
    for rel in system_spec.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        source = classes.get(rel.get("sourceClass") or rel.get("source"))
        target = classes.get(rel.get("targetClass") or rel.get("target"))
        if source is None or target is None:
            continue
        kind = str(rel.get("type") or "").lower()
        if kind in ("inheritance", "generalization"):
            generalizations.add(Generalization(general=target, specific=source))
            continue
        source_end = Property(
            name=_unique(_identifier(rel.get("sourceRole") or source.name.lower()),
                         end_names.setdefault(target.name, set())),
            type=source,
            multiplicity=_multiplicity(rel.get("sourceMultiplicity")),
        )
        target_end = Property(
            name=_unique(_identifier(target.name.lower()), end_names.setdefault(source.name, set())),
            type=target,
            multiplicity=_multiplicity(rel.get("targetMultiplicity")),
            is_composite=kind == "composition",
        )
        association_name = _identifier(rel.get("name") or f"{source.name}_{target.name}")
        associations.add(BinaryAssociation(
            name=_unique(association_name, association_names), ends={source_end, target_end},
        ))

    return DomainModel(
        name=_identifier(name) or "KG_Imported_Diagram",
        types=set(classes.values()),
        associations=associations,
        generalizations=generalizations,
    )


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
    LLMAnswerError
        The model's answer could not be turned into a class diagram (a
        :class:`RuntimeError` subclass).
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

    system_spec = parse_json_safely(clean_json_response(raw_content))
    if not system_spec:
        raise LLMAnswerError("Failed to parse JSON from the model's answer.")
    return spec_to_domain_model(system_spec)
