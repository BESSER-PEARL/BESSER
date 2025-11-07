import requests
import json
import os

from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.metamodel.structural import data_types

def kg_to_plantuml(kg_path: str, openai_token: str, openai_model: str = "gpt-4o"):
    """Transforms a Knowledge Graph (KG) into a PlantUML model. RDF and OWL KGs can be directly transformed. Requests the LLM to use data_types from besser.BUML.metamodel.structural
    
    Neo4J graphs can be exported by the following CYPHER query:
    CALL apoc.export.json.all(null, {stream:true, pretty:true})
    YIELD data
    RETURN data AS json

    Args:
        kg_path (str): the path of the kg to transform.
        openai_token (str): the OpenAI token.
        openai_model (str, optional): the OpenAI model. Defaults to "gpt-4o".

    Returns:
        plant_uml_chunk (str): the PlantUML code.
    """

    kg = ""

    with open(kg_path, "r", encoding="utf-8") as kg_file:
        kg = kg_file.read()  # plain text

    availabel_types = [d_type.name for d_type in data_types]

    prompt = f""" 
    Prompt for AI Agent:

    Task: Transform a knowledge graph into a class diagram.

    Instructions:
    Input format: You will receive a list of nodes and relations. Each node may represent either a class or an instance. Relations connect nodes and represent associations, attributes, hierarchies, or possible behaviors (methods).

    Instance handling: Nodes that are instances (e.g., Alice in (Alice, Person)) should not appear in the class diagram. Instead, identify their class (Person) and ensure the instance would be an object of that class in an implementation.

    Class identification: If a node represents a type or category (e.g., Person, Car, Book), treat it as a class in the class diagram.

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
    Only use data types among the ones listed here: {str(availabel_types)}
    All parameters from methods must be named and typed, as follows 'method(parm_name : int)'

    Goal: Generate a class diagram that can fully represent the structure and behavior implied by the knowledge graph, suitable for PlantUML rendering.

    IMPORTANT: Output ONLY the PlantUML code. Do not include explanations or metadata.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai_token
    }

    payload = {
        "model": openai_model,
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "text",
                "text": kg
            }
            ]
        }
        ],
        "max_tokens": 10000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = json.loads(response.text)
    if ('error' in response_data):
        print("An error took place during the request to transform your given KG:")
        print(response_data['error']['message'])
        return
    # Extracting the message
    messages = [choice['message'] for choice in response_data.get('choices', {})]
    message = messages[0]["content"]
    # Find the start and end positions of the desired chunk
    start_position = message.find("@startuml")
    end_position = message.find("@enduml", start_position)

    # Extract the chunk
    plant_uml_chunk = message[start_position:end_position + len("@enduml")]
    
    return plant_uml_chunk


def kg_to_buml(kg_path: str, openai_token: str, openai_model: str = "gpt-4o"):
    """Transforms a KG into a B-UML model. RDF and OWL KGs can be directly transformed. Neo4J graphs can be exported by the following CYPHER query:
    CALL apoc.export.json.all(null, {stream:true, pretty:true})
    YIELD data
    RETURN data AS json

    Args:
        kg_path (str): the path of the kg to transform.
        openai_token (str): the OpenAI token.
        openai_model (str, optional): the OpenAI model. Defaults to "gpt-4o".

    Returns:
        domain (DomainModel): the B-UML model object.
    """
    
    plant_uml_chunk = kg_to_plantuml(kg_path=kg_path, openai_token=openai_token, openai_model=openai_model)

    # Create and write to the file
    with open("kg.txt", "w") as file:
        file.write(plant_uml_chunk)    

    domain = plantuml_to_buml(plantUML_model_path="kg.txt", buml_file_path="buml_model_from_kg")
    
    os.remove("kg.txt")
    
    return domain
