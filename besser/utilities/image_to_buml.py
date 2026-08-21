import requests
import json
import base64
import os

from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.structuralPlantUML import plantuml_to_buml

UNLIMITED_MULTIPLICITY = 9999


def _describe_domain_model(dm: DomainModel) -> str:
    """Compact textual summary of a DomainModel for use in LLM prompts."""
    def _mult(m):
        if m is None:
            return ""
        hi = "*" if m.max == UNLIMITED_MULTIPLICITY else m.max
        return f"{m.min}..{hi}"

    classes = []
    for cls in sorted(dm.get_classes(), key=lambda c: c.name):
        attrs = ", ".join(
            f"{a.name}: {getattr(a.type, 'name', str(a.type))}"
            for a in sorted(cls.attributes, key=lambda p: p.name)
        )
        methods = ", ".join(f"{m.name}()" for m in sorted(cls.methods, key=lambda m: m.name))
        parts = [cls.name]
        if attrs:
            parts.append(f"attrs: {attrs}")
        if methods:
            parts.append(f"methods: {methods}")
        classes.append(" | ".join(parts))

    associations = []
    for assoc in sorted(dm.associations, key=lambda a: a.name or ""):
        ends = list(assoc.ends)
        if len(ends) == 2:
            a, b = ends
            associations.append(
                f"{a.type.name} {_mult(a.multiplicity)} <-> {_mult(b.multiplicity)} {b.type.name}"
                + (f" ({assoc.name})" if assoc.name else "")
            )

    generalizations = [
        f"{g.specific.name} --|> {g.general.name}" for g in dm.generalizations
    ]

    sections = []
    if classes:
        sections.append("Classes:\n  - " + "\n  - ".join(classes))
    if associations:
        sections.append("Associations:\n  - " + "\n  - ".join(associations))
    if generalizations:
        sections.append("Inheritance:\n  - " + "\n  - ".join(generalizations))
    return "\n".join(sections) if sections else "(empty model)"


def image_to_plantuml(image_path: str, openai_token: str, existing_model: DomainModel = None,
                      openai_model: str = "gpt-4o"):
    """Transforms an image into a PlantUML model.

    Args:
        image_path (str): the path of the image to transform.
        openai_token (str): the OpenAI token.
        existing_model (DomainModel, optional): an existing B-UML model to extend.
            When provided, the LLM is instructed to produce a single merged PlantUML
            diagram that preserves the existing classes/associations and adds what
            the image introduces. Classes matched by name are kept; only new
            attributes/methods/relationships are appended.
        openai_model (str, optional): the OpenAI model. Defaults to "gpt-4o".

    Returns:
        plant_uml_chunk (str): the PlantUML code.
    """
    base64_image = ""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    if existing_model is not None and existing_model.get_classes():
        text = (
            "I already have a UML class diagram and I want to extend it using the "
            "attached hand-drawn diagram. Produce ONE merged PlantUML class diagram "
            "that contains both sides.\n\n"
            "Merge rules:\n"
            "- For classes with the same name on both sides, keep the existing one. "
            "Only add attributes/methods/relationships that the image introduces and "
            "that are not already present.\n"
            "- Keep ALL existing classes, attributes, methods, and associations, even "
            "if they are not in the image.\n"
            "- Add every new class and relationship shown in the image.\n"
            "- Preserve original names exactly.\n"
            "- If an association end has two labels, omit one — PlantUML does not "
            "support naming both ends.\n\n"
            "Existing diagram:\n"
            f"{_describe_domain_model(existing_model)}"
        )
    else:
        text = (
            "Can you turn this drawn uml class diagram into the corresponding "
            "PlantUML class diagram? The diagram I sent might have some weird "
            "relations or classes, I want you to keep them as they are. If the "
            "ends of an association have two labels, omit one of the labels, as "
            "PlantUML does not support naming the ends."
        )
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
                "text": text
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = json.loads(response.text)
    if ('error' in response_data):
        print("An error took place during the request to transform your given image:")
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
    #
    return plant_uml_chunk


def image_to_buml(image_path: str, openai_token: str, existing_model: DomainModel = None,
                  openai_model: str = "gpt-4o"):
    """Transforms an image into a B-UML model.

    Args:
        image_path (str): the path of the image to transform.
        openai_token (str): the OpenAI token.
        existing_model (DomainModel, optional): an existing B-UML model to extend
            with the classes and relationships shown in the image. Same-name classes
            are merged; new ones are added. When omitted, behaves like before and
            returns a fresh model from the image alone.
        openai_model (str, optional): the OpenAI model. Defaults to "gpt-4o".

    Returns:
        domain (DomainModel): the B-UML model object.
    """
    plant_uml_chunk = image_to_plantuml(
        image_path=image_path,
        openai_token=openai_token,
        existing_model=existing_model,
        openai_model=openai_model,
    )
    # Create and write to the file
    with open("image.txt", "w") as file:
        file.write(plant_uml_chunk)
    domain = plantuml_to_buml(plantUML_model_path="image.txt", buml_file_path="buml_model_from_image")
    os.remove("image.txt")
    return domain
