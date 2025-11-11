from besser.BUML.metamodel.structural import DomainModel
from besser.utilities import kg_to_plantuml, kg_to_buml

# KG to BUML model
kg_example_model: DomainModel = kg_to_buml(kg_path="kg_example.ttl", openai_token="****")

# KG to PlantUML model
plantUML_model: str = kg_to_plantuml(kg_path="kg_example.ttl", openai_token="****")
print(plantUML_model)