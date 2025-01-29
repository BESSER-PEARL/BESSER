from besser.BUML.metamodel.structural import DomainModel
from besser.utilities import image_to_plantuml, image_to_buml

# Image to BUML model
library_model: DomainModel = image_to_buml(image_path="library_hand_draw.png", openai_token="****")

# Image to PlantUML model
plantUML_model: str = image_to_plantuml(image_path="library_hand_draw.png", openai_token="****")
print(plantUML_model)