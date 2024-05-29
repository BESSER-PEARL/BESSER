from besser.BUML.notations.structuralPlantUML import plantuml_to_buml
from besser.BUML.notations.deployment import buml_deployment_model
from besser.BUML.metamodel.structural import DomainModel
from besser.generators.backend import BackendGenerator
from besser.generators.terraform import TerraformGenerator

# Structural model
dpp_buml: DomainModel = plantuml_to_buml(plantUML_model_path='dpp.plantuml')

# Deployment architecture model
deployment_model = buml_deployment_model(deployment_textfile="deployment.txt")

# Docker image and backend code generation
backend_generator = BackendGenerator(model=dpp_buml, docker_image=True, docker_config_path="config_docker.conf")
backend_generator.generate()

# Terraform code generation
terraform_generator = TerraformGenerator(deployment_model=deployment_model)
terraform_generator.generate()