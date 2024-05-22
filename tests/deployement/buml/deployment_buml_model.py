from besser.BUML.metamodel.deployment import *
from besser.generators.deployement import DeploymentGenerator

# Deployment architecture model definition

app1 : Application = Application(name="app1", image_repo="image/latest", port=8000, required_resources=Resources(cpu=10, memory=100), domain_model="library_model")
service1 : Service = Service(name="service1", port=80, target_port=8000, protocol=Protocol.http, type=ServiceType.lb, application=app1)
container1 : Container = Container(name="container1", application=app1, resources_limit=Resources(cpu=500, memory=512))
deployment1 : Deployment = Deployment(name="deployment1", replicas=2, containers={container1, container1})
us_east1 : Region = Region(name="us-east1", zones={})
us_east2 : Region = Region(name="us-east2", zones={})

# Public cluster definition
public_cluster : PublicCluster = PublicCluster(name="public_cluster", num_nodes=3, provider=Provider.google, config_file="config_google.conf", services={service1}, deployments={deployment1}, regions={us_east1})

# Deployment architecture model definition
deployment_model : DeploymentModel = DeploymentModel(name="deployment_model", clusters={public_cluster})


generator = DeploymentGenerator(deployment_model=deployment_model)
generator.generate()