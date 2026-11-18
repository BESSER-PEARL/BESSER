"""A starter B-UML model: the deployment of a containerized library application on a Google Cloud cluster.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate terraform MODEL_FILE -o out   generate Terraform files ('besser generators' lists them all)

The terraform generator reads the provider settings from the cluster's config_file (here cluster.conf, a
path relative to where the command runs); for Google Cloud that file holds one line: project = "my-gcp-project"

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/deployment.html
"""

from besser.BUML.metamodel.deployment import (
    Application,
    Container,
    Deployment,
    DeploymentModel,
    Protocol,
    Provider,
    PublicCluster,
    Region,
    Resources,
    Service,
    ServiceType,
    Zone,
)

# Application: a container image, the port it listens on, and the CPU (millicores) and memory (MB) it needs
library_app = Application(
    name="library_app",
    image_repo="example/library-app:latest",
    port=8000,
    required_resources=Resources(cpu=100, memory=256),
    domain_model=None,
)

# Container: the application at run time, with the most resources it may use
library_container = Container(name="library_container", application=library_app,
                              resources_limit=Resources(cpu=500, memory=512))

# Deployment: how many replicas of its containers the cluster keeps running
library_deployment = Deployment(name="library_deployment", replicas=2, containers={library_container})

# Service: exposes the application's port (target_port) outside the cluster, here behind a load balancer
library_service = Service(name="library_service", port=80, target_port=8000, type=ServiceType.lb,
                          protocol=Protocol.tcp, application=library_app)

# Region and zones: where the cluster runs
europe_west = Region(name="europe_west1", zones={Zone(name="europe_west1_b")})

# Cluster: a managed Kubernetes cluster of a cloud provider, with what is deployed on it
library_cluster = PublicCluster(
    name="library_cluster",
    services={library_service},
    deployments={library_deployment},
    regions={europe_west},
    num_nodes=2,
    provider=Provider.google,
    config_file="cluster.conf",
)

# The model. The besser command reads every model bound to a top-level name of this file.
library_deployment_model = DeploymentModel(name="LibraryDeployment", clusters={library_cluster})
