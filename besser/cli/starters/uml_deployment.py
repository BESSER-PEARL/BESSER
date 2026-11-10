"""A starter B-UML model: where the parts of a library application run (a UML deployment model).

    besser validate MODEL_FILE                         check the models
    besser info MODEL_FILE                             what it holds, and the generators that read it
    besser generate docker_compose MODEL_FILE -o out   generate docker-compose.yml ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json            open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/uml_deployment.html
"""

from besser.BUML.metamodel.structural import Multiplicity
from besser.BUML.metamodel.uml_component import (
    Component,
    ComponentDependency,
    ComponentModel,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    CommunicationPath,
    DeploymentDependency,
    DeploymentModel,
    DeploymentRelation,
    Locality,
    Node,
    NodeKind,
)

# Component model: the software parts that get deployed. layout["id"] is the id an artifact refers to.
web_ui = Component(name="WebUI", layout={"id": "web_ui"})
library_api = Component(name="LibraryAPI", layout={"id": "library_api"})
database = Component(name="Database", layout={"id": "database"})
library_components = ComponentModel(
    name="LibraryComponents",
    components={web_ui, library_api, database},
    relationships={
        ComponentDependency(web_ui, library_api, name="ui_uses_api"),
        ComponentDependency(library_api, database, name="api_uses_database"),
    },
)

# Artifacts: the deployable units; manifests lists the ids of the components an artifact packages.
# A local artifact is built from source, an external one is pulled as a ready-made image.
web_ui_artifact = Artifact(name="web_ui", manifests=["web_ui"])
api_artifact = Artifact(name="library_api", manifests=["library_api"])
database_artifact = Artifact(name="postgres", locality=Locality.EXTERNAL, manifests=["database"])

# Nodes: where artifacts run (a device or an execution environment); a node can nest its artifacts
app_server = Node(name="AppServer", kind=NodeKind.EXECUTION_ENVIRONMENT, nested_artifacts={web_ui_artifact})
database_server = Node(name="DatabaseServer", kind=NodeKind.DEVICE, nested_artifacts={database_artifact})

# Relationships
relationships = {
    # An artifact deployed on a node; the upper bound of the multiplicity is the number of instances
    DeploymentRelation(api_artifact, app_server, multiplicity=Multiplicity(1, 2), name="api_on_app_server"),
    # Two nodes that can reach each other
    CommunicationPath(app_server, database_server, name="app_to_database"),
    # An artifact that needs another one to be running
    DeploymentDependency(api_artifact, database_artifact, name="api_needs_database"),
    DeploymentDependency(web_ui_artifact, api_artifact, name="ui_needs_api"),
}

# The model: nodes, the artifacts no node nests, and the relationships.
# The besser command reads every model bound to a top-level name of this file.
library_deployment = DeploymentModel(
    name="LibraryDeployment",
    nodes={app_server, database_server},
    artifacts={api_artifact},
    relationships=relationships,
)
