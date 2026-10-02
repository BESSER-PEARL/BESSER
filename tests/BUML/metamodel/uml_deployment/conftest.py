"""Shared fixtures for the UML Deployment metamodel tests.

Three reusable, well-formed models. ``minimal_deployment_model`` and
``unp_topology_model`` validate cleanly (no errors); ``unp_topology_model``
models the UNP scenario §6 from the thesis milestone (developer
workstation, UNP sandbox, CI runner ×3, LLM host external).
"""

import pytest

from besser.BUML.metamodel.structural import Multiplicity, UNLIMITED_MAX_MULTIPLICITY
from besser.BUML.metamodel.uml_deployment import (
    Artifact, CommunicationPath, DeploymentModel, DeploymentRelation, Interface,
    InterfaceProvided, Locality, Node, NodeKind,
)


@pytest.fixture
def minimal_deployment_model() -> DeploymentModel:
    """A single ``«device»`` node hosting a single artifact."""
    artifact = Artifact("WebApp", manifests=["component_webapp_id"])
    node = Node("ServerVM", kind=NodeKind.DEVICE,
                nested_artifacts={artifact})
    return DeploymentModel(
        "minimal_deployment_model",
        nodes={node},
    )


@pytest.fixture
def unp_topology_model() -> DeploymentModel:
    """The UNP topology from the worked example: workstation, sandbox VM,
    CI runner [3], external LLM host. Code Tester [3] is the load-bearing
    proof of D2's "count belongs on Deployment" directive (R-20)."""
    # Nodes
    workstation = Node("DeveloperWorkstation", kind=NodeKind.DEVICE)
    sandbox = Node("UNPSandboxVM", kind=NodeKind.DEVICE)
    ci_runner = Node("CIRunnerCluster", kind=NodeKind.EXECUTION_ENVIRONMENT)
    llm_host = Node("LLMHost", kind=NodeKind.DEVICE, locality=Locality.EXTERNAL)

    # Artifacts
    ide_plugin = Artifact("IDEPlugin", manifests=["component_ide_plugin_id"])
    code_advisor = Artifact(
        "CodeAdvisor", manifests=["component_code_advisor_id"]
    )
    code_tester = Artifact("CodeTester", manifests=["component_code_tester_id"])
    llm_endpoint = Artifact("LLMEndpoint", locality=Locality.EXTERNAL,
                            manifests=["component_llm_endpoint_id"])

    # Containment (R-08: device / VM placement)
    workstation.add_artifact(ide_plugin)
    sandbox.add_artifact(code_advisor)

    # Deployment relations carry the multiplicity (R-06, R-20)
    deploy_tester = DeploymentRelation(
        code_tester, ci_runner, multiplicity=Multiplicity(3, 3),
        name="ci_runs_three_testers",
    )
    deploy_llm = DeploymentRelation(
        llm_endpoint, llm_host, name="llm_on_host",
    )

    # Communication paths between nodes
    ws_to_sandbox = CommunicationPath(
        workstation, sandbox, name="dev_to_sandbox"
    )
    sandbox_to_ci = CommunicationPath(
        sandbox, ci_runner, name="sandbox_to_ci"
    )
    sandbox_to_llm = CommunicationPath(
        sandbox, llm_host, name="sandbox_to_llm"
    )

    return DeploymentModel(
        "unp_topology_model",
        nodes={workstation, sandbox, ci_runner, llm_host},
        artifacts={code_tester, llm_endpoint},
        relationships={
            deploy_tester, deploy_llm,
            ws_to_sandbox, sandbox_to_ci, sandbox_to_llm,
        },
    )


@pytest.fixture
def interface_model() -> DeploymentModel:
    """A node provides an interface that another node requires."""
    server = Node("Server", kind=NodeKind.DEVICE,
                  stereotypes=["«device»"])
    client = Node("Client", kind=NodeKind.DEVICE,
                  stereotypes=["«device»"])
    iface = Interface("HTTPAPI")
    provided = InterfaceProvided(server, iface, name="server_provides")
    required = InterfaceProvided(client, iface, name="client_provides")  # same kind for variety
    # Add at least one artifact under each node so W1 (empty node) doesn't fire.
    server.add_artifact(Artifact("ServerProc", manifests=["c_server"]))
    client.add_artifact(Artifact("ClientProc", manifests=["c_client"]))
    return DeploymentModel(
        "interface_model",
        nodes={server, client},
        interfaces={iface},
        relationships={provided, required},
    )


# Re-export to keep test imports terse.
__all__ = [
    "minimal_deployment_model",
    "unp_topology_model",
    "interface_model",
    "UNLIMITED_MAX_MULTIPLICITY",
]
