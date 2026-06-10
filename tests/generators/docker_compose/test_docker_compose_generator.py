"""Tests for the DockerComposeGenerator.

Covers: empty model, locality mapping (LOCAL/EXTERNAL/HYBRID), multiplicity →
replicas, node → network, CommunicationPath → bridging network,
DeploymentDependency → depends_on, determinism, _build_view unit assertions,
yaml.safe_load validity, and the A1 end-to-end bracket-parse → replicas chain.
"""

from __future__ import annotations

import os

import pytest
import yaml

from besser.BUML.metamodel.structural import Multiplicity
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
from besser.generators.docker_compose import DockerComposeGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(model: DeploymentModel, tmp_path) -> str:
    """Generate docker-compose.yml into tmp_path and return its content."""
    gen = DockerComposeGenerator(model, output_dir=str(tmp_path))
    gen.generate()
    compose_path = os.path.join(str(tmp_path), "docker-compose.yml")
    assert os.path.isfile(compose_path), "docker-compose.yml was not created"
    with open(compose_path, encoding="utf-8") as f:
        return f.read()


def _parse(content: str) -> dict:
    """yaml.safe_load with an assertion that it returned a dict."""
    parsed = yaml.safe_load(content)
    assert isinstance(parsed, dict), f"Expected dict from yaml, got {type(parsed)}"
    return parsed


# ---------------------------------------------------------------------------
# Empty model
# ---------------------------------------------------------------------------

def test_empty_model_valid_yaml(tmp_path):
    model = DeploymentModel("empty")
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    assert "services" in parsed
    assert "version" in parsed


# ---------------------------------------------------------------------------
# Locality → build: / image:
# ---------------------------------------------------------------------------

def test_local_artifact_emits_build(tmp_path):
    node = Node("Backend", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("webapp", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["webapp"]
    assert "build" in svc
    assert svc["build"] == "./webapp"
    assert "image" not in svc


def test_external_artifact_emits_image(tmp_path):
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("redis", locality=Locality.EXTERNAL)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["redis"]
    assert "image" in svc
    assert svc["image"] == "redis:latest"
    assert "build" not in svc


def test_hybrid_artifact_emits_image_with_comment(tmp_path):
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("proxy", locality=Locality.HYBRID)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    # The # hybrid comment makes it a YAML comment; parsed value is still the image name
    parsed = _parse(content)
    svc = parsed["services"]["proxy"]
    assert svc["image"] == "proxy:latest"
    assert "build" not in svc
    # The raw text should carry the comment marker
    assert "# hybrid" in content


# ---------------------------------------------------------------------------
# Multiplicity → deploy.replicas
# ---------------------------------------------------------------------------

def test_multiplicity_max_gt_1_emits_replicas(tmp_path):
    node = Node("Cluster", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("worker", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node, multiplicity=Multiplicity(1, 3))
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["worker"]
    assert "deploy" in svc
    assert svc["deploy"]["replicas"] == 3


def test_multiplicity_1_1_no_deploy_block(tmp_path):
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("app", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node, multiplicity=Multiplicity(1, 1))
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["app"]
    assert "deploy" not in svc


# ---------------------------------------------------------------------------
# Node → network
# ---------------------------------------------------------------------------

def test_node_becomes_network(tmp_path):
    node = Node("AgentRuntime", kind=NodeKind.EXECUTION_ENVIRONMENT)
    model = DeploymentModel("m", nodes={node})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    assert "networks" in parsed
    assert "agent_runtime" in parsed["networks"]


def test_deployment_relation_joins_artifact_to_node_network(tmp_path):
    node = Node("Backend", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("api", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["api"]
    assert "backend" in svc.get("networks", [])


def test_containment_joins_artifact_to_node_network(tmp_path):
    """Nested artifact (containment, no explicit relation) also joins the node network."""
    node = Node("Backend", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("app", locality=Locality.LOCAL)
    node.add_artifact(art)
    model = DeploymentModel("m", nodes={node})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    # Artifact is only reachable via all_artifacts(); ensure it's in a service
    services = parsed.get("services") or {}
    assert "app" in services
    svc = services["app"]
    assert "backend" in svc.get("networks", [])


# ---------------------------------------------------------------------------
# CommunicationPath → bridging network
# ---------------------------------------------------------------------------

def test_communication_path_creates_link_network(tmp_path):
    node_a = Node("Alpha", kind=NodeKind.DEVICE)
    node_b = Node("Beta", kind=NodeKind.DEVICE)
    cp = CommunicationPath(node_a, node_b)
    model = DeploymentModel("m", nodes={node_a, node_b}, relationships={cp})
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    assert "networks" in parsed
    assert "alpha_beta_link" in parsed["networks"]


def test_communication_path_artifacts_join_link_network(tmp_path):
    node_a = Node("Alpha", kind=NodeKind.DEVICE)
    node_b = Node("Beta", kind=NodeKind.DEVICE)
    art_a = Artifact("svc_a", locality=Locality.LOCAL)
    art_b = Artifact("svc_b", locality=Locality.LOCAL)
    cp = CommunicationPath(node_a, node_b)
    dr_a = DeploymentRelation(art_a, node_a)
    dr_b = DeploymentRelation(art_b, node_b)
    model = DeploymentModel(
        "m",
        nodes={node_a, node_b},
        artifacts={art_a, art_b},
        relationships={cp, dr_a, dr_b},
    )
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    assert "alpha_beta_link" in parsed["services"]["svc_a"].get("networks", [])
    assert "alpha_beta_link" in parsed["services"]["svc_b"].get("networks", [])


# ---------------------------------------------------------------------------
# DeploymentDependency → depends_on
# ---------------------------------------------------------------------------

def test_deployment_dependency_emits_depends_on(tmp_path):
    node = Node("Host", kind=NodeKind.DEVICE)
    art_a = Artifact("gateway", locality=Locality.LOCAL)
    art_b = Artifact("advisor", locality=Locality.LOCAL)
    dr_a = DeploymentRelation(art_a, node)
    dr_b = DeploymentRelation(art_b, node)
    dep = DeploymentDependency(art_a, art_b)
    model = DeploymentModel(
        "m",
        nodes={node},
        artifacts={art_a, art_b},
        relationships={dr_a, dr_b, dep},
    )
    content = _generate(model, tmp_path)
    parsed = _parse(content)
    svc = parsed["services"]["gateway"]
    assert "depends_on" in svc
    assert "advisor" in svc["depends_on"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism(tmp_path):
    node = Node("Net", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art1 = Artifact("alpha", locality=Locality.LOCAL)
    art2 = Artifact("beta", locality=Locality.EXTERNAL)
    dr1 = DeploymentRelation(art1, node, multiplicity=Multiplicity(1, 2))
    dr2 = DeploymentRelation(art2, node)
    dep = DeploymentDependency(art2, art1)
    model = DeploymentModel(
        "m",
        nodes={node},
        artifacts={art1, art2},
        relationships={dr1, dr2, dep},
    )
    dir1 = str(tmp_path / "r1")
    dir2 = str(tmp_path / "r2")
    os.makedirs(dir1)
    os.makedirs(dir2)
    DockerComposeGenerator(model, output_dir=dir1).generate()
    DockerComposeGenerator(model, output_dir=dir2).generate()
    c1 = open(os.path.join(dir1, "docker-compose.yml"), encoding="utf-8").read()
    c2 = open(os.path.join(dir2, "docker-compose.yml"), encoding="utf-8").read()
    assert c1 == c2


# ---------------------------------------------------------------------------
# _build_view unit tests (assert on the resolved dicts directly)
# ---------------------------------------------------------------------------

def test_build_view_local_service_dict(tmp_path):
    node = Node("Prod", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("api", locality=Locality.LOCAL, manifests=["cid1"])
    dr = DeploymentRelation(art, node, multiplicity=Multiplicity(1, 5))
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    gen = DockerComposeGenerator(model, output_dir=str(tmp_path))
    services, networks = gen._build_view(model)

    assert len(services) == 1
    svc = services[0]
    assert svc["name"] == "api"
    assert svc["build"] == "./api"
    assert svc["image"] is None
    assert svc["replicas"] == 5
    assert "prod" in svc["networks"]
    assert svc["manifests"] == "cid1"

    assert len(networks) == 1
    assert networks[0]["name"] == "prod"


def test_build_view_communication_path_dicts(tmp_path):
    node_a = Node("Foo", kind=NodeKind.DEVICE)
    node_b = Node("Bar", kind=NodeKind.DEVICE)
    cp = CommunicationPath(node_a, node_b)
    model = DeploymentModel("m", nodes={node_a, node_b}, relationships={cp})
    gen = DockerComposeGenerator(model, output_dir=str(tmp_path))
    _, networks = gen._build_view(model)
    net_names = [n["name"] for n in networks]
    assert "foo" in net_names
    assert "bar" in net_names
    assert "foo_bar_link" in net_names


# ---------------------------------------------------------------------------
# All generated files parse with yaml.safe_load (validity sweep)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("locality", [Locality.LOCAL, Locality.EXTERNAL, Locality.HYBRID])
def test_all_localities_produce_valid_yaml(tmp_path, locality):
    node = Node("N", kind=NodeKind.DEVICE)
    art = Artifact("svc", locality=locality)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})
    content = _generate(model, tmp_path)
    parsed = yaml.safe_load(content)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# A1 end-to-end: Artifact named "Foo [3]" in WME JSON → replicas: 3
# ---------------------------------------------------------------------------

def test_a1_end_to_end_bracket_n_to_replicas(tmp_path):
    """A1 (memo §4): artifact named 'Foo [3]' in WME JSON →
    process_deployment_diagram parses [3] → DeploymentRelation.multiplicity.max=3
    → generated compose contains deploy: replicas: 3.

    This test exercises the full parse path from the real processor, not a
    hand-set multiplicity.
    """
    from besser.utilities.web_modeling_editor.backend.services.converters import (
        process_deployment_diagram,
    )

    json_data = {
        "title": "A1 swarm test",
        "model": {
            "elements": {
                "node-001": {
                    "type": "DeploymentNode",
                    "name": "Backend",
                    "stereotype": "executionEnvironment",
                },
                "artifact-001": {
                    "type": "DeploymentArtifact",
                    "name": "Foo [3]",
                    "owner": "node-001",
                },
            },
            "relationships": {},
        },
    }

    # --- chain step 1: processor parses [3] from artifact name --------------
    model = process_deployment_diagram(json_data)

    artifacts = model.all_artifacts()
    assert len(artifacts) == 1, "Expected exactly one artifact"
    foo = next(iter(artifacts))
    assert foo.name == "Foo", f"Expected clean name 'Foo', got '{foo.name}'"

    deploy_rels = [
        r for r in model.relationships
        if isinstance(r, DeploymentRelation)
    ]
    assert len(deploy_rels) == 1, "Expected one synthesised DeploymentRelation"
    assert deploy_rels[0].multiplicity.max == 3, (
        f"Expected max=3, got {deploy_rels[0].multiplicity.max}"
    )

    # --- chain step 2: generator maps multiplicity.max → replicas: 3 --------
    gen = DockerComposeGenerator(model, output_dir=str(tmp_path))
    gen.generate()

    compose_path = tmp_path / "docker-compose.yml"
    assert compose_path.exists()
    content = compose_path.read_text(encoding="utf-8")

    parsed = yaml.safe_load(content)
    services = parsed.get("services") or {}
    foo_svc = services.get("foo")
    assert foo_svc is not None, f"Service 'foo' not found in services: {list(services)}"
    deploy = foo_svc.get("deploy") or {}
    assert deploy.get("replicas") == 3, (
        f"Expected replicas=3, got {deploy.get('replicas')}"
    )


# ---------------------------------------------------------------------------
# 6b-2 — agent baking tests (the load-bearing test for this feature)
# ---------------------------------------------------------------------------

@pytest.fixture
def _researcher_agent():
    """Minimal BUML Agent model for baking tests."""
    from besser.BUML.metamodel.state_machine.agent import Agent, WebSocketPlatform
    agent = Agent("Researcher")
    agent.platforms.append(WebSocketPlatform())
    agent.new_state("initial", initial=True)
    return agent


def test_baking_produces_build_context(tmp_path, _researcher_agent):
    """6c — LOCAL artifact with resolvable agent_model_ref gets a baked build context."""
    node = Node("Cluster", kind=NodeKind.EXECUTION_ENVIRONMENT)
    art = Artifact("Researcher", locality=Locality.LOCAL)
    art.agent_model_ref = "agent-1"
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})

    gen = DockerComposeGenerator(
        model, output_dir=str(tmp_path),
        agent_models_by_id={"agent-1": _researcher_agent},
    )
    gen.generate()

    # compose file exists and references the build context
    compose_path = tmp_path / "docker-compose.yml"
    assert compose_path.exists()
    content = compose_path.read_text(encoding="utf-8")
    assert "build: ./researcher" in content

    # build context files
    ctx = tmp_path / "researcher"
    assert (ctx / "Dockerfile").exists(), "Dockerfile should be baked"
    dockerfile = (ctx / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "Researcher.py"]' in dockerfile

    assert (ctx / "Researcher.py").exists(), "BAFGenerator output should be present"


def test_baking_skips_artifact_without_agent_ref(tmp_path, _researcher_agent):
    """6c — LOCAL artifact with no agent_model_ref gets no Dockerfile baked."""
    node = Node("Cluster", kind=NodeKind.EXECUTION_ENVIRONMENT)
    agentic = Artifact("Researcher", locality=Locality.LOCAL)
    agentic.agent_model_ref = "agent-1"
    plain = Artifact("database", locality=Locality.LOCAL)
    # plain has no agent_model_ref (defaults None)
    dr1 = DeploymentRelation(agentic, node)
    dr2 = DeploymentRelation(plain, node)
    model = DeploymentModel("m", nodes={node}, artifacts={agentic, plain},
                            relationships={dr1, dr2})

    gen = DockerComposeGenerator(
        model, output_dir=str(tmp_path),
        agent_models_by_id={"agent-1": _researcher_agent},
    )
    gen.generate()

    # agentic artifact was baked
    assert (tmp_path / "researcher" / "Dockerfile").exists()
    # plain artifact was NOT baked
    assert not (tmp_path / "database" / "Dockerfile").exists()


def test_baking_no_op_when_agent_models_empty(tmp_path):
    """6c back-compat — empty agent_models_by_id (6a path) produces only compose."""
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("svc", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})

    gen = DockerComposeGenerator(model, output_dir=str(tmp_path), agent_models_by_id={})
    gen.generate()

    assert (tmp_path / "docker-compose.yml").exists()
    # No build-context subdirectory created
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs == [], f"Expected no subdirs, got {subdirs}"


def test_baking_no_op_when_no_agent_models_arg(tmp_path):
    """6c back-compat — 2-arg constructor (default agent_models_by_id=None) still works."""
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("svc", locality=Locality.LOCAL)
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})

    gen = DockerComposeGenerator(model, output_dir=str(tmp_path))
    gen.generate()

    assert (tmp_path / "docker-compose.yml").exists()
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs == [], f"Expected no subdirs, got {subdirs}"


def test_baking_skips_unresolvable_ref(tmp_path):
    """6c — artifact whose agent_model_ref has no match is skipped (no crash)."""
    node = Node("Host", kind=NodeKind.DEVICE)
    art = Artifact("ghost", locality=Locality.LOCAL)
    art.agent_model_ref = "no-such-uuid"
    dr = DeploymentRelation(art, node)
    model = DeploymentModel("m", nodes={node}, artifacts={art}, relationships={dr})

    gen = DockerComposeGenerator(
        model, output_dir=str(tmp_path),
        agent_models_by_id={"other-uuid": None},
    )
    gen.generate()

    assert (tmp_path / "docker-compose.yml").exists()
    assert not (tmp_path / "ghost" / "Dockerfile").exists()
