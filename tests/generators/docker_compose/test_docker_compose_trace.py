"""DockerComposeGenerator records which lines of docker-compose.yml come from which artifact or node, the
Dockerfile of each baked agent from its artifact, and merges the traces of the BAF generators it runs."""

import os

from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, WebSocketPlatform
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.BUML.metamodel.structural import Multiplicity
from besser.BUML.metamodel.uml_deployment import (
    Artifact, CommunicationPath, DeploymentDependency, DeploymentModel, DeploymentRelation, Locality, Node, NodeKind,
)
from besser.generators.docker_compose import DockerComposeGenerator


def _agent(name, with_ids, **a2a):
    agent = Agent(name)
    agent.platforms.append(WebSocketPlatform())
    state = agent.new_state("initial", initial=True)
    state.set_body(Body(f"{name}_body", actions=[AgentReply("hi")]))
    if a2a:
        agent._a2a = {"outbound": a2a.get("outbound", []), "inbound": a2a.get("inbound", [])}
    if with_ids:
        agent.id, state.id, state.body.id = f"agent-{name}", f"state-{name}", f"body-{name}"
    return agent


def _model(with_ids: bool):
    def ident(element, value):
        if with_ids:
            element.id = value
        return element

    cluster = ident(Node("Cluster", kind=NodeKind.EXECUTION_ENVIRONMENT), "node-cluster")
    edge = ident(Node("Edge Box", kind=NodeKind.DEVICE), "node-edge")
    supervisor = ident(Artifact("AgentSupervisor", locality=Locality.LOCAL), "art-sup")
    supervisor.agent_model_ref = "sup"
    coder = ident(Artifact("AgentCoder", locality=Locality.LOCAL), "art-coder")
    coder.agent_model_ref = "coder"
    redis = ident(Artifact("redis", locality=Locality.EXTERNAL), "art-redis")
    relationships = {DeploymentRelation(supervisor, cluster), DeploymentRelation(coder, cluster, multiplicity=Multiplicity(1, 3)),
                     DeploymentRelation(redis, edge), DeploymentDependency(coder, redis), CommunicationPath(cluster, edge)}
    model = DeploymentModel("m", nodes={cluster, edge}, artifacts={supervisor, coder, redis}, relationships=relationships)
    agents = {
        "sup": _agent("AgentSupervisor", with_ids, outbound=[
            {"peer": "AgentCoder", "ref": "coder", "order": 1, "kind": "delegates", "state": "initial"}]),
        "coder": _agent("AgentCoder", with_ids, inbound=[
            {"peer": "AgentSupervisor", "ref": "sup", "order": 9999, "kind": "delegates"}]),
    }
    return model, agents


def _generate(with_ids, out):
    model, agents = _model(with_ids)
    generator = DockerComposeGenerator(model, output_dir=str(out), agent_models_by_id=agents)
    generator.generate()
    files = {}
    for root, _, names in os.walk(str(out)):
        for name in names:
            with open(os.path.join(root, name), encoding="utf-8", newline="") as f:
                files[os.path.relpath(os.path.join(root, name), str(out)).replace(os.sep, "/")] = f.read()
    return generator, files


def _text(files, path, region):
    return files[path].split("\n")[region.start_line - 1:region.end_line]


def test_services_and_networks_are_regions_of_the_compose_file(tmp_path):
    generator, files = _generate(True, tmp_path)
    regions = {(r.element_id, r.rule.split("#")[1]): r for r in generator.traced_files["docker-compose.yml"].regions}

    coder = _text(files, "docker-compose.yml", regions[("art-coder", "service")])
    assert coder[0] == "  agent_coder:"
    assert "    build: ./agent_coder" in coder and "      replicas: 3" in coder and "    depends_on: [redis]" in coder
    assert _text(files, "docker-compose.yml", regions[("art-redis", "service")])[0] == "  redis:"
    assert _text(files, "docker-compose.yml", regions[("node-cluster", "network")]) == [
        "  cluster: {}  # kind: executionEnvironment"]
    assert _text(files, "docker-compose.yml", regions[("node-edge", "network")]) == [
        "  edge_box: {}  # kind: device"]
    # A service block holds only its own lines; the bridging network of the communication path has no element.
    assert all(not line.startswith("  redis") for line in coder)
    assert len([r for r in generator.traced_files["docker-compose.yml"].regions if r.rule.endswith("#network")]) == 2


def test_each_baked_agent_has_a_dockerfile_from_its_artifact_and_its_baf_traces(tmp_path):
    generator, files = _generate(True, tmp_path)
    traced = generator.traced_files
    for service, artifact, agent in (("agent_supervisor", "art-sup", "AgentSupervisor"),
                                     ("agent_coder", "art-coder", "AgentCoder")):
        dockerfile = traced[f"{service}/Dockerfile"]
        assert [(s.element.element_id, s.rule) for s in dockerfile.sources] == [(artifact, "Dockerfile.j2#dockerfile")]
        script = traced[f"{service}/{agent}.py"]
        assert [s.element.element_id for s in script.sources] == [f"agent-{agent}"]
        regions = {(r.element_id, r.rule.split("#")[1]): r for r in script.regions}
        assert _text(files, f"{service}/{agent}.py", regions[(f"state-{agent}", "state")]) == [
            "initial = agent.new_state('initial', initial=True)"]
        assert _text(files, f"{service}/{agent}.py", regions[(f"body-{agent}", "body")])[0] == f"def {agent}_body(session: Session):"
        assert f"{service}/config.yaml" in traced
    assert set(traced) <= set(files)
    assert not generator.ai_files


def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path):
    traced, traced_files = _generate(True, tmp_path / "traced")
    untraced, untraced_files = _generate(False, tmp_path / "untraced")
    assert traced_files == untraced_files
    assert not any("\x02" in text or "\x03" in text for text in traced_files.values())
    assert set(untraced.traced_files) == set(traced.traced_files)
    assert all(trace.regions == () and trace.sources == () for trace in untraced.traced_files.values())


def test_a_compose_only_run_traces_only_the_compose_file(tmp_path):
    model, _ = _model(True)
    generator = DockerComposeGenerator(model, output_dir=str(tmp_path))
    generator.generate()
    assert set(generator.traced_files) == {"docker-compose.yml"}


def test_files_an_llm_personalization_wrote_are_listed_and_left_at_file_level(tmp_path):
    from unittest.mock import patch

    from besser.generators.agents import agent_personalization

    model, agents = _model(True)
    generator = DockerComposeGenerator(
        model, output_dir=str(tmp_path), agent_models_by_id=agents,
        agent_configs_by_id={"sup": {"agentStyle": "formal", "openaiApiKey": "k"}})
    with patch.object(agent_personalization, "call_openai_chat",
                      side_effect=lambda *_a, **_k: "\n".join(f"{n}. rewritten" for n in range(1, 30))):
        generator.generate()
    assert generator.ai_files == {"agent_supervisor/AgentSupervisor.py", "agent_supervisor/personalized_agent_model.py",
                                  "agent_supervisor/personalized_agent_model.json"}
    assert "agent_supervisor/AgentSupervisor.py" not in generator.traced_files
    assert "agent_coder/AgentCoder.py" in generator.traced_files
    assert (tmp_path / "agent_supervisor" / "AgentSupervisor.py").is_file()
