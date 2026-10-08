"""Saved BAF settings must stay isolated and agree with Docker listeners."""

import ast
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from besser.BUML.metamodel.state_machine.agent import Agent
from besser.BUML.metamodel.uml_deployment import Artifact, DeploymentModel, DeploymentRelation, Locality, Node
from besser.generators.docker_compose.docker_compose_generator import DockerComposeGenerator
from besser.generators.docker_compose.runtime_config import resolve_runtime_yaml


def _yaml(a2a=8000, websocket=8765, streamlit=5000):
    return yaml.safe_dump({
        'nlp': {'language': 'ca', 'openai': {'api_key': 'test-only-placeholder'}},
        'platforms': {'a2a': {'port': a2a}, 'websocket': {
            'host': 'localhost', 'port': websocket,
            'streamlit': {'host': 'localhost', 'port': streamlit, 'chat': {'font': 'serif'}},
        }},
        'db': {'sql': [{'fff': {'dialect': 'sqlite', 'database': 'saved.db'}}]},
    }, sort_keys=False)


def _deployment(agents):
    node = Node('Cluster')
    artifacts = []
    for service, ref in agents:
        artifact = Artifact(service, locality=Locality.LOCAL)
        artifact.agent_model_ref = ref
        artifacts.append(artifact)
    return DeploymentModel('swarm', nodes={node}, artifacts=set(artifacts),
                           relationships={DeploymentRelation(a, node) for a in artifacts})


@pytest.mark.parametrize('source', [None, '', '  \n'])
def test_absent_yaml_uses_existing_defaults(source):
    assert resolve_runtime_yaml(source) == (None, {'a2a': 8000, 'websocket': 8765, 'streamlit': 5000})


def test_saved_yaml_preserves_non_listener_data_and_normalizes_hosts():
    original = _yaml(9123, '9876', 5100)
    normalized, ports = resolve_runtime_yaml(original)
    before, after = yaml.safe_load(original), yaml.safe_load(normalized)
    assert after['db'] == before['db']
    assert after['nlp'] == before['nlp']
    assert after['platforms']['websocket']['streamlit']['chat'] == {'font': 'serif'}
    assert after['platforms']['websocket']['host'] == '0.0.0.0'
    assert after['platforms']['websocket']['streamlit']['host'] == '0.0.0.0'
    assert ports == {'a2a': 9123, 'websocket': 9876, 'streamlit': 5100}
    assert yaml.safe_load(original) == before


@pytest.mark.parametrize('value', [0, 65536, True, 8000.5, 'bad'])
def test_invalid_a2a_ports_fail_generation(value):
    with pytest.raises(ValueError, match='A2A port'):
        resolve_runtime_yaml(_yaml(a2a=value))


@pytest.mark.parametrize('source', ['[]', 'plain text', 'platforms: [bad]', 'platforms: ['])
def test_invalid_saved_yaml_is_not_silently_discarded(source):
    with pytest.raises(ValueError):
        resolve_runtime_yaml(source)


def test_duplicate_agent_names_keep_uuid_configuration_and_distinct_ports(tmp_path):
    alpha = Agent('Duplicate')
    beta = Agent('Duplicate')
    alpha.new_state('initial', initial=True)
    beta.new_state('initial', initial=True)
    configs = {'uuid-alpha': {'system': {'agentPlatform': 'streamlit'}},
               'uuid-beta': {'system': {'agentPlatform': 'websocket'}}}
    saved = deepcopy(configs)
    model = _deployment([('Alpha', 'uuid-alpha'), ('Beta', 'uuid-beta')])
    DockerComposeGenerator(
        model, str(tmp_path), {'uuid-alpha': alpha, 'uuid-beta': beta},
        agent_configs_by_id=configs,
        agent_config_yamls_by_id={'uuid-alpha': _yaml(9001, 8700, 5100),
                                 'uuid-beta': _yaml(9002, 8800, 5200)},
    ).generate()
    compose = yaml.safe_load((tmp_path / 'docker-compose.yml').read_text(encoding='utf-8'))
    alpha_ports = compose['services']['alpha']['ports']
    beta_ports = compose['services']['beta']['ports']
    assert {p.split(':')[1] for p in alpha_ports} == {'5100', '8700'}
    assert {p.split(':')[1] for p in beta_ports} == {'8800'}
    assert len({p.split(':')[0] for p in alpha_ports + beta_ports}) == 3
    for service, port in [('alpha', 9001), ('beta', 9002)]:
        config = yaml.safe_load((tmp_path / service / 'config.yaml').read_text(encoding='utf-8'))
        assert config['platforms']['a2a']['port'] == port
        assert config['db']['sql'][0]['fff']['file'] == 'saved.db'
        code = (tmp_path / service / 'Duplicate.py').read_text(encoding='utf-8')
        ast.parse(code)
        assert code.count('agent = Agent(') == 1
        assert code.count('agent.load_properties(') == 1
        assert code.count('initial=True') == 1
        assert 'use_a2a_platform' not in code
    assert 'use_websocket_platform(use_ui=True)' in (tmp_path / 'alpha' / 'Duplicate.py').read_text()
    assert 'use_websocket_platform(use_ui=False)' in (tmp_path / 'beta' / 'Duplicate.py').read_text()
    assert configs == saved


def test_tagged_peer_ports_match_the_worker_config(tmp_path):
    sender, worker = Agent('Sender'), Agent('Worker')
    sender.new_state('initial', initial=True)
    worker.new_state('initial', initial=True)
    sender._a2a = {'outbound': [{'peer': 'Worker', 'state': 'initial', 'kind': 'delegates'}], 'inbound': []}
    worker._a2a = {'outbound': [], 'inbound': [{'peer': 'Sender', 'target_state': 'initial'}]}
    model = _deployment([('Sender', 'sender-id'), ('Worker', 'worker-id')])
    DockerComposeGenerator(
        model, str(tmp_path), {'sender-id': sender, 'worker-id': worker},
        agent_config_yamls_by_id={'worker-id': _yaml(a2a=9123)},
    ).generate()
    code = (tmp_path / 'sender' / 'Sender.py').read_text(encoding='utf-8')
    tree = ast.parse(code)
    peer_ports = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == '_A2A_PEER_PORTS' for t in n.targets))
    assert peer_ports['worker'] == 9123
    config = yaml.safe_load((tmp_path / 'worker' / 'config.yaml').read_text(encoding='utf-8'))
    assert config['platforms']['a2a']['port'] == 9123
    compose = yaml.safe_load((tmp_path / 'docker-compose.yml').read_text(encoding='utf-8'))
    assert not compose['services']['worker'].get('ports')


def test_deployment_route_forwards_uuid_settings_and_normalizes_profiles(monkeypatch):
    from besser.utilities.web_modeling_editor.backend.models import DiagramInput
    from besser.utilities.web_modeling_editor.backend.routers import generation_router as route

    captured = {}
    normalized = []
    entries = [DiagramInput(id='first', title='First', model={'elements': {'dummy': {}}},
                            config={'agentPlatform': 'websocket'}, configYaml=_yaml(a2a=9001)),
               DiagramInput(id='second', title='Second', model={'elements': {'dummy': {}}},
                            config={'personalizationMapping': [{'user_profile': {'raw': True}}]},
                            configYaml=_yaml(a2a=9002))]
    saved = deepcopy(entries)
    diagram = SimpleNamespace(model_dump=lambda: {})
    project = SimpleNamespace(name='Example', diagrams={'AgentDiagram': entries},
                              get_active_diagram=lambda kind: diagram)
    monkeypatch.setattr(route, 'process_deployment_diagram', lambda payload: 'deployment')
    monkeypatch.setattr(route, 'process_agent_diagram', lambda payload: Agent('SameName'))
    monkeypatch.setattr(route, 'annotate_agent_with_a2a', lambda *args: None)

    def normalize(config, payload, callback):
        normalized.append(payload['id'])
        config['personalizationMapping'][0]['user_profile'] = {'normalized': True}

    monkeypatch.setattr(route, 'normalize_personalization_mapping', normalize)

    class CaptureGenerator:
        def __init__(self, model, output_dir, **kwargs):
            captured.update(kwargs)
            self.output_dir = output_dir

        def generate(self):
            Path(self.output_dir, 'docker-compose.yml').write_text('services: {}\n', encoding='utf-8')

    response = asyncio.run(route._handle_deployment_project_generation(
        project, SimpleNamespace(generator_class=CaptureGenerator), 'docker_compose'))
    assert response.media_type == 'application/zip'
    assert set(captured['agent_models_by_id']) == {'first', 'second'}
    assert captured['agent_configs_by_id']['first'] == entries[0].config
    assert captured['agent_configs_by_id']['second']['personalizationMapping'][0]['user_profile'] == {
        'normalized': True}
    assert captured['agent_config_yamls_by_id'] == {e.id: e.configYaml for e in entries}
    assert normalized == ['second']
    assert entries == saved


def test_default_config_binds_all_interfaces_only_inside_containers(tmp_path):
    """Without saved YAML the Compose build context binds 0.0.0.0; a standalone agent keeps localhost."""
    from besser.generators.agents.baf_generator import BAFGenerator

    agent = Agent('Solo')
    agent.new_state('initial', initial=True)
    DockerComposeGenerator(_deployment([('Solo', 'uuid-solo')]), str(tmp_path / 'compose'),
                           {'uuid-solo': agent}).generate()
    container = yaml.safe_load((tmp_path / 'compose' / 'solo' / 'config.yaml').read_text(encoding='utf-8'))
    assert container['platforms']['websocket']['host'] == '0.0.0.0'
    assert container['platforms']['websocket']['streamlit']['host'] == '0.0.0.0'

    BAFGenerator(agent, output_dir=str(tmp_path / 'standalone')).generate()
    standalone = yaml.safe_load((tmp_path / 'standalone' / 'config.yaml').read_text(encoding='utf-8'))
    assert standalone['platforms']['websocket']['host'] == 'localhost'
    assert standalone['platforms']['websocket']['streamlit']['host'] == 'localhost'
