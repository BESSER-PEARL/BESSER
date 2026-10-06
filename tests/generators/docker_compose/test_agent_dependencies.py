"""Generated Docker dependencies follow the actual BAF configuration."""

import ast

import pytest

from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, RAGTextSplitter, RAGVectorStore, SimpleIntentClassifierConfiguration,
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.BUML.metamodel.uml_deployment import (
    Artifact, DeploymentModel, DeploymentRelation, Locality, Node, NodeKind,
)
from besser.generators.docker_compose import DockerComposeGenerator
from besser.generators.docker_compose.agent_dependencies import baf_dependency_extras


BASE_EXTRAS = ('extras', 'llms')


def _agent():
    agent = Agent('Worker')
    initial = agent.new_state('initial', initial=True)
    initial.set_body(Body('welcome', actions=[AgentReply('Authored welcome')]))
    agent.new_state('waiting')
    initial.go_to(agent.states[1])
    return agent


def _bake(tmp_path, agent, config=None, a2a=False):
    node = Node('Cluster', kind=NodeKind.EXECUTION_ENVIRONMENT)
    worker = Artifact('Worker', locality=Locality.LOCAL)
    worker.agent_model_ref = 'worker-id'
    artifacts = {worker}
    agents = {'worker-id': agent}
    if a2a:
        peer = Agent('Peer')
        peer.new_state('initial', initial=True)
        agent._a2a = {'inbound': [{'peer': 'Peer', 'target_state': 'initial'}], 'outbound': []}
        peer._a2a = {'inbound': [{'peer': 'Worker', 'target_state': 'initial'}], 'outbound': []}
        peer_artifact = Artifact('Peer', locality=Locality.LOCAL)
        peer_artifact.agent_model_ref = 'peer-id'
        artifacts.add(peer_artifact)
        agents['peer-id'] = peer
    deployment = DeploymentModel('Swarm', nodes={node}, artifacts=artifacts,
                                 relationships={DeploymentRelation(a, node) for a in artifacts})
    DockerComposeGenerator(deployment, str(tmp_path), agents,
                           agent_configs_by_id={'worker-id': config} if config else {}).generate()
    code = (tmp_path / 'worker' / 'Worker.py').read_text(encoding='utf-8')
    ast.parse(code)
    dockerfile = (tmp_path / 'worker' / 'Dockerfile').read_text(encoding='utf-8')
    assert 'initial.go_to(waiting)' in code
    assert code.count('agent = Agent(') == code.count('agent.load_properties(') == 1
    assert code.count('initial=True') == code.count('if __name__') == 1
    assert ('_A2A_INBOUND' in code) == a2a
    return code, dockerfile


def _assert_install(dockerfile, extras):
    assert f'RUN pip install --no-cache-dir "besser-agentic-framework[{",".join(extras)}]"' in dockerfile
    assert '[all]' not in dockerfile
    assert 'CMD ["python", "Worker.py"]' in dockerfile


@pytest.mark.parametrize('provider', sorted(Agent._LLM_PROVIDERS))
@pytest.mark.parametrize('a2a', [False, True])
def test_every_provider_gets_only_its_required_backend(tmp_path, provider, a2a):
    agent = _agent()
    llm = agent.new_llm('model', provider=provider, parameters={})
    _, dockerfile = _bake(tmp_path, agent, a2a=a2a)
    local = type(llm).__name__ == 'LLMHuggingFace'
    _assert_install(dockerfile, BASE_EXTRAS + (('torch',) if local else ()))
    if a2a:
        # A local worker does not force a backend onto the API/fallback peer.
        peer_dockerfile = (tmp_path / 'peer' / 'Dockerfile').read_text(encoding='utf-8')
        assert 'besser-agentic-framework[extras,llms]' in peer_dockerfile
        assert '[all]' not in peer_dockerfile


@pytest.mark.parametrize('config,expected', [
    ({'intentRecognitionTechnology': 'classical'}, BASE_EXTRAS + ('torch',)),
    ({'system': {'intentRecognitionTechnology': 'classical'}}, BASE_EXTRAS + ('torch',)),
    ({'intentRecognitionTechnology': 'llm-based'}, BASE_EXTRAS),
    ({'system': {'intentRecognitionTechnology': 'llm-based'}}, BASE_EXTRAS),
])
def test_saved_classical_and_llm_settings_select_dependencies(tmp_path, config, expected):
    agent = _agent()
    agent.new_llm('model', provider='openai')
    code, dockerfile = _bake(tmp_path, agent, config=config)
    _assert_install(dockerfile, expected)
    assert ("framework='pytorch'" in code) == ('torch' in expected)


@pytest.mark.parametrize('scope', ['default', 'state', 'personalized_state'])
def test_classifiers_in_authored_and_personalized_graphs_keep_torch(tmp_path, scope):
    agent = _agent()
    agent.new_llm('model', provider='openai')
    classifier = SimpleIntentClassifierConfiguration()
    config = None
    if scope == 'default':
        agent.default_ic_config = classifier
    elif scope == 'state':
        agent.states[1].ic_config = classifier
    else:
        profile = _agent()
        profile.states[1].ic_config = classifier
        config = {'personalizationMapping': [{'name': 'Reader', 'configuration': {},
                                              'user_profile': {}, 'agent_model': profile}]}
    code, dockerfile = _bake(tmp_path, agent, config=config, a2a=True)
    _assert_install(dockerfile, BASE_EXTRAS + ('torch',))
    assert 'SimpleIntentClassifierConfiguration(' in code
    if scope == 'personalized_state':
        assert 'waiting_Reader' in code


def test_api_agent_keeps_rag_tools_and_skills_without_local_backends(tmp_path):
    agent = _agent()
    agent.new_llm('model', provider='openai')
    agent.new_rag('knowledge', RAGVectorStore(embedding_provider='openai'),
                  RAGTextSplitter('recursive_character', 1000, 100), 'model')
    agent.new_tool('declared_tool', description='Authored tool', code='def useful_tool():\n    return "OK"\n')
    agent.new_skill('declared_skill', content='Use the authored tool.', description='Authored skill')
    code, dockerfile = _bake(tmp_path, agent, a2a=True)
    _assert_install(dockerfile, BASE_EXTRAS)
    assert 'OpenAIEmbeddings' in code and 'Chroma(' in code and 'RAG(' in code
    assert "'name': 'declared_tool'" in code and "'name': 'declared_skill'" in code
    assert (tmp_path / 'worker' / 'declared_tools' / 'tool_000.py').is_file()
    assert (tmp_path / 'worker' / 'skills' / 'skill_000.md').is_file()


@pytest.mark.parametrize('code,backends', [
    ('from baf.nlp.llm.llm_huggingface import LLMHuggingFace\n'
     '# LLMHuggingFace()\ntext = "SimpleIntentClassifierConfiguration()"', ()),
    ('LLMHuggingFaceAPI(agent=agent, name="model", parameters={})\nLLMOllama(agent=agent, name="model2")', ()),
    ('SimpleIntentClassifierConfiguration()', ('torch',)),
    ('SimpleIntentClassifierConfiguration(framework="pytorch")', ('torch',)),
    ('SimpleIntentClassifierConfiguration(framework="tensorflow")', ('tensorflow',)),
    ('SimpleIntentClassifierConfiguration("tensorflow")', ('tensorflow',)),
    ('LLMHuggingFace(agent=agent)\nSimpleIntentClassifierConfiguration(framework="tensorflow")', ('torch', 'tensorflow')),
    ('baf.LLMHuggingFace(agent=agent)', ('torch',)),
    ('SimpleIntentClassifierConfiguration(framework=selected_framework)', ('torch', 'tensorflow')),
    ('SimpleIntentClassifierConfiguration(**settings)', ('torch', 'tensorflow')),
    ('SimpleIntentClassifierConfiguration(framework="pytorch", **settings)', ('torch', 'tensorflow')),
])
def test_backend_union_and_dynamic_configuration_preserve_required_groups(code, backends):
    assert baf_dependency_extras(code) == BASE_EXTRAS + backends
