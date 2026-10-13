"""Combined generation and real BAF graph execution with offline LLM/transport doubles."""

import ast
import asyncio
import json
import socket
from datetime import datetime
from types import SimpleNamespace

import pytest

from besser.BUML.metamodel.gui import GUIModel, Module, Screen, Text
from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, GUIEvent, GUIReplyAction, Intent, LLMReply, ReceiveTextEvent, WebSocketPlatform,
)
from besser.BUML.metamodel.state_machine.state_machine import Body, Condition
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode
from besser.generators.docker_compose.docker_compose_generator import (
    DockerComposeGenerator, _a2a_descriptor, _a2a_descriptor_from_tags,
)


def _model(name='Authored'):
    agent = Agent(name)
    agent.platforms.append(WebSocketPlatform())
    agent.new_llm(name='authored_model', provider='openai', parameters={})
    ask = agent.new_state('ask', initial=True)
    receive = agent.new_state('receive')
    compute = agent.new_state('compute')
    wait = agent.new_state('wait')
    done = agent.new_state('done')
    ask.set_body(Body('ask_body', actions=[AgentReply('Welcome'), GUIReplyAction('order-form')]))
    ask.when_event(GUIEvent(message_id='order-form')).go_to(receive)
    receive.set_body(Body('receive_body', actions=[AgentReply('Received {user_message}', use_session_vars=True)]))
    receive.go_to(compute)
    compute.set_body(Body('compute_body', actions=[
        LLMReply(prompt='Authored instruction', store_in_session='answer'), GUIReplyAction('order-form'),
    ]))
    compute.go_to(wait)
    wait.when_event(GUIEvent(message_id='order-form')).go_to(done)
    wait.set_fallback_body(Body('wait_fallback', actions=[AgentReply('Try the form')]))
    done.set_body(Body('done_body', actions=[AgentReply('Finished')]))
    screen = Screen('order', '', view_elements={Text('title', 'Order')}, is_main_page=True)
    gui = GUIModel(name='order_gui', package='', versionCode='1', versionName='1.0', description='',
                   modules={Module('module', screens={screen})})
    agent.add_gui_model('order-form', gui)
    return agent


def _generate(agent, directory, descriptor=None, config=None):
    BAFGenerator(agent, str(directory), generation_mode=GenerationMode.CODE_ONLY,
                 a2a_descriptor=descriptor, config=config, test_mode=True).generate()
    for path in directory.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    return (directory / f'{agent.name}.py').read_text(encoding='utf-8')


def _singleton_checks(code):
    tree = ast.parse(code)
    assert code.count('agent = Agent(') == 1
    assert code.count('agent.load_properties(') == 1
    assert code.count('initial=True') == 1
    assert len([n for n in tree.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare) and '__name__' in ast.unparse(n.test)]) == 1


@pytest.fixture
def load_runtime(monkeypatch):
    """Use real BAF Agent/State/Session/Condition; replace external providers/UI only.

    The installed BAF may lack newer optional provider imports or AgentGUI.
    Remove those imports from the parsed module, while testing their generation
    separately. No framework graph code is replaced.
    """
    core = pytest.importorskip('baf.core.agent')
    from baf.core.session import Session
    from baf.core.transition.event import Event
    from baf.nlp.intent_classifier.intent_classifier_configuration import (
        LLMIntentClassifierConfiguration, SimpleIntentClassifierConfiguration,
    )

    class RuntimeGUIEvent(Event):
        def __init__(self, message_id=None):
            super().__init__('gui', timestamp=datetime.now())
            self.message_id = message_id

        def is_matching(self, event):
            return isinstance(event, RuntimeGUIEvent) and event.message_id == self.message_id

    class OfflinePlatform:
        def __init__(self):
            self.sent = []
            self.router = SimpleNamespace(methods={})
            self.router.register = self.router.methods.__setitem__

        def reply(self, session, message):
            self.sent.append(('reply', message))

        def reply_gui(self, session, gui):
            self.sent.append(('gui', gui))

    class OfflineLLM:
        def __init__(self, agent, name, **kwargs):
            self.name = name
            self.calls = []
            self.contexts = []

        def predict(self, message, session=None, **kwargs):
            self.calls.append((message, session, kwargs))
            return 'authored answer: ' + message

        def add_user_context(self, **kwargs):
            self.contexts.append(kwargs)

    class RuntimeAgent(core.Agent):
        def load_properties(self, path):
            self.loads = [path]

        def use_websocket_platform(self, **kwargs):
            self.ui = OfflinePlatform()
            self.ui_options = kwargs
            return self.ui

        def use_a2a_platform(self):
            self.rpc = OfflinePlatform()
            return self.rpc

    # Session storage still uses the real API, with database writes disabled.
    monkeypatch.setattr(RuntimeAgent, '_monitoring_db_store_session_variables', lambda *args: None)

    def load(code, directory):
        tree = ast.parse(code)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and (
                node.module.startswith('baf.nlp') or node.module.startswith('guis.')
                or node.module == 'baf.core.agent'))]
        ns = {'__name__': 'generated_test', '__file__': str(directory / 'agent.py'),
              'Agent': RuntimeAgent, 'LLMOpenAI': OfflineLLM, 'GUIEvent': RuntimeGUIEvent,
              'LLMIntentClassifierConfiguration': LLMIntentClassifierConfiguration,
              'SimpleIntentClassifierConfiguration': SimpleIntentClassifierConfiguration,
              'order_form': 'order_gui', 'Session': Session}
        exec(compile(tree, str(directory / 'agent.py'), 'exec'), ns)
        return ns

    return load


def test_plain_output_keeps_gui_graph_and_ignores_empty_descriptor(tmp_path):
    agent = _model()
    plain = _generate(agent, tmp_path / 'plain')
    empty = _generate(agent, tmp_path / 'empty', {'to_peers': [], 'a2a_server': False})
    assert plain == empty
    _singleton_checks(plain)
    assert 'ask.when_event(GUIEvent(message_id="order-form")).go_to(receive)' in plain
    assert 'compute.go_to(wait)' in plain
    assert 'wait.set_fallback_body(wait_fallback)' in plain
    assert 'platform.reply_gui(session, order_form)' in plain
    assert 'use_a2a_platform' not in plain


def test_plain_agent_executes_authored_gui_transition(tmp_path, load_runtime):
    ns = load_runtime(_generate(_model(), tmp_path), tmp_path)
    session = ns['Session']('human', ns['agent'], ns['agent'].ui)
    ns['ask']._body(session)
    assert ns['agent'].ui.sent == [('reply', 'Welcome'), ('gui', 'order_gui')]
    session.event = ns['GUIEvent']('order-form')
    transition = ns['ask'].transitions[0]
    assert transition.evaluate(session, session.event)
    # GUI transition and automatic continuation use the actual BAF graph.
    session._current_state = transition.dest
    session.event = ns['ReceiveTextEvent']('order')
    ns['receive']._body(session)
    assert ns['receive'].transitions[0].is_auto()
    ns['compute']._body(session)
    assert session.get('answer') == 'authored answer: order'


@pytest.mark.parametrize('human_facing', [False, True])
def test_tagged_rpc_runs_authored_states_gui_and_isolated_sessions(tmp_path, load_runtime, human_facing):
    agent = _model()
    agent._human_facing = human_facing
    agent._a2a = {'outbound': [], 'inbound': [
        {'peer': 'Peer', 'flow': 'order', 'target_state': 'receive', 'source_state': 'ask', 'intent': 'receive_order'},
    ]}
    descriptor = _a2a_descriptor_from_tags(agent, {'authored', 'peer'})
    code = _generate(agent, tmp_path, descriptor)
    _singleton_checks(code)
    assert ('use_websocket_platform' in code) is human_facing
    ns = load_runtime(code, tmp_path)
    first = asyncio.run(ns['handle'](message='first', flow='order', **{'from': 'peer'}))
    second = asyncio.run(ns['handle'](message='second', flow='order'))
    assert first == {'reply': 'Received first\n\nauthored answer: first',
                     'actions': [{'action': 'reply_gui', 'value': 'order_gui'}]}
    assert second['reply'] == 'Received second\n\nauthored answer: second'
    sessions = [call[1] for call in ns['default_llm'].calls]
    assert sessions[0].id != sessions[1].id
    assert all(s.current_state.name == 'wait' for s in sessions)
    assert sessions[0].get('answer') == 'authored answer: first'
    assert ns['wait'].transitions[0].event.message_id == 'order-form'
    assert ns['wait']._fallback_body.__name__ == 'wait_fallback'
    if human_facing:
        assert ns['agent'].ui.sent == []  # RPC actions stay in their request.


def test_legacy_inbound_enters_authored_boundary(tmp_path, load_runtime):
    agent = _model()
    boundary = agent.new_state('from_peer')
    boundary.set_body(Body('from_peer_body', actions=[AgentReply('Legacy receipt')]))
    boundary.go_to(agent.states[2])
    code = _generate(agent, tmp_path, _a2a_descriptor(agent, {'authored', 'peer'}))
    ns = load_runtime(code, tmp_path)
    result = asyncio.run(ns['handle'](message='legacy'))
    assert result['reply'] == 'Legacy receipt\n\nauthored answer: legacy'


def test_outbound_calls_happen_at_each_authored_state_with_its_kind(tmp_path, load_runtime):
    agent = _model()
    agent._human_facing = True
    agent._a2a = {'inbound': [{'peer': 'Peer', 'target_state': 'receive'}], 'outbound': [
        {'peer': 'Peer', 'state': 'receive', 'order': 1, 'kind': 'delegates'},
        {'peer': 'Peer', 'state': 'compute', 'order': 2, 'kind': 'supervises'},
    ]}
    code = _generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'}))
    ns = load_runtime(code, tmp_path)
    calls = []
    ns['_peer_replica_urls'] = lambda service: [f'http://{service}:8000']
    ns['_a2a_call'] = lambda url, peer, message, **kw: calls.append((peer, message, kw)) or 'peer output'
    result = asyncio.run(ns['handle'](message='task'))
    assert len(calls) == 2
    assert calls[0][1].startswith('Delegated subtask')
    assert calls[1][1].startswith('Review the following work')
    assert 'Received task' in result['reply'] and 'authored answer: task' in result['reply']
    calls.clear()
    leaf = asyncio.run(ns['handle'](message='ballot', leaf=True))
    assert calls == []
    assert leaf['reply'] == 'Received ballot'


def test_personalization_render_keeps_a2a_and_profile_graph(tmp_path, load_runtime):
    agent = _model()
    profile = _model('Profile')
    profile.states[1].set_body(Body('receive_body', actions=[AgentReply('Personalized receipt')]))
    agent._a2a = {'outbound': [], 'inbound': [{'peer': 'Peer', 'target_state': 'receive'}]}
    config = {'personalizationMapping': [{'name': 'Reader', 'configuration': {},
                                          'user_profile': {}, 'agent_model': profile}]}
    code = _generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'}), config)
    _singleton_checks(code)
    assert "router_initial_state = agent.new_state('router_initial_state', initial=True)" in code
    assert 'receive_Reader' in code and 'platform.reply_gui(session, order_form)' in code
    ns = load_runtime(code, tmp_path)
    result = asyncio.run(ns['handle'](message='profile task', user_profile='Reader'))
    assert result['reply'] == 'Personalized receipt\n\nauthored answer: profile task'
    assert ns['default_llm'].calls[0][1].current_state.name == 'wait_Reader'


def test_compose_bakes_authored_gui_graph_once(tmp_path):
    from besser.BUML.metamodel.uml_deployment import (
        Artifact, DeploymentModel, DeploymentRelation, Locality, Node, NodeKind,
    )
    agent = _model()
    peer = Agent('Peer')
    peer.new_state('initial', initial=True)
    agent._a2a = {'outbound': [{'peer': 'Peer', 'state': 'compute', 'kind': 'delegates'}], 'inbound': []}
    peer._a2a = {'outbound': [], 'inbound': [{'peer': 'Authored'}]}
    node = Node('Cluster', kind=NodeKind.EXECUTION_ENVIRONMENT)
    artifacts = [Artifact(name, locality=Locality.LOCAL) for name in ('Authored', 'Peer')]
    for artifact in artifacts:
        artifact.agent_model_ref = artifact.name
    model = DeploymentModel('swarm', nodes={node}, artifacts=set(artifacts),
                            relationships={DeploymentRelation(a, node) for a in artifacts})
    DockerComposeGenerator(model, str(tmp_path), {'Authored': agent, 'Peer': peer}).generate()
    code = (tmp_path / 'authored' / 'Authored.py').read_text(encoding='utf-8')
    _singleton_checks(code)
    assert 'compute.go_to(wait)' in code
    assert '_A2A_OUTBOUND' in code
    assert (tmp_path / 'authored' / 'guis' / 'order_form.py').is_file()
    assert (tmp_path / 'authored' / 'Dockerfile').is_file()


def _governed_agent(human_facing=False, requires_human=False):
    agent = _model('Owner')
    agent._human_facing = human_facing
    inbound, policies = [], []
    for name, flow, policy in [('merge_a', 'gw1', 'MajorityPolicy'), ('merge_b', 'gw2', 'LeaderDrivenPolicy')]:
        state = agent.new_state(name)
        state.set_body(Body(name + '_body', actions=[AgentReply('Authored ' + name)]))
        state.go_to(agent.states[4])
        inbound.append({'peer': 'Peer', 'target_state': name, 'flow': flow})
        policies.append({'gateway_id': flow, 'merge_state': name, 'policy_type': policy, 'ratio': 0.5,
                         'requires_human': requires_human, 'participants': [
                             {'name': 'Peer', 'kind': 'agent', 'confidence': 1.0}],
                         'producers': ['Peer'], 'instruction': 'Govern the merge', 'summary': 'Policy facts'})
    agent._a2a = {'outbound': [], 'inbound': inbound}
    agent._governance = policies
    agent._governance_by_state = dict(zip(['merge_a', 'merge_b'], policies))
    return agent


def test_governed_rpc_dispatches_each_merge_and_continues_authored_graph(tmp_path, load_runtime):
    agent = _governed_agent(human_facing=True)
    code = _generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'owner', 'peer'}))
    _singleton_checks(code)
    assert code.count('def tally(') == 1
    ns = load_runtime(code, tmp_path)
    calls = []

    def fanout(task, only=None, leaf=False):
        calls.append((only, leaf))
        return [('peer', 'vote', 'BALLOT: C1')]

    ns['_run_fanout'] = fanout
    first = asyncio.run(ns['handle'](message='candidate A', flow='gw1'))
    second = asyncio.run(ns['handle'](message='candidate B', flow='gw2'))
    assert first['reply'].startswith('Authored merge_a\n\ncandidate A')
    assert 'winner C1' in first['reply'] and first['reply'].endswith('Finished')
    assert calls == [({'peer'}, True)]
    assert second['reply'].startswith('Authored merge_b\n\nauthored answer: Candidate to finalize:')
    assert second['reply'].endswith('Finished')
    assert ns['default_llm'].calls[-1][2]['parameters'] is None
    with pytest.raises(ValueError, match='Ambiguous'):
        asyncio.run(ns['handle'](message='unrouted candidate'))
    leaf = asyncio.run(ns['handle'](message='vote locally', leaf=True))
    assert leaf['reply'] == 'authored answer: vote locally'
    assert calls == [({'peer'}, True)]


@pytest.mark.parametrize('flow', ['gw1', 'gw2'])
def test_governed_rpc_pauses_before_authored_continuation(tmp_path, load_runtime, flow):
    agent = _governed_agent(requires_human=True)
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'owner', 'peer'})), tmp_path)
    ns['_run_fanout'] = lambda *args, **kwargs: [('peer', 'vote', 'BALLOT: C1')]
    result = asyncio.run(ns['handle'](message='candidate', flow=flow))
    assert 'human approval' in result['reply']
    assert 'Finished' not in result['reply']
    assert result['gov_pending']['gateway'] == flow
    assert result['gov_pending'].get('is_voting', True) is (flow == 'gw1')


@pytest.mark.parametrize('policy_type', ['MajorityPolicy', 'LeaderDrivenPolicy'])
def test_hybrid_single_authored_merge_holds_and_resumes_human_decision(
        tmp_path, load_runtime, policy_type):
    agent = _governed_agent(human_facing=True, requires_human=True)
    agent._a2a['inbound'] = agent._a2a['inbound'][:1]
    gov = dict(agent._governance[0], policy_type=policy_type)
    agent._governance = [gov]
    agent._governance_by_state = {'merge_a': gov}
    descriptor = _a2a_descriptor_from_tags(agent, {'owner', 'peer'})
    assert descriptor['states'][0]['name'] == 'merge_a'
    ns = load_runtime(_generate(agent, tmp_path, descriptor), tmp_path)
    votes = []
    ns['_run_fanout'] = lambda *args, **kwargs: votes.append('vote') or [('peer', 'vote', 'BALLOT: C1')]
    session = ns['Session']('human', ns['agent'], ns['agent'].ui)
    session.call_manage_transition = lambda: None
    session._current_state = ns['merge_a']
    ns['agent']._sessions[session.id] = session
    session.event = ns['ReceiveTextEvent']('candidate', session.id, human=True)
    ns['merge_a']._body(session)
    transition = ns['merge_a'].transitions[0]
    assert not transition.is_condition_true(session)
    assert session.get('_a2a_pending_state') == 'merge_a'
    ns['agent'].receive_event(ns['ReceiveTextEvent']('C1', session.id, human=True))
    assert transition.is_condition_true(session)
    assert session.get('_a2a_pending_state') is None
    session.move(transition)
    replies = [message for action, message in ns['agent'].ui.sent if action == 'reply']
    assert replies.count('Authored merge_a') == 1
    assert replies[-1] == 'Finished'
    if policy_type == 'MajorityPolicy':
        assert votes == ['vote']
        assert any('winner C1' in message for message in replies)
    else:
        assert votes == []
        assert 'Human decision: C1' in ns['default_llm'].calls[-1][0]


def test_bound_producer_sends_each_governed_stage_at_authored_state(tmp_path, load_runtime):
    agent = _model('Producer')
    agent._human_facing = True
    agent._a2a = {'inbound': [{'peer': 'Owner', 'target_state': 'receive'}], 'outbound': [
        {'peer': 'Owner', 'state': 'receive', 'target_gateway': 'gw1', 'order': 1},
        {'peer': 'Owner', 'state': 'compute', 'target_gateway': 'gw2', 'order': 2},
    ]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'producer', 'owner'})), tmp_path)
    stages = []
    ns['_merge_send'] = lambda message, service, flow: stages.append((message, service, flow)) or (flow + ' result', None)
    result = asyncio.run(ns['handle'](message='task'))
    assert [(service, flow) for _, service, flow in stages] == [('owner', 'gw1'), ('owner', 'gw2')]
    assert stages[0][0] == 'Received task'
    assert stages[1][0] == 'authored answer: task'
    assert result['reply'].endswith('gw2 result')


def test_topology_only_voting_fallback_keeps_legacy_two_round_behavior(tmp_path, load_runtime):
    agent = Agent('Owner')
    agent.new_state('initial', initial=True)
    agent._a2a = {'outbound': [{'peer': 'Peer', 'state': 'missing'}], 'inbound': [{'peer': 'Peer'}]}
    agent._governance = [{'policy_type': 'VotingPolicy', 'ratio': 0.5, 'requires_human': False,
                          'participants': [{'name': 'Peer', 'confidence': 1.0}], 'producers': ['Peer'],
                          'instruction': 'Vote', 'summary': 'Policy facts'}]
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'owner', 'peer'})), tmp_path)
    rounds = []

    def fanout(task, only=None, leaf=False):
        rounds.append((only, leaf))
        return [('peer', 'reply', 'original candidate' if len(rounds) == 1 else 'BALLOT: C1')]

    ns['_run_fanout'] = fanout
    result = asyncio.run(ns['handle'](message='task', _dispatch_merge={'is_voting': False},
                                    _result={'gov_pending': {'injected': True}}))
    assert rounds == [({'peer'}, True), ({'peer'}, True)]
    assert result['reply'].startswith('original candidate') and 'winner C1' in result['reply']
    assert 'gov_pending' not in result


def test_a2a_transport_keeps_flow_leaf_and_pending_envelope(tmp_path, load_runtime, monkeypatch):
    import json
    agent = _model('Producer')
    agent._a2a = {'inbound': [{'peer': 'Owner', 'target_state': 'receive'}], 'outbound': [
        {'peer': 'Owner', 'state': 'compute', 'target_gateway': 'gw1', 'order': 1},
    ]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'producer', 'owner'})), tmp_path)
    sent = []
    envelope = {'reply': 'approval notice', 'gov_pending': {'gateway': 'gw1'}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({'result': envelope}).encode()

    def open_request(request, timeout):
        sent.append((request.full_url, json.loads(request.data), timeout))
        return Response()

    monkeypatch.setattr(ns['urllib'].request, 'urlopen', open_request)
    assert ns['_a2a_call']('http://owner:8000', 'owner', 'task', flow='gw1', leaf=True) == 'approval notice'
    assert sent[-1] == ('http://owner:8000/a2a', {'jsonrpc': '2.0', 'agent_id': 'owner', 'method': 'handle',
                                               'params': {'message': 'task', 'from': 'producer',
                                                          'flow': 'gw1', 'leaf': True},
                                               'id': 1}, 120)
    assert ns['_a2a_call_full']('http://owner:8000', 'owner', 'task', 'gw1') == envelope


def test_inbound_dispatch_honors_authored_transition_guard(tmp_path, load_runtime):
    agent = _model()
    guard = Condition('allowed_request', source="""
        def allowed_request(session):
            return session.params.get('allow', False)
    """)
    agent.states[0].when_event(ReceiveTextEvent()).with_condition(guard).go_to(agent.states[1])
    agent._a2a = {'outbound': [], 'inbound': [
        {'peer': 'Peer', 'source_state': 'ask', 'target_state': 'receive', 'flow': 'order'},
    ]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'})), tmp_path)
    with pytest.raises(ValueError, match='conditions were not satisfied'):
        asyncio.run(ns['handle'](message='blocked', flow='order'))
    assert ns['default_llm'].calls == []
    result = asyncio.run(ns['handle'](message='allowed', flow='order', allow=True))
    assert result['reply'] == 'Received allowed\n\nauthored answer: allowed'


@pytest.mark.parametrize('mode', ['rpc', 'human', 'legacy'])
def test_peer_response_triggers_authored_receive_transition(tmp_path, load_runtime, mode):
    agent = _model()
    agent._human_facing = True
    returned = agent.new_state('peer_returned')
    returned.set_body(Body('peer_returned_body', actions=[
        AgentReply('Remote received {user_message}', use_session_vars=True), GUIReplyAction('order-form'),
    ]))
    returned.go_to(agent.states[3])
    intent = Intent('recv_peer_result', training_sentences=['peer returned'])
    agent.intents.append(intent)
    receive = agent.states[1]
    receive.transitions.clear()
    receive.when_intent_matched(intent).go_to(returned)
    agent._a2a = {'outbound': [{'peer': 'Peer', 'state': 'receive', 'flow': 'peer_work', 'kind': 'delegates'}],
                  'inbound': [{'peer': 'Peer', 'target_state': 'receive', 'flow': 'kickoff'},
                              {'peer': 'Peer', 'source_state': 'receive', 'target_state': 'peer_returned',
                               'flow': 'peer_reply', 'intent': 'recv_peer_result'}]}
    if mode == 'legacy':
        del agent._a2a
        receive.name = 'to_peer'
        returned.name = 'from_peer'
        descriptor = _a2a_descriptor(agent, {'authored', 'peer'})
    else:
        descriptor = _a2a_descriptor_from_tags(agent, {'authored', 'peer'})
    ns = load_runtime(_generate(agent, tmp_path, descriptor), tmp_path)
    runtime_receive = ns[receive.name]
    calls = []
    ns['_peer_replica_urls'] = lambda service: [f'http://{service}:8000']
    ns['_a2a_call'] = lambda url, service, message, **kw: calls.append(kw) or 'remote result'
    if mode != 'rpc':
        session = ns['Session']('human', ns['agent'], ns['agent'].ui)
        session.call_manage_transition = lambda: None
        session._current_state = runtime_receive
        session.event = ns['ReceiveTextEvent']('task', session.id, human=True)
        runtime_receive._body(session)
        runtime_receive.check_transitions(session)
        assert session.current_state.name == returned.name
        session.move(session.current_state.transitions[0])
        assert session.current_state.name == 'wait'
        assert ('reply', 'Remote received remote result') in ns['agent'].ui.sent
        assert ('gui', 'order_gui') in ns['agent'].ui.sent
    else:
        result = asyncio.run(ns['handle'](message='task', flow='kickoff'))
        assert result['reply'].endswith('Remote received remote result')
        assert result['actions'] == [{'action': 'reply_gui', 'value': 'order_gui'}]
    assert calls == ([{'leaf': False}] if mode == 'legacy' else [{'leaf': False, 'flow': 'peer_work'}])


def test_a2a_requires_an_authored_initial_state(tmp_path):
    agent = Agent('MissingInitial')
    with pytest.raises(ValueError, match='exactly one authored initial state'):
        BAFGenerator(agent, str(tmp_path), a2a_descriptor={'to_peers': ['peer']})

@pytest.mark.parametrize('a2a_enabled', [False, True])
def test_deployment_metadata_registers_distinct_tools_and_skill_once(tmp_path, load_runtime, a2a_enabled):
    from baf.core.agent import Agent as RuntimeAgent
    assert hasattr(RuntimeAgent, 'new_tool') and hasattr(RuntimeAgent, 'new_skill'), 'Use BAF 4.5.2 for this test'
    agent = _model()
    agent.new_tool('test', description='First\nquoted "description"',
                   code="def tool_name(session):\n    return 'first'\n")
    agent.new_tool('name_for_testing', description='Second tool',
                   code="def tool_name(session):\n    return 'second'\n")
    agent.new_skill('product_lkno', description='fefe\nmore detail', content='# Skill\nActual skill body')
    descriptor = None
    if a2a_enabled:
        agent._a2a = {'outbound': [], 'inbound': [{'peer': 'Peer', 'target_state': 'receive'}]}
        descriptor = _a2a_descriptor_from_tags(agent, {'authored', 'peer'})
    BAFGenerator(agent, str(tmp_path), generation_mode=GenerationMode.CODE_ONLY, test_mode=True,
                 a2a_descriptor=descriptor, deployment_component_metadata=True).generate()
    code = (tmp_path / 'Authored.py').read_text(encoding='utf-8')
    _singleton_checks(code)
    assert 'agent.load_tools(' not in code and 'agent.load_skills(' not in code
    assert not (tmp_path / 'tools.py').exists()
    assert len(list((tmp_path / 'declared_tools').glob('*.py'))) == 2
    assert len(list((tmp_path / 'skills').glob('*.md'))) == 1
    ns = load_runtime(code, tmp_path)
    tools = ns['agent']._tools
    skills = ns['agent']._skills
    assert set(tools) == {'test', 'name_for_testing'}
    assert tools['test'].call({'session': 'offline'}) == 'first'
    assert tools['name_for_testing'].call({'session': 'offline'}) == 'second'
    assert tools['test'].description == 'First\nquoted "description"'
    assert set(skills) == {'product_lkno'}
    assert skills['product_lkno'].description == 'fefe\nmore detail'
    assert skills['product_lkno'].content == '# Skill\nActual skill body'
    session = ns['Session']('human', ns['agent'], getattr(ns['agent'], 'ui', None))
    if a2a_enabled:
        reply = asyncio.run(ns['handle'](message='task'))
        assert reply['reply'] == 'Received task\n\nauthored answer: task'
        assert reply['actions'] == [{'action': 'reply_gui', 'value': 'order_gui'}]
    else:
        ns['ask']._body(session)
        assert ns['agent'].ui.sent == [('reply', 'Welcome'), ('gui', 'order_gui')]
        assert ns['ask'].transitions[0].event.message_id == 'order-form'
        session.event = ns['ReceiveTextEvent']('task')
        ns['compute']._body(session)
        assert session.get('answer') == 'authored answer: task'


def test_deployment_metadata_rejects_ambiguous_tool_entrypoint(tmp_path):
    agent = _model()
    agent.new_tool('public_label', code='def first(session):\n    pass\ndef second(session):\n    pass\n')
    with pytest.raises(ValueError, match='one public top-level function'):
        BAFGenerator(agent, str(tmp_path), generation_mode=GenerationMode.CODE_ONLY,
                     deployment_component_metadata=True).generate()


def test_deployment_metadata_accepts_named_entrypoint_and_private_helpers(tmp_path, load_runtime):
    agent = _model()
    agent.new_tool('selected', description='Authored', code=(
        'def _helper():\n    return 7\n'
        'def other(session):\n    return 0\n'
        'def selected(session):\n    return _helper()\n'))
    BAFGenerator(agent, str(tmp_path), generation_mode=GenerationMode.CODE_ONLY,
                 deployment_component_metadata=True).generate()
    ns = load_runtime((tmp_path / 'Authored.py').read_text(encoding='utf-8'), tmp_path)
    assert set(ns['agent']._tools) == {'selected'}
    assert ns['agent']._tools['selected'].fn(None) == 7


def test_ordinary_metadata_opt_in_default_keeps_legacy_output(tmp_path):
    agent = _model()
    agent.new_tool('declared', code='def callable_name(session):\n    return 1\n')
    agent.new_skill('declared_skill', content='# Different title')
    normal = _generate(agent, tmp_path / 'normal')
    BAFGenerator(agent, str(tmp_path / 'explicit_default'), generation_mode=GenerationMode.CODE_ONLY,
                 test_mode=True, deployment_component_metadata=False).generate()
    explicit = (tmp_path / 'explicit_default' / 'Authored.py').read_text(encoding='utf-8')
    assert normal == explicit
    assert 'agent.load_tools(os.path.join(_HERE, "tools.py"))' in normal
    assert 'agent.load_skills(os.path.join(_HERE, "skills"))' in normal
    assert (tmp_path / 'normal' / 'tools.py').is_file()
    assert (tmp_path / 'normal' / 'skills' / 'declared_skill.md').is_file()


def test_a2a_peer_url_uses_map_defaults_and_explicit_override(tmp_path, load_runtime, monkeypatch):
    agent = _model()
    agent._a2a = {'inbound': [{'peer': 'Peer', 'target_state': 'receive'}], 'outbound': [
        {'peer': 'Peer', 'state': 'compute', 'kind': 'delegates'}]}
    descriptor = _a2a_descriptor_from_tags(agent, {'authored', 'peer'})
    descriptor['peer_ports'] = {'peer': 9123}
    ns = load_runtime(_generate(agent, tmp_path, descriptor), tmp_path)
    ports = []

    def dns(service, port, **kwargs):
        ports.append(port)
        return []

    monkeypatch.setattr(ns['socket'], 'getaddrinfo', dns)
    assert ns['_peer_replica_urls']('peer') == ['http://peer:9123']
    assert ns['_peer_replica_urls']('other') == ['http://other:8000']
    assert ns['_peer_replica_urls']('peer', 4321) == ['http://peer:4321']
    assert ports == [9123, 8000, 4321]


def test_first_and_later_a2a_turns_clear_stale_replies_without_errors(tmp_path, load_runtime, caplog):
    agent = _model()
    agent._human_facing = True
    agent._a2a = {'inbound': [{'peer': 'Peer', 'target_state': 'receive'}], 'outbound': [
        {'peer': 'Peer', 'state': 'compute', 'kind': 'delegates'}]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'})), tmp_path)
    session = ns['Session']('human', ns['agent'], ns['agent'].ui)
    session.call_manage_transition = lambda: None
    tasks = []
    ns['_a2a_send_edges'] = lambda task, edges, session=None: tasks.append(task) or []
    ns['compute']._body = lambda session: None  # A state may author no text reply on this turn.
    ns['_a2a_wrap'](ns['compute'], [{'service': 'peer', 'label': 'Peer', 'kind': 'delegates'}], [])
    import logging
    with caplog.at_level(logging.ERROR):
        session.event = ns['ReceiveTextEvent']('first')
        ns['compute']._body(session)
        session.set('_a2a_last_reply', 'stale captured reply')
        session.set('a2a_result', None)  # There is no intentional stored result in this scenario.
        session.event = ns['ReceiveTextEvent']('second')
        ns['compute']._body(session)
    assert tasks == ['first', 'second']
    assert not caplog.records


@pytest.mark.parametrize('a2a_enabled', [False, True])
def test_deployment_metadata_survives_personalization_render(tmp_path, load_runtime, a2a_enabled):
    agent = _model()
    profile = _model('Profile')
    profile.states[1].set_body(Body('receive_body', actions=[AgentReply('Personalized receipt')]))
    agent.new_tool('declared', description='Tool metadata', code='def internal_name(session):\n    return 1\n')
    agent.new_skill('product_lkno', description='Skill metadata', content='# Different heading')
    config = {'personalizationMapping': [{'name': 'Reader', 'configuration': {},
                                         'user_profile': {}, 'agent_model': profile}]}
    descriptor = None
    if a2a_enabled:
        agent._a2a = {'outbound': [], 'inbound': [{'peer': 'Peer', 'target_state': 'receive'}]}
        descriptor = _a2a_descriptor_from_tags(agent, {'authored', 'peer'})
    BAFGenerator(agent, str(tmp_path), config=config, generation_mode=GenerationMode.CODE_ONLY,
                 test_mode=True, a2a_descriptor=descriptor, deployment_component_metadata=True).generate()
    code = (tmp_path / 'Authored.py').read_text(encoding='utf-8')
    _singleton_checks(code)
    assert 'receive_Reader' in code and 'platform.reply_gui(session, order_form)' in code
    ns = load_runtime(code, tmp_path)
    assert set(ns['agent']._tools) == {'declared'}
    assert ns['agent']._tools['declared'].description == 'Tool metadata'
    assert set(ns['agent']._skills) == {'product_lkno'}
    assert ns['agent']._skills['product_lkno'].description == 'Skill metadata'
    if a2a_enabled:
        result = asyncio.run(ns['handle'](message='profile task', user_profile='Reader'))
        assert result['reply'] == 'Personalized receipt\n\nauthored answer: profile task'
        assert ns['default_llm'].calls[0][1].current_state.name == 'wait_Reader'


@pytest.mark.parametrize('peer_fails', [False, True])
def test_wrapped_human_reply_uses_real_baf_platform_identity(tmp_path, load_runtime, peer_fails):
    from baf.platforms.websocket.websocket_platform import WebSocketPlatform as BAFWebSocketPlatform

    agent = _model()
    agent._human_facing = True
    agent._a2a = {'inbound': [], 'outbound': [{'peer': 'Peer', 'state': 'receive'}]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'})), tmp_path)
    platform = BAFWebSocketPlatform(ns['agent'], use_ui=False)
    ns['platform'].human_platform = platform
    session = ns['Session']('human', ns['agent'], platform)
    session.call_manage_transition = lambda: None
    session._current_state = ns['receive']
    session.event = ns['ReceiveTextEvent']('order', session.id, human=True)
    payloads, tasks = [], []
    platform._connections[session.id] = SimpleNamespace(send=lambda data: payloads.append(json.loads(data)))

    def send(task, edges, session=None):
        tasks.append(task)
        if peer_fails:
            raise RuntimeError('peer failed')
        return [('peer', 'reply', 'remote output')]

    ns['_a2a_send_edges'] = send
    if peer_fails:
        with pytest.raises(RuntimeError, match='peer failed'):
            ns['receive']._body(session)
    else:
        ns['receive']._body(session)
        assert 'remote output' in payloads[-1]['message']
    assert payloads[0]['message'] == 'Received order'
    assert tasks == ['Received order']
    assert session.platform is platform


@pytest.mark.parametrize('parameters', [
    {'reasoning_effort': 'none', 'max_completion_tokens': 500},
    {'temperature': 0.3, 'top_p': 0.8},
])
@pytest.mark.parametrize('legacy', [False, True])
def test_governance_uses_real_baf_llm_configured_parameters(tmp_path, load_runtime, parameters, legacy):
    from baf.nlp.llm.llm_openai_api import LLMOpenAI as BAFOpenAI

    agent = _governed_agent()
    if legacy:
        agent = Agent('Owner')
        agent.new_state('initial', initial=True)
        agent._a2a = {'outbound': [{'peer': 'Peer', 'state': 'missing'}], 'inbound': [{'peer': 'Peer'}]}
        agent._governance = [{'policy_type': 'VotingPolicy', 'ratio': 0.5, 'requires_human': False,
                             'participants': [{'name': 'Peer', 'confidence': 1.0}], 'producers': ['Peer'],
                             'instruction': 'Vote', 'summary': 'Policy facts'}]
    agent.new_llm(name='configured_model', provider='openai', parameters=dict(parameters))
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'owner', 'peer'})), tmp_path)
    llm = BAFOpenAI(agent=ns['agent'], name='configured_model', parameters=dict(parameters))
    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='BALLOT: C1'))])

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    ns['reply_llm'] = llm
    if legacy:
        assert ns['_gov_owner_ballot'](None, '[C1] candidate', ['C1']) == 'BALLOT: C1'
    else:
        assert ns['_fm_owner_ballot']('[C1] candidate', ['C1']) == 'BALLOT: C1'
    assert len(requests) == 1
    assert {k: v for k, v in requests[0].items() if k not in {'model', 'messages'}} == parameters
    assert llm.parameters == parameters


@pytest.mark.parametrize('owner_fails', [False, True])
@pytest.mark.parametrize('silent_peer', [False, True])
def test_governed_http_round_trip_serves_ballots_on_initiating_replica(
        tmp_path, load_runtime, monkeypatch, owner_fails, silent_peer):
    from aiohttp import ClientSession, ClientTimeout, web
    from baf.platforms.a2a.message_router import A2ARouter
    from besser.BUML.metamodel.state_machine.state_machine import CustomCodeAction
    import urllib.request

    peer = Agent('Peer')
    peer.new_state('initial', initial=True)
    work = peer.new_state('work')
    work.set_body(Body('work_body', actions=[CustomCodeAction(source="""
def work_body(session: Session) -> None:
    session.reply('BALLOT: C1' if '[C1]' in session.event.message else 'CANDIDATE_OK')
""")]))
    if silent_peer:
        peer.new_llm(name='authored_model', provider='openai', parameters={})
        work.set_body(Body('work_body', actions=[
            LLMReply(send_reply=False, store_in_session='local_result'),
        ]))
    peer._a2a = {'inbound': [{'peer': 'Owner', 'target_state': 'work', 'flow': 'work'}],
                 'outbound': [{'peer': 'Owner', 'state': 'work', 'target_gateway': 'gw1'}]}
    code = _generate(peer, tmp_path / 'peer', _a2a_descriptor_from_tags(peer, {'peer', 'owner'}))
    peers = [load_runtime(code, tmp_path / f'peer_{index}') for index in range(2)]
    if silent_peer:
        for ns in peers:
            ns['default_llm'].predict = (
                lambda message, session=None, **kwargs:
                'BALLOT: C1' if '[C1]' in message else 'CANDIDATE_OK'
            )
    owner_model = _governed_agent()
    owner = load_runtime(_generate(owner_model, tmp_path / 'owner',
                                  _a2a_descriptor_from_tags(owner_model, {'owner', 'peer'})), tmp_path / 'owner')
    if owner_fails:
        def fail_merge(*args, **kwargs):
            raise RuntimeError('owner merge failed')

        owner['_run_fanout'] = fail_merge
    original_open = urllib.request.urlopen
    # A broken generated dispatcher must fail promptly instead of waiting 120 seconds.
    monkeypatch.setattr(urllib.request, 'urlopen', lambda request, timeout=120:
                        original_open(request, timeout=min(timeout, 2)))

    async def scenario():
        runners, ballots = [], []

        async def serve(ns, replica):
            router = A2ARouter()

            async def handle(**params):
                result = await ns['handle'](**params)
                if params.get('leaf'):
                    ballots.append(replica)
                return result

            router.register('handle', handle)
            app = web.Application()
            app.router.add_post('/a2a', router.aiohttp_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            runners.append(runner)
            listener = socket.socket()
            listener.bind(('127.0.0.1', 0))
            listener.setblocking(False)
            port = listener.getsockname()[1]
            await web.SockSite(runner, listener).start()
            return f'http://127.0.0.1:{port}'

        try:
            urls = [await serve(ns, index) for index, ns in enumerate(peers)]
            owner_url = await serve(owner, 'owner')
            for ns in peers:
                ns['_peer_replica_urls'] = lambda service: [owner_url]
            owner['_peer_replica_urls'] = lambda service: urls
            async with ClientSession(timeout=ClientTimeout(total=8)) as client:
                response = await client.post(urls[0] + '/a2a', json={
                    'jsonrpc': '2.0', 'method': 'handle', 'id': 1,
                    'params': {'message': 'work', 'from': 'owner', 'flow': 'work'},
                })
                envelope = await response.json()
            if owner_fails:
                assert 'error' in envelope, envelope
                assert 'result' not in envelope
                return
            assert 'error' not in envelope, envelope
            assert 'CANDIDATE_OK' in envelope['result']['reply']
            assert 'winner C1' in envelope['result']['reply']
            assert 'votes [C1=2]' in envelope['result']['reply']
            assert 'abstain 0' in envelope['result']['reply']
            assert sorted(ballots) == [0, 1]
        finally:
            for runner in reversed(runners):
                await runner.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize('full_result', [False, True])
def test_a2a_client_rejects_json_rpc_error_envelopes(tmp_path, load_runtime, monkeypatch, full_result):
    import io
    import urllib.request

    agent = _model()
    agent._a2a = {'inbound': [], 'outbound': [
        {'peer': 'Peer', 'state': 'receive', 'target_gateway': 'gw1'},
    ]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'})), tmp_path)
    envelope = {'jsonrpc': '2.0', 'id': 1, 'error': {'code': -32603, 'message': 'owner failed'}}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs:
                        io.BytesIO(json.dumps(envelope).encode()))
    call = ns['_a2a_call_full' if full_result else '_a2a_call']
    with pytest.raises(RuntimeError, match='owner failed'):
        call('http://peer', 'peer', 'candidate', 'gw1')


@pytest.mark.parametrize('healthy_replica', [False, True])
def test_merge_send_requires_a_successful_owner_replica(tmp_path, load_runtime, healthy_replica):
    agent = _model()
    agent._a2a = {'inbound': [], 'outbound': [
        {'peer': 'Peer', 'state': 'receive', 'target_gateway': 'gw1'},
    ]}
    ns = load_runtime(_generate(agent, tmp_path, _a2a_descriptor_from_tags(agent, {'authored', 'peer'})), tmp_path)
    ns['_peer_replica_urls'] = lambda service: ['http://failed', 'http://other']

    def call(url, *args):
        if url == 'http://other' and healthy_replica:
            return {'reply': 'governed output'}
        raise RuntimeError('owner failed')

    ns['_a2a_call_full'] = call
    if healthy_replica:
        assert ns['_merge_send']('local candidate', 'peer', 'gw1') == ('governed output', None)
    else:
        with pytest.raises(RuntimeError, match='No successful governed merge'):
            ns['_merge_send']('local candidate', 'peer', 'gw1')

def _capture_model(send_reply=False, store=None):
    model = Agent('Silent')
    model.platforms.append(WebSocketPlatform())
    model.new_llm(name='authored_model', provider='openai', parameters={})
    welcome = model.new_state('welcome', initial=True)
    work = model.new_state('work')
    after = model.new_state('after')
    welcome.set_body(Body('welcome_body', actions=[AgentReply('Welcome')]))
    work.set_body(Body('work_body', actions=[
        LLMReply(prompt='Assess', send_reply=send_reply, store_in_session=store),
    ]))
    after.set_body(Body('after_body', actions=[
        LLMReply(prompt='This later task must not run during a leaf call'),
    ]))
    work.go_to(after)
    model._human_facing = True
    model._a2a = {
        'inbound': [{'peer': 'Peer', 'target_state': 'work', 'flow': 'work'}],
        'outbound': [{'peer': 'Peer', 'state': 'work', 'kind': 'delegates'}],
    }
    return model


@pytest.mark.parametrize('send_reply', [False, True])
@pytest.mark.parametrize('store', [None, 'answer'])
def test_fresh_result_forwarding_preserves_visibility(
        tmp_path, load_runtime, caplog, send_reply, store):
    import logging

    model = _capture_model(send_reply, store)
    descriptor = _a2a_descriptor_from_tags(model, {'silent', 'peer'})
    ns = load_runtime(_generate(model, tmp_path, descriptor), tmp_path)
    tasks = []
    ns['_a2a_send_edges'] = lambda task, edges, session=None: tasks.append(task) or []
    ns['default_llm'].predict = (
        lambda message, session=None, **kwargs:
        'final response' if message.startswith('Task:') else 'same local assessment'
    )
    session = ns['Session']('human', ns['agent'], ns['agent'].ui)
    session.call_manage_transition = lambda: None
    session.set('_a2a_last_reply', 'stale reply')
    session.set('a2a_result', 'stale final')
    session.set('answer', 'stale stored value')
    with caplog.at_level(logging.INFO):
        for incoming in ['first', 'second']:
            session._current_state = ns['work']
            session.event = ns['ReceiveTextEvent'](incoming)
            ns['work']._body(session)

    # The second turn stores an identical value; it is still newly produced.
    assert tasks == ['same local assessment', 'same local assessment']
    messages = [value for kind, value in ns['agent'].ui.sent if kind == 'reply']
    if send_reply:
        assert messages == [
            'same local assessment', 'final response',
            'same local assessment', 'final response',
        ]
    else:
        assert messages == ['final response', 'final response']
    if store:
        assert session.get(store) == 'same local assessment'
    assert sum('[a2a-result]' in record.getMessage() for record in caplog.records) == 2
    assert session._a2a_active_capture is None


@pytest.mark.parametrize('bound', [False, True])
def test_silent_leaf_stops_and_concurrent_results_are_isolated(
        tmp_path, load_runtime, bound):
    model = _capture_model(store='answer')
    if not bound:
        model._a2a['outbound'] = []
    descriptor = _a2a_descriptor_from_tags(model, {'silent', 'peer'})
    ns = load_runtime(_generate(model, tmp_path, descriptor), tmp_path)

    def unexpected_peer_call(*args, **kwargs):
        raise AssertionError('Leaf request attempted outbound A2A')

    ns['_a2a_send_edges'] = unexpected_peer_call
    result = asyncio.run(ns['handle'](message='ballot', flow='work', leaf=True))
    assert result['reply'] == 'authored answer: ballot'
    assert len(ns['default_llm'].calls) == 1
    assert ns['agent'].ui.sent == []
    ns['default_llm'].calls.clear()

    async def concurrent():
        return await asyncio.gather(
            ns['handle'](message='one', flow='work', leaf=True),
            ns['handle'](message='two', flow='work', leaf=True),
        )

    results = asyncio.run(concurrent())
    assert [result['reply'] for result in results] == [
        'authored answer: one', 'authored answer: two',
    ]
    assert len(ns['default_llm'].calls) == 2
    assert len({call[1].id for call in ns['default_llm'].calls}) == 2


def test_empty_computed_leaf_result_is_an_error(tmp_path, load_runtime):
    model = _capture_model()
    descriptor = _a2a_descriptor_from_tags(model, {'silent', 'peer'})
    ns = load_runtime(_generate(model, tmp_path, descriptor), tmp_path)
    ns['default_llm'].predict = lambda **kwargs: ''
    with pytest.raises(ValueError, match='Empty A2A result'):
        asyncio.run(ns['handle'](
            message='input must not substitute for an empty result',
            flow='work', leaf=True,
        ))


@pytest.mark.parametrize('send_reply', [False, True])
def test_ordinary_agent_reply_flags_remain_unchanged(
        tmp_path, load_runtime, send_reply):
    model = _capture_model(send_reply=send_reply, store='answer')
    del model._a2a
    code = _generate(model, tmp_path)
    assert '_a2a_record_result' not in code
    ns = load_runtime(code, tmp_path)
    session = ns['Session']('human', ns['agent'], ns['agent'].ui)
    session.call_manage_transition = lambda: None
    session._current_state = ns['work']
    session.event = ns['ReceiveTextEvent']('ordinary input')
    ns['work']._body(session)
    assert session.get('answer') == 'authored answer: ordinary input'
    assert ns['agent'].ui.sent == (
        [('reply', 'authored answer: ordinary input')] if send_reply else []
    )
