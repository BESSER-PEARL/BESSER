"""Shared fixtures for the BPMN metamodel tests.

Three reusable, well-formed models that ``validate()`` cleanly (no errors, no warnings):
``minimal_process_model`` (pool-less), ``collaboration_model`` (two pools + a message flow),
and ``gateway_model`` (an exclusive gateway with a default flow).
"""

import pytest

from besser.BUML.metamodel.bpmn import (
    BPMNModel, Collaboration, EndEvent, Gateway, GatewayType, MessageFlow, Participant,
    Process, SequenceFlow, StartEvent, Task, TaskType,
)


@pytest.fixture
def minimal_process_model() -> BPMNModel:
    """Pool-less model: StartEvent -> Task -> EndEvent."""
    start = StartEvent("start")
    task = Task("do work", task_type=TaskType.USER)
    end = EndEvent("end")
    process = Process(
        "order_process",
        flow_nodes={start, task, end},
        sequence_flows={SequenceFlow(start, task), SequenceFlow(task, end)},
    )
    return BPMNModel("minimal_process_model", processes={process})


@pytest.fixture
def collaboration_model() -> BPMNModel:
    """Two pools, each StartEvent -> Task -> EndEvent, with a message flow between them."""
    s1, t1, e1 = StartEvent("s1"), Task("send"), EndEvent("e1")
    p1 = Process(
        "process_1",
        flow_nodes={s1, t1, e1},
        sequence_flows={SequenceFlow(s1, t1), SequenceFlow(t1, e1)},
    )
    s2, t2, e2 = StartEvent("s2"), Task("receive"), EndEvent("e2")
    p2 = Process(
        "process_2",
        flow_nodes={s2, t2, e2},
        sequence_flows={SequenceFlow(s2, t2), SequenceFlow(t2, e2)},
    )
    participant_1 = Participant("Pool 1", process=p1)
    participant_2 = Participant("Pool 2", process=p2)
    message = MessageFlow(t1, t2, name="order")
    collaboration = Collaboration(
        "collaboration",
        participants={participant_1, participant_2},
        message_flows={message},
    )
    return BPMNModel("collaboration_model", processes={p1, p2}, collaboration=collaboration)


@pytest.fixture
def gateway_model() -> BPMNModel:
    """An exclusive gateway with two outgoing flows, one marked as the default."""
    start = StartEvent("start")
    gateway = Gateway("choice", gateway_type=GatewayType.EXCLUSIVE)
    path_a, path_b = Task("path A"), Task("path B")
    end_a, end_b = EndEvent("end A"), EndEvent("end B")
    process = Process(
        "gateway_process",
        flow_nodes={start, gateway, path_a, path_b, end_a, end_b},
        sequence_flows={
            SequenceFlow(start, gateway),
            SequenceFlow(gateway, path_a, is_default=True),
            SequenceFlow(gateway, path_b),
            SequenceFlow(path_a, end_a),
            SequenceFlow(path_b, end_b),
        },
    )
    return BPMNModel("gateway_model", processes={process})
