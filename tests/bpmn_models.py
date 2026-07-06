"""Shared programmatic BPMN model builders for the test suite.

These plain-function builders are reused by the BPMN generator tests
(``tests/generators/bpmn/test_bpmn_generator.py``) and the BPMN code-builder tests
(``tests/utilities/buml_code_builder/test_bpmn_model_builder.py``) so the model shapes
stay in one place. (The metamodel fixtures in ``tests/BUML/metamodel/bpmn/conftest.py``
are deliberately separate: they are different, name-sensitive models exercised by the
metamodel suite.)
"""

from __future__ import annotations

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    Collaboration,
    EndEvent,
    Gateway,
    GatewayType,
    Lane,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
)


def _poolless_model() -> BPMNModel:
    """Pool-less StartEvent -> Task -> EndEvent."""
    s = StartEvent(name="received")
    t = Task(name="Review", task_type=TaskType.USER)
    e = EndEvent(name="done")
    p = Process(name="P", flow_nodes={s, t, e},
                sequence_flows={SequenceFlow(s, t), SequenceFlow(t, e)})
    return BPMNModel(name="Poolless", processes={p})


def _two_pool_model() -> BPMNModel:
    """Two pools, one task each, one MessageFlow."""
    t1 = Task(name="Place order")
    t2 = Task(name="Ship")
    p_buyer = Process(name="Buyer", flow_nodes={t1})
    p_seller = Process(name="Seller", flow_nodes={t2})
    part_buyer = Participant(name="Buyer", process=p_buyer)
    part_seller = Participant(name="Seller", process=p_seller)
    coll = Collaboration(name="Buyer-Seller",
                         participants={part_buyer, part_seller},
                         message_flows={MessageFlow(t1, t2, name="order")})
    return BPMNModel(name="Buyer-Seller", processes={p_buyer, p_seller},
                     collaboration=coll)


def _gateway_model() -> BPMNModel:
    """ExclusiveGateway with a marked default flow."""
    s = StartEvent(name="")
    g = Gateway(name="?", gateway_type=GatewayType.EXCLUSIVE)
    a = Task(name="A")
    b = Task(name="B")
    e1 = EndEvent(name="")
    e2 = EndEvent(name="")
    f0 = SequenceFlow(s, g)
    f1 = SequenceFlow(g, a, name="happy")
    f2 = SequenceFlow(g, b, name="default", is_default=True)
    fa = SequenceFlow(a, e1)
    fb = SequenceFlow(b, e2)
    p = Process(name="Gateway",
                flow_nodes={s, g, a, b, e1, e2},
                sequence_flows={f0, f1, f2, fa, fb})
    return BPMNModel(name="Gateway", processes={p})


def _subprocess_model() -> BPMNModel:
    """SubProcess containing two flow nodes and one inner sequence flow."""
    inner_t = Task(name="inner")
    inner_e = EndEvent(name="inner end")
    sub = SubProcess(name="Sub",
                     flow_nodes={inner_t, inner_e},
                     sequence_flows={SequenceFlow(inner_t, inner_e)})
    s = StartEvent(name="")
    e = EndEvent(name="")
    p = Process(name="Subprocess",
                flow_nodes={s, sub, e},
                sequence_flows={SequenceFlow(s, sub), SequenceFlow(sub, e)})
    return BPMNModel(name="Subprocess", processes={p})


def _lane_model() -> BPMNModel:
    """Single process partitioned by one lane holding one task."""
    t = Task(name="Review")
    lane = Lane(name="Reviewer", flow_nodes={t})
    p = Process(name="P", flow_nodes={t}, lanes={lane})
    return BPMNModel(name="Lanes", processes={p})
