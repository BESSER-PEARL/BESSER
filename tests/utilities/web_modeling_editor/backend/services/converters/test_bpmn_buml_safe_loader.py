"""BPMN B-UML import runs through the AST-allowlist loader, not a raw ``exec``.

``bpmn_buml_to_json`` used ``exec`` with a trimmed ``__builtins__``, which does not
contain a payload: any object in scope leads back to ``object`` through
``__class__.__mro__``. Every other converter already used ``safe_load_buml``.
"""
import pytest

from besser.BUML.metamodel.bpmn import (
    Association,
    BPMNModel,
    CallActivity,
    Collaboration,
    DataAssociation,
    DataObject,
    DataStore,
    EndEvent,
    EventDefinitionType,
    EventDirection,
    Gateway,
    GatewayType,
    Group,
    IntermediateEvent,
    Lane,
    LoopCharacteristics,
    MessageFlow,
    Participant,
    Process,
    SequenceFlow,
    StartEvent,
    SubProcess,
    Task,
    TaskType,
    TextAnnotation,
    Transaction,
)
from besser.utilities.buml_code_builder.bpmn_model_builder import bpmn_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
    bpmn_buml_to_json,
    bpmn_object_to_json,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


def _rich_model() -> BPMNModel:
    """One of every element kind the builder emits, with layout on some of them."""
    start = StartEvent(name="order in", event_definition=EventDefinitionType.MESSAGE,
                       layout={"id": "n-start", "x": -10, "y": 20.5})
    check = Task(name="check 'stock'", task_type=TaskType.SERVICE,
                 loop_characteristics=LoopCharacteristics.STANDARD_LOOP)
    split = Gateway(name="in stock?", gateway_type=GatewayType.EXCLUSIVE)
    wait = IntermediateEvent(name="wait", direction=EventDirection.CATCH,
                             event_definition=EventDefinitionType.TIMER)
    inner = Task(name="pick")
    sub = SubProcess(name="fulfil", flow_nodes={inner})
    tx = Transaction(name="charge")
    call = CallActivity(name="ship")
    end = EndEvent(name="done")
    flows = {
        SequenceFlow(start, check), SequenceFlow(check, split),
        SequenceFlow(split, wait, name="no"), SequenceFlow(split, sub, name="yes", is_default=True),
        SequenceFlow(wait, tx), SequenceFlow(sub, tx), SequenceFlow(tx, call), SequenceFlow(call, end),
    }
    note = TextAnnotation(text="multi\nline note")
    group = Group(name="billing")
    order = DataObject(name="order")
    shop = Process(name="Shop", flow_nodes={start, check, split, wait, sub, tx, call, end},
                   sequence_flows=flows)
    shop.add_artifact(note)
    shop.add_artifact(group)
    shop.add_data_object(order)
    lane = Lane(name="clerk")
    shop.add_lane(lane)
    lane.add_flow_node(check)
    shop.add_association(Association(note, check))
    shop.add_data_association(DataAssociation(check, order))

    receive = StartEvent(name="receive")
    customer = Process(name="Customer", flow_nodes={receive})
    collaboration = Collaboration(name="c")
    collaboration.add_participant(Participant(name="Shop", process=shop))
    collaboration.add_participant(Participant(name="Customer", process=customer))
    collaboration.add_message_flow(MessageFlow(receive, start, name="order"))

    model = BPMNModel(name="Rich", processes={shop, customer}, collaboration=collaboration)
    model.add_data_store(DataStore(name="db"))
    return model


def _elements(json_model):
    return sorted(
        (e["type"], e["name"], e.get("taskType"), e.get("marker"), e.get("eventType"), e.get("gatewayType"))
        for e in json_model["elements"].values()
    )


def _relationships(json_model):
    return sorted(
        (r["type"], r["name"], r.get("flowType"), r.get("isDefault") if r.get("flowType") == "sequence" else None)
        for r in json_model["relationships"].values()
    )


def test_every_element_kind_round_trips_through_the_safe_loader():
    model = _rich_model()
    source = bpmn_model_to_code(model)

    out = bpmn_buml_to_json(source)

    direct = bpmn_object_to_json(model)
    assert _elements(out) == _elements(direct)
    assert _relationships(out) == _relationships(direct)
    assert len(out["elements"]) >= 15


@pytest.mark.parametrize("payload", [
    "x = __import__('os').system('echo pwned')\n",
    "x = BPMNModel.__class__.__mro__\n",
    "x = BPMNModel(name='a').__dict__\n",
    "x = set.__subclasses__()\n",
    "x = '{0.__class__}'.format(set)\n",
    "import os\nos.system('echo pwned')\n",
    "def f():\n    pass\n",
])
def test_a_malicious_payload_is_refused(payload):
    with pytest.raises(ConversionError, match="failed to execute"):
        bpmn_buml_to_json(payload)


def test_the_former_builtins_are_no_longer_reachable():
    """``len`` / ``range`` / ``print`` were handed to user code; nothing emitted uses them."""
    with pytest.raises(ConversionError, match="failed to execute"):
        bpmn_buml_to_json("n = len([1])\nbpmn_model = BPMNModel(name='x')\n")
