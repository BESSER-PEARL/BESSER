"""A starter B-UML model: a BPMN process that checks an order, then ships or rejects it.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate bpmn MODEL_FILE -o out        generate BPMN 2.0 XML ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/bpmn.html
"""

from besser.BUML.metamodel.bpmn import (
    BPMNModel,
    EndEvent,
    Gateway,
    GatewayType,
    Process,
    SequenceFlow,
    StartEvent,
    Task,
    TaskType,
)

# Events: where the process starts and where it ends
order_received = StartEvent(name="Order received")
order_shipped = EndEvent(name="Order shipped")
order_rejected = EndEvent(name="Order rejected")

# Tasks: the work to do; the task type says who or what does it
check_stock = Task(name="Check stock", task_type=TaskType.SERVICE)
ship_order = Task(name="Ship order", task_type=TaskType.USER)
notify_customer = Task(name="Notify customer", task_type=TaskType.SEND)

# Gateway: an exclusive gateway takes exactly one of its outgoing flows
in_stock = Gateway(name="In stock?", gateway_type=GatewayType.EXCLUSIVE)

# Sequence flows: the order of the flow nodes; a flow's name is its label, the default flow is taken otherwise
flows = {
    SequenceFlow(order_received, check_stock),
    SequenceFlow(check_stock, in_stock),
    SequenceFlow(in_stock, ship_order, name="yes"),
    SequenceFlow(in_stock, notify_customer, name="no", is_default=True),
    SequenceFlow(ship_order, order_shipped),
    SequenceFlow(notify_customer, order_rejected),
}

# Process: the flow nodes and the sequence flows between them
order_process = Process(
    name="Order handling",
    flow_nodes={order_received, check_stock, in_stock, ship_order, notify_customer, order_shipped, order_rejected},
    sequence_flows=flows,
)

# The model. The besser command reads every model bound to a top-level name of this file.
order_model = BPMNModel(name="OrderHandling", processes={order_process})
