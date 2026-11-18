"""A starter B-UML state machine: the life of an order, from its creation to its delivery.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

No generator reads a state machine alone ('besser generators' lists what each one reads).
B-UML reference: https://besser.readthedocs.io/en/latest/buml_language.html
"""

from besser.BUML.metamodel.state_machine import (
    Body,
    Condition,
    Event,
    Session,
    StateMachine,
)

order_lifecycle = StateMachine(name="OrderLifecycle")

# States: exactly one is initial, and it is created first; a final state has no outgoing transition
created = order_lifecycle.new_state(name="created", initial=True)
paid = order_lifecycle.new_state(name="paid")
shipped = order_lifecycle.new_state(name="shipped")
delivered = order_lifecycle.new_state(name="delivered", final=True)
cancelled = order_lifecycle.new_state(name="cancelled", final=True)


# Bodies: the function a state runs when the machine enters it. The session stores the data of one run.
def paid_body(session: Session):
    session.set("invoice_sent", True)


def cancelled_body(session: Session):
    session.set("refund_due", session.get("amount_paid"))


paid.set_body(Body(name="paid_body", callable=paid_body))
cancelled.set_body(Body(name="cancelled_body", callable=cancelled_body))

# Events: what happens to the machine from outside
payment_received = Event(name="payment_received")
parcel_sent = Event(name="parcel_sent")
parcel_received = Event(name="parcel_received")
order_cancelled = Event(name="order_cancelled")


# Conditions: a function that says whether a transition may be taken
def refund_allowed(session: Session, params: dict):
    return session.get("amount_paid") <= session.get("refund_limit")


refund_allowed_condition = Condition(name="refund_allowed", callable=refund_allowed)

# Transitions: from a state, on an event and/or a condition, to another state
created.when_event(payment_received).go_to(paid)
created.when_event(order_cancelled).go_to(cancelled)
paid.when_event(parcel_sent).go_to(shipped)
paid.when_event(order_cancelled).with_condition(refund_allowed_condition).go_to(cancelled)
shipped.when_event(parcel_received).go_to(delivered)
