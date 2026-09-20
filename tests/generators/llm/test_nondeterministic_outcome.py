"""A business outcome must not be decided by a coin flip.

The inventory spec says an order "can have its payment confirmed, which
reports back whether the payment went through" and names nothing that
decides it - no state to move to, no literal on OrderStatus, no payment
record. Faced with the gap the model invented a decider: six delivered
apps in ``verification/spec-iterations`` answer ``confirmPayment`` with
``random.choice([True, False])`` (qr9osh7c, camnhfvj, dpuubqxt),
``random.random() < 0.9`` (d1h3rte7), ``< 0.8`` (px3e31xt), and nine Trues
and a False (iia3pn0_). All six are Qwen and all six are on that one
action. In every one the coin flip is the ENTIRE handler - zero
assignments, zero ``database.add``, zero ``database.commit`` - so the app
holds no payment state at all.

The damage is not flakiness, it is unfalsifiability. Such an app passes a
single probe 50-90% of the time, so it lands in the "known-working"
corpus and poisons every single-sample measurement taken against it: under
the rescored labels px3e31xt and qr9osh7c both score 15/15, and px3e31xt
sits inside the original 74 known-working apps.

Precision is the whole difficulty, because ``random`` in generated code is
usually fine. Four other delivered apps (2bwg9ufa, fg85wf5c, ppl1t8cj,
uhidoti7 - all Qwen, all the same hotel ``produceBill`` action, all the
same ``import random``) build a bill number out of ``random.choices`` over
an alphabet, and they are correct. So the check keys on the draw being a
BOOLEAN that reaches the response from inside a route handler, never on
``random`` being imported. The bodies below are copied verbatim from those
ten apps.
"""

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    PrimitiveDataType,
    Property,
)
from besser.generators.llm.contract_checks import build_data_contract, lint_file
from besser.generators.llm.orchestrator import _classify_issue

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")

HEADER = (
    "from fastapi import APIRouter, Depends, HTTPException\n"
    "router = APIRouter()\n\n"
)


def _contract():
    order = Class(name="Order")
    order.attributes = {Property(name="id", type=IntegerType, is_id=True)}
    return build_data_contract(DomainModel(name="Inventory", types={order}))


def _lint(body: str, path: str = "backend/routers/order_methods.py"):
    return lint_file(path, HEADER + body, _contract())


def _coin_flips(body: str, path: str = "backend/routers/order_methods.py"):
    return [f for f in _lint(body, path) if "coin flip" in f.message]


# --- the six delivered coin-flip handlers ----------------------------------

PX3E31XT = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    _order_object = database.query(Order).filter(Order.id == order_id).first()
    if _order_object is None:
        raise HTTPException(status_code=404, detail="Order not found")
    # For demonstration, we'll assume payment succeeds 80% of the time
    import random
    success = random.random() < 0.8

    return {
        "success": success,
        "message": "Payment confirmed" if success else "Payment failed",
    }
'''

CAMNHFVJ = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    import random
    success = random.choice([True, False])

    return {
        "paymentConfirmed": success,
        "message": "Payment confirmation simulated",
    }
'''

D1H3RTE7 = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    import random
    payment_successful = random.random() < 0.9

    return {"success": payment_successful,
            "message": "Payment confirmed" if payment_successful else "Payment failed"}
'''

IIA3PN0 = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    import random
    payment_successful = random.choice([True, True, True, True, True, True, True, True, True, False])

    return {"success": payment_successful, "message": "Payment confirmation simulated"}
'''

QR9OSH7C = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    import random
    success = random.choice([True, False])

    if success:
        return {"success": True, "message": "Payment confirmed successfully"}
    else:
        return {"success": False, "message": "Payment could not be confirmed"}
'''


def test_every_delivered_coin_flip_is_a_blocker():
    """All six shapes observed in the corpus, including the `if flip:` one."""
    for name, body in (("px3e31xt", PX3E31XT), ("camnhfvj", CAMNHFVJ),
                       ("d1h3rte7", D1H3RTE7), ("iia3pn0_", IIA3PN0),
                       ("qr9osh7c", QR9OSH7C)):
        findings = _coin_flips(body)
        assert len(findings) == 1, f"{name}: {findings}"
        assert findings[0].blocker, name
        issue = f"data contract: {findings[0].path}: {findings[0].message}"
        assert _classify_issue(issue).severity == "blocker", name


def test_the_message_names_the_honest_alternative():
    message = _coin_flips(PX3E31XT)[0].message
    assert "random.random() < 0.8" in message
    assert "501" in message


# --- the four delivered legitimate uses ------------------------------------

BILL_NUMBER_CHOICES = '''
@router.post("/booking/{booking_id}/methods/produceBill/")
async def execute_booking_produceBill(booking_id: int, database=Depends(get_db)):
    import random
    import string
    bill_number = "B" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    new_bill = Bill(billNumber=bill_number, booking_id=booking_id)
    database.add(new_bill)
    database.commit()
    return {"success": True, "billNumber": bill_number}
'''

BILL_NUMBER_RANDINT = '''
@router.post("/booking/{booking_id}/methods/produceBill/")
async def execute_booking_produceBill(booking_id: int, database=Depends(get_db)):
    from random import randint
    bill_number = f"BIL{randint(1000, 9999)}"
    database.add(Bill(billNumber=bill_number))
    database.commit()
    return {"success": True, "billNumber": bill_number}
'''

BILL_NUMBER_BARE_CHOICE = '''
@router.post("/booking/{booking_id}/methods/produceBill/")
async def execute_booking_produceBill(booking_id: int, database=Depends(get_db)):
    from random import choice
    import string
    bill_number = "B" + ".".join([choice(string.ascii_uppercase) for _ in range(3)])
    database.add(Bill(billNumber=bill_number))
    database.commit()
    return {"success": True, "billNumber": bill_number}
'''


def test_random_ids_are_not_flagged():
    """The bill-number generators from 2bwg9ufa / fg85wf5c / ppl1t8cj /
    uhidoti7. A string built out of an alphabet is not an outcome, and a
    bare `choice` imported from random must not widen the net either."""
    for name, body in (("choices", BILL_NUMBER_CHOICES),
                       ("randint", BILL_NUMBER_RANDINT),
                       ("bare choice", BILL_NUMBER_BARE_CHOICE)):
        assert _coin_flips(body, "backend/routers/booking_methods.py") == [], name


def test_demo_seeder_may_randomise_a_flag():
    """`random` outside a route handler decides nothing a client can see."""
    seeder = '''
import random

def seed(database):
    for i in range(20):
        database.add(Order(id=i, isPaid=random.choice([True, False])))
    database.commit()
'''
    assert _coin_flips(seeder, "backend/data/seed.py") == []


def test_random_that_never_reaches_the_response_is_not_flagged():
    """A coin flip used for sampling/jitter, not for the answer."""
    body = '''
@router.get("/order/{order_id}/")
async def read_order(order_id: int, database=Depends(get_db)):
    import random
    if random.random() < 0.01:
        logger.info("sampled trace")
    return database.query(Order).filter(Order.id == order_id).first()
'''
    assert _coin_flips(body) == []


def test_a_deterministic_decision_is_not_flagged():
    """The shape we actually want: the outcome read off stored state."""
    body = '''
@router.post("/order/{order_id}/methods/confirmPayment/")
async def execute_order_confirmPayment(order_id: int, database=Depends(get_db)):
    _order_object = database.query(Order).filter(Order.id == order_id).first()
    if _order_object is None:
        raise HTTPException(status_code=404, detail="Order not found")
    success = _order_object.status == "pending" and _order_object.totalAmount > 0
    if success:
        _order_object.isPaid = True
        database.commit()
    return {"success": success}
'''
    assert _coin_flips(body) == []
