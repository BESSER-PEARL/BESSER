"""Calling a local ``async def`` without ``await`` must be reported at write time.

Run 36e9c8a6 (2026-09-18) shipped a hotel app whose ``produceBill`` route did
``booking_computeAndReturnAmountOwed(booking_id, database)["amountOwed"]`` — a
call to a sibling async route function. ast.parse accepted it, pyflakes
accepted it, Phase 3 passed with zero blockers, and the app booted; only
POSTing to the route produced ``'coroutine' object is not subscriptable``.

The check flagged 0 of 564 known-good files (BESSER, the modeling agent,
fastapi, starlette, anyio, httpx) and exactly this one call.
"""
from besser.generators.llm.write_diagnostics import diagnose_written_content


def codes(content, path="routers/booking_methods.py"):
    return [f["code"] for f in diagnose_written_content(path, content)]


SHIPPED = '''
from fastapi import Body, Depends

@router.post("/booking/methods/computeAndReturnAmountOwed/")
async def booking_computeAndReturnAmountOwed(booking_id: int = Body(...), database=None):
    return {"amountOwed": 0.0}

@router.post("/booking/methods/produceBill/")
async def booking_produceBill(booking_id: int = Body(...), database=None):
    amount_owed = booking_computeAndReturnAmountOwed(booking_id, database)["amountOwed"]
    return {"amount": amount_owed}
'''


def test_the_shipped_defect_is_reported():
    assert "unawaited-coroutine" in codes(SHIPPED)


def test_the_message_names_the_function_and_the_fix():
    finding = next(f for f in diagnose_written_content("r.py", SHIPPED)
                   if f["code"] == "unawaited-coroutine")
    assert "booking_computeAndReturnAmountOwed" in finding["message"]
    assert "await" in finding["message"]
    assert finding["line"] == 10


def test_an_awaited_call_is_clean():
    assert "unawaited-coroutine" not in codes(
        SHIPPED.replace("amount_owed = booking_computeAndReturnAmountOwed(",
                        "amount_owed = (await booking_computeAndReturnAmountOwed("
                        ).replace(', database)["amountOwed"]', ', database))["amountOwed"]'))


def test_an_async_generator_may_be_called_without_await():
    """StreamingResponse(_stream()) is correct — BESSER's own router does it."""
    assert codes('''
from fastapi.responses import StreamingResponse

async def _stream():
    yield b"chunk"

async def download():
    return StreamingResponse(_stream())
''') == []


def test_a_nested_yield_does_not_make_the_outer_function_a_generator():
    assert "unawaited-coroutine" in codes('''
async def compute():
    def inner():
        yield 1
    return 2

async def caller():
    return compute()
''')


def test_a_scheduled_coroutine_is_not_flagged():
    assert codes('''
import asyncio

async def work():
    return 1

async def caller():
    asyncio.create_task(work())
    return await asyncio.gather(work(), work())
''') == []


def test_a_bare_function_reference_is_not_a_call():
    assert codes('''
from fastapi import Depends

async def get_user():
    return None

async def route(user=Depends(get_user)):
    return user
''') == []


def test_a_sync_function_of_the_same_name_shape_is_not_flagged():
    assert codes('''
def helper():
    return 1

async def route():
    return helper()
''') == []


def test_a_syntax_error_still_wins():
    assert codes("async def f(:\n    pass\n") == ["syntax"]
