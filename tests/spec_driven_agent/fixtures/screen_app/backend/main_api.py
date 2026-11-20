"""Stand-in for a generated backend: one route per outcome a screen can meet."""

import asyncio
import urllib.request

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.get("/api/orders")
def list_orders():
    return [{"id": 7, "total": 12.5}]


@app.get("/api/report")
def report():
    raise RuntimeError("the report query failed")


@app.get("/api/customers")
def list_customers():
    raise HTTPException(status_code=404, detail="no such collection")


@app.get("/api/slow")
async def slow():
    await asyncio.sleep(120)
    return []


@app.get("/api/items")
def list_items():
    return []


@app.get("/api/me")
def current_user():
    raise HTTPException(status_code=404, detail="no profile yet")


@app.get("/api/quote")
def quote():
    # An outside service: unreachable from the check, whose sandbox has no network.
    with urllib.request.urlopen("http://quotes.invalid/latest", timeout=5) as answer:
        return {"quote": answer.read().decode()}
