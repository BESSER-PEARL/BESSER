"""Stand-in for a generated backend: one route per outcome a screen can meet."""

import asyncio

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
