# Excerpt of gpt-5.6-terra-61nry6wd: the single create was updated for the
# schema change, the generated bulk create still reads the removed fields.
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic_classes import *
from sql_alchemy import *
from database import get_db

router = APIRouter()


@router.post("/bill/", response_model=None, tags=["Bill"])
async def create_bill(bill_data: BillCreate, database: Session = Depends(get_db)) -> Bill:
    db_booking = database.query(Booking).filter(Booking.id == bill_data.booking).first()
    db_bill = Bill(
        totalAmountDue=0.0,
        settled=False,
        billNumber=bill_data.billNumber,
        issuedDate=date.today(),
        booking_id=db_booking.id,
    )
    database.add(db_bill)
    database.commit()
    return db_bill


@router.post("/bill/bulk/", response_model=None, tags=["Bill"])
async def bulk_create_bill(items: list[BillCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Bill entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.booking:
                raise ValueError("Booking ID is required")

            db_bill = Bill(
                totalAmountDue=item_data.totalAmountDue,                settled=item_data.settled,                billNumber=item_data.billNumber,                issuedDate=item_data.issuedDate,                booking_id=item_data.booking            )
            database.add(db_bill)
            database.flush()  # Get ID without committing
            created_items.append(db_bill.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {"created_count": len(created_items), "created_ids": created_items}
