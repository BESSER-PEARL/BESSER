from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/bill/", response_model=None, tags=["Bill"])
def get_all_bill(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Bill)
        query = query.options(joinedload(Bill.booking))
        bill_list = query.all()

        # Serialize with relationships included
        result = []
        for bill_item in bill_list:
            item_dict = bill_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if bill_item.booking:
                related_obj = bill_item.booking
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['booking'] = related_dict
            else:
                item_dict['booking'] = None


            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Bill).all()


@router.get("/bill/count/", response_model=None, tags=["Bill"])
def get_count_bill(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Bill entities"""
    count = database.query(Bill).count()
    return {"count": count}


@router.get("/bill/paginated/", response_model=None, tags=["Bill"])
def get_paginated_bill(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Bill entities"""
    total = database.query(Bill).count()
    bill_list = database.query(Bill).offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": bill_list
    }
@router.get("/bill/search/", response_model=None, tags=["Bill"])
def search_bill(
    id: int = None,
    settled: bool = None,
    totalAmountDue: float = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Bill entities by attributes"""
    query = database.query(Bill)

    if id is not None:
        query = query.filter(Bill.id == id)
    if settled is not None:
        query = query.filter(Bill.settled == settled)
    if totalAmountDue is not None:
        query = query.filter(Bill.totalAmountDue == totalAmountDue)

    results = query.all()
    return results


@router.get("/bill/{bill_id}/", response_model=None, tags=["Bill"])
async def get_bill(bill_id: int, database: Session = Depends(get_db)) -> Bill:
    db_bill = database.query(Bill).filter(Bill.id == bill_id).first()
    if db_bill is None:
        raise HTTPException(status_code=404, detail="Bill not found")

    response_data = {
        "bill": db_bill,
}
    return response_data



@router.post("/bill/", response_model=None, tags=["Bill"])
async def create_bill(bill_data: BillCreate, database: Session = Depends(get_db)) -> Bill:

    if bill_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bill_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
    else:
        raise HTTPException(status_code=400, detail="Booking ID is required")

    db_bill = Bill(
        totalAmountDue=bill_data.totalAmountDue,        settled=bill_data.settled,        issuedDate=bill_data.issuedDate,        booking_id=bill_data.booking        )

    database.add(db_bill)
    database.flush()
    database.refresh(db_bill)




    database.commit()
    database.refresh(db_bill)
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
                totalAmountDue=item_data.totalAmountDue,                settled=item_data.settled,                issuedDate=item_data.issuedDate,                booking_id=item_data.booking            )
            database.add(db_bill)
            database.flush()  # Get ID without committing
            created_items.append(db_bill.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Bill entities"
    }


@router.delete("/bill/bulk/", response_model=None, tags=["Bill"])
async def bulk_delete_bill(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Bill entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_bill = database.query(Bill).filter(Bill.id == item_id).first()
        if db_bill:
            database.delete(db_bill)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Bill entities"
    }

@router.put("/bill/{bill_id}/", response_model=None, tags=["Bill"])
async def update_bill(bill_id: int, bill_data: BillCreate, database: Session = Depends(get_db)) -> Bill:
    db_bill = database.query(Bill).filter(Bill.id == bill_id).first()
    if db_bill is None:
        raise HTTPException(status_code=404, detail="Bill not found")

    setattr(db_bill, 'totalAmountDue', bill_data.totalAmountDue)
    setattr(db_bill, 'settled', bill_data.settled)
    setattr(db_bill, 'issuedDate', bill_data.issuedDate)
    if bill_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bill_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
        setattr(db_bill, 'booking_id', bill_data.booking)
    database.commit()
    database.refresh(db_bill)

    return db_bill


@router.delete("/bill/{bill_id}/", response_model=None, tags=["Bill"])
async def delete_bill(bill_id: int, database: Session = Depends(get_db)):
    db_bill = database.query(Bill).filter(Bill.id == bill_id).first()
    if db_bill is None:
        raise HTTPException(status_code=404, detail="Bill not found")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_bill = {
        attr.key: getattr(db_bill, attr.key)
        for attr in db_bill.__mapper__.column_attrs
    }
    database.delete(db_bill)
    database.commit()
    return deleted_bill



