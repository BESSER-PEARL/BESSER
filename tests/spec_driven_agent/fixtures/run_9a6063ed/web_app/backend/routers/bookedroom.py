from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/bookedroom/", response_model=None, tags=["BookedRoom"])
def get_all_bookedroom(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(BookedRoom)
        query = query.options(joinedload(BookedRoom.booking))
        query = query.options(joinedload(BookedRoom.booking_1))
        query = query.options(joinedload(BookedRoom.room))
        bookedroom_list = query.all()

        # Serialize with relationships included
        result = []
        for bookedroom_item in bookedroom_list:
            item_dict = bookedroom_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if bookedroom_item.booking:
                related_obj = bookedroom_item.booking
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['booking'] = related_dict
            else:
                item_dict['booking'] = None
            if bookedroom_item.booking_1:
                related_obj = bookedroom_item.booking_1
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['booking_1'] = related_dict
            else:
                item_dict['booking_1'] = None
            if bookedroom_item.room:
                related_obj = bookedroom_item.room
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['room'] = related_dict
            else:
                item_dict['room'] = None


            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(BookedRoom).all()


@router.get("/bookedroom/count/", response_model=None, tags=["BookedRoom"])
def get_count_bookedroom(database: Session = Depends(get_db)) -> dict:
    """Get the total count of BookedRoom entities"""
    count = database.query(BookedRoom).count()
    return {"count": count}


@router.get("/bookedroom/paginated/", response_model=None, tags=["BookedRoom"])
def get_paginated_bookedroom(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of BookedRoom entities"""
    total = database.query(BookedRoom).count()
    bookedroom_list = database.query(BookedRoom).offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": bookedroom_list
    }
@router.get("/bookedroom/search/", response_model=None, tags=["BookedRoom"])
def search_bookedroom(
    agreedPrice: float = None,
    id: int = None,
    database: Session = Depends(get_db)
) -> list:
    """Search BookedRoom entities by attributes"""
    query = database.query(BookedRoom)

    if agreedPrice is not None:
        query = query.filter(BookedRoom.agreedPrice == agreedPrice)
    if id is not None:
        query = query.filter(BookedRoom.id == id)

    results = query.all()
    return results


@router.get("/bookedroom/{bookedroom_id}/", response_model=None, tags=["BookedRoom"])
async def get_bookedroom(bookedroom_id: int, database: Session = Depends(get_db)) -> BookedRoom:
    db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == bookedroom_id).first()
    if db_bookedroom is None:
        raise HTTPException(status_code=404, detail="BookedRoom not found")

    response_data = {
        "bookedroom": db_bookedroom,
}
    return response_data



@router.post("/bookedroom/", response_model=None, tags=["BookedRoom"])
async def create_bookedroom(bookedroom_data: BookedRoomCreate, database: Session = Depends(get_db)) -> BookedRoom:

    if bookedroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bookedroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
    else:
        raise HTTPException(status_code=400, detail="Booking ID is required")
    if bookedroom_data.booking_1 is not None:
        db_booking_1 = database.query(Booking).filter(Booking.id == bookedroom_data.booking_1).first()
        if not db_booking_1:
            raise HTTPException(status_code=400, detail="Booking not found")
    else:
        raise HTTPException(status_code=400, detail="Booking ID is required")
    if bookedroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == bookedroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
    else:
        raise HTTPException(status_code=400, detail="Room ID is required")

    db_bookedroom = BookedRoom(
        agreedPrice=bookedroom_data.agreedPrice,        booking_id=bookedroom_data.booking,        booking_1_id=bookedroom_data.booking_1,        room_id=bookedroom_data.room        )

    database.add(db_bookedroom)
    database.flush()
    database.refresh(db_bookedroom)




    database.commit()
    database.refresh(db_bookedroom)
    return db_bookedroom


@router.post("/bookedroom/bulk/", response_model=None, tags=["BookedRoom"])
async def bulk_create_bookedroom(items: list[BookedRoomCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple BookedRoom entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.booking:
                raise ValueError("Booking ID is required")
            if not item_data.booking_1:
                raise ValueError("Booking ID is required")
            if not item_data.room:
                raise ValueError("Room ID is required")

            db_bookedroom = BookedRoom(
                agreedPrice=item_data.agreedPrice,                booking_id=item_data.booking,                booking_1_id=item_data.booking_1,                room_id=item_data.room,
                extraCharges=item_data.extraCharges if hasattr(item_data, 'extraCharges') else 0.0            )
            database.add(db_bookedroom)
            database.flush()  # Get ID without committing
            created_items.append(db_bookedroom.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} BookedRoom entities"
    }


@router.delete("/bookedroom/bulk/", response_model=None, tags=["BookedRoom"])
async def bulk_delete_bookedroom(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple BookedRoom entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == item_id).first()
        if db_bookedroom:
            database.delete(db_bookedroom)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} BookedRoom entities"
    }

@router.put("/bookedroom/{bookedroom_id}/", response_model=None, tags=["BookedRoom"])
async def update_bookedroom(bookedroom_id: int, bookedroom_data: BookedRoomCreate, database: Session = Depends(get_db)) -> BookedRoom:
    db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == bookedroom_id).first()
    if db_bookedroom is None:
        raise HTTPException(status_code=404, detail="BookedRoom not found")

    setattr(db_bookedroom, 'agreedPrice', bookedroom_data.agreedPrice)
    if bookedroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bookedroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
        setattr(db_bookedroom, 'booking_id', bookedroom_data.booking)
    if bookedroom_data.booking_1 is not None:
        db_booking_1 = database.query(Booking).filter(Booking.id == bookedroom_data.booking_1).first()
        if not db_booking_1:
            raise HTTPException(status_code=400, detail="Booking not found")
        setattr(db_bookedroom, 'booking_1_id', bookedroom_data.booking_1)
    if bookedroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == bookedroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
        setattr(db_bookedroom, 'room_id', bookedroom_data.room)
    if hasattr(bookedroom_data, 'extraCharges'):
        db_bookedroom.extraCharges = bookedroom_data.extraCharges
    database.commit()
    database.refresh(db_bookedroom)

    return db_bookedroom


@router.delete("/bookedroom/{bookedroom_id}/", response_model=None, tags=["BookedRoom"])
async def delete_bookedroom(bookedroom_id: int, database: Session = Depends(get_db)):
    db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == bookedroom_id).first()
    if db_bookedroom is None:
        raise HTTPException(status_code=404, detail="BookedRoom not found")
    related_booking = db_bookedroom.booking
    for related in (list(related_booking) if isinstance(related_booking, list) else [item for item in [related_booking] if item is not None]):
        remaining = len(related.bookedRooms) - 1
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete BookedRoom " + str(bookedroom_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 bookedRooms")
    related_booking_1 = db_bookedroom.booking_1
    for related in (list(related_booking_1) if isinstance(related_booking_1, list) else [item for item in [related_booking_1] if item is not None]):
        remaining = len(related.bookedroom) - 1
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete BookedRoom " + str(bookedroom_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 bookedroom")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_bookedroom = {
        attr.key: getattr(db_bookedroom, attr.key)
        for attr in db_bookedroom.__mapper__.column_attrs
    }
    database.delete(db_bookedroom)
    database.commit()
    return deleted_bookedroom



