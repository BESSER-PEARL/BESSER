from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/bookingroom/", response_model=None, tags=["BookingRoom"])
def get_all_bookingroom(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(BookingRoom)
        query = query.options(joinedload(BookingRoom.booking))
        query = query.options(joinedload(BookingRoom.room))
        bookingroom_list = query.all()

        # Serialize with relationships included
        result = []
        for bookingroom_item in bookingroom_list:
            item_dict = bookingroom_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if bookingroom_item.booking:
                related_obj = bookingroom_item.booking
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['booking'] = related_dict
            else:
                item_dict['booking'] = None
            if bookingroom_item.room:
                related_obj = bookingroom_item.room
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['room'] = related_dict
            else:
                item_dict['room'] = None


            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(BookingRoom).all()


@router.get("/bookingroom/count/", response_model=None, tags=["BookingRoom"])
def get_count_bookingroom(database: Session = Depends(get_db)) -> dict:
    """Get the total count of BookingRoom entities"""
    count = database.query(BookingRoom).count()
    return {"count": count}


@router.get("/bookingroom/paginated/", response_model=None, tags=["BookingRoom"])
def get_paginated_bookingroom(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of BookingRoom entities"""
    total = database.query(BookingRoom).count()
    bookingroom_list = database.query(BookingRoom).offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": bookingroom_list
    }
@router.get("/bookingroom/search/", response_model=None, tags=["BookingRoom"])
def search_bookingroom(
    agreedPrice: float = None,
    id: int = None,
    database: Session = Depends(get_db)
) -> list:
    """Search BookingRoom entities by attributes"""
    query = database.query(BookingRoom)

    if agreedPrice is not None:
        query = query.filter(BookingRoom.agreedPrice == agreedPrice)
    if id is not None:
        query = query.filter(BookingRoom.id == id)

    results = query.all()
    return results


@router.get("/bookingroom/{bookingroom_id}/", response_model=None, tags=["BookingRoom"])
async def get_bookingroom(bookingroom_id: int, database: Session = Depends(get_db)) -> BookingRoom:
    db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == bookingroom_id).first()
    if db_bookingroom is None:
        raise HTTPException(status_code=404, detail="BookingRoom not found")

    response_data = {
        "bookingroom": db_bookingroom,
}
    return response_data



@router.post("/bookingroom/", response_model=None, tags=["BookingRoom"])
async def create_bookingroom(bookingroom_data: BookingRoomCreate, database: Session = Depends(get_db)) -> BookingRoom:

    if bookingroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bookingroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
    else:
        raise HTTPException(status_code=400, detail="Booking ID is required")
    if bookingroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == bookingroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
    else:
        raise HTTPException(status_code=400, detail="Room ID is required")

    db_bookingroom = BookingRoom(
        agreedPrice=bookingroom_data.agreedPrice,
        extraCharges=bookingroom_data.extraCharges,
        booking_id=bookingroom_data.booking,
        room_id=bookingroom_data.room        )

    database.add(db_bookingroom)
    database.flush()
    database.refresh(db_bookingroom)




    database.commit()
    database.refresh(db_bookingroom)
    return db_bookingroom


@router.post("/bookingroom/bulk/", response_model=None, tags=["BookingRoom"])
async def bulk_create_bookingroom(items: list[BookingRoomCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple BookingRoom entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.booking:
                raise ValueError("Booking ID is required")
            if not item_data.room:
                raise ValueError("Room ID is required")

            db_bookingroom = BookingRoom(
                agreedPrice=item_data.agreedPrice,                booking_id=item_data.booking,                room_id=item_data.room            )
            database.add(db_bookingroom)
            database.flush()  # Get ID without committing
            created_items.append(db_bookingroom.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} BookingRoom entities"
    }


@router.delete("/bookingroom/bulk/", response_model=None, tags=["BookingRoom"])
async def bulk_delete_bookingroom(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple BookingRoom entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == item_id).first()
        if db_bookingroom:
            database.delete(db_bookingroom)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} BookingRoom entities"
    }

@router.put("/bookingroom/{bookingroom_id}/", response_model=None, tags=["BookingRoom"])
async def update_bookingroom(bookingroom_id: int, bookingroom_data: BookingRoomCreate, database: Session = Depends(get_db)) -> BookingRoom:
    db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == bookingroom_id).first()
    if db_bookingroom is None:
        raise HTTPException(status_code=404, detail="BookingRoom not found")

    setattr(db_bookingroom, 'agreedPrice', bookingroom_data.agreedPrice)
    setattr(db_bookingroom, 'extraCharges', bookingroom_data.extraCharges)
    if bookingroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == bookingroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
        setattr(db_bookingroom, 'booking_id', bookingroom_data.booking)
    if bookingroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == bookingroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
        setattr(db_bookingroom, 'room_id', bookingroom_data.room)
    database.commit()
    database.refresh(db_bookingroom)

    return db_bookingroom


@router.delete("/bookingroom/{bookingroom_id}/", response_model=None, tags=["BookingRoom"])
async def delete_bookingroom(bookingroom_id: int, database: Session = Depends(get_db)):
    db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == bookingroom_id).first()
    if db_bookingroom is None:
        raise HTTPException(status_code=404, detail="BookingRoom not found")
    related_booking = db_bookingroom.booking
    for related in (list(related_booking) if isinstance(related_booking, list) else [item for item in [related_booking] if item is not None]):
        remaining = len(related.bookingRooms) - 1
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete BookingRoom " + str(bookingroom_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 bookingRooms")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_bookingroom = {
        attr.key: getattr(db_bookingroom, attr.key)
        for attr in db_bookingroom.__mapper__.column_attrs
    }
    database.delete(db_bookingroom)
    database.commit()
    return deleted_bookingroom



