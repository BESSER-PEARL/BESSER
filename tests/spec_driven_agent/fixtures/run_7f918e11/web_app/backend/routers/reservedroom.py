from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/reservedroom/", response_model=None, tags=["ReservedRoom"])
def get_all_reservedroom(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(ReservedRoom)
        query = query.options(joinedload(ReservedRoom.booking))
        query = query.options(joinedload(ReservedRoom.room))
        reservedroom_list = query.all()

        # Serialize with relationships included
        result = []
        for reservedroom_item in reservedroom_list:
            item_dict = reservedroom_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if reservedroom_item.booking:
                related_obj = reservedroom_item.booking
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['booking'] = related_dict
            else:
                item_dict['booking'] = None
            if reservedroom_item.room:
                related_obj = reservedroom_item.room
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['room'] = related_dict
            else:
                item_dict['room'] = None


            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(ReservedRoom).all()


@router.get("/reservedroom/count/", response_model=None, tags=["ReservedRoom"])
def get_count_reservedroom(database: Session = Depends(get_db)) -> dict:
    """Get the total count of ReservedRoom entities"""
    count = database.query(ReservedRoom).count()
    return {"count": count}


@router.get("/reservedroom/paginated/", response_model=None, tags=["ReservedRoom"])
def get_paginated_reservedroom(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of ReservedRoom entities"""
    total = database.query(ReservedRoom).count()
    reservedroom_list = database.query(ReservedRoom).offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": reservedroom_list
    }
@router.get("/reservedroom/search/", response_model=None, tags=["ReservedRoom"])
def search_reservedroom(
    agreedPrice: float = None,
    id: int = None,
    database: Session = Depends(get_db)
) -> list:
    """Search ReservedRoom entities by attributes"""
    query = database.query(ReservedRoom)

    if agreedPrice is not None:
        query = query.filter(ReservedRoom.agreedPrice == agreedPrice)
    if id is not None:
        query = query.filter(ReservedRoom.id == id)

    results = query.all()
    return results


@router.get("/reservedroom/{reservedroom_id}/", response_model=None, tags=["ReservedRoom"])
async def get_reservedroom(reservedroom_id: int, database: Session = Depends(get_db)) -> ReservedRoom:
    db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == reservedroom_id).first()
    if db_reservedroom is None:
        raise HTTPException(status_code=404, detail="ReservedRoom not found")

    response_data = {
        "reservedroom": db_reservedroom,
}
    return response_data



@router.post("/reservedroom/", response_model=None, tags=["ReservedRoom"])
async def create_reservedroom(reservedroom_data: ReservedRoomCreate, database: Session = Depends(get_db)) -> ReservedRoom:

    if reservedroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == reservedroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
    else:
        raise HTTPException(status_code=400, detail="Booking ID is required")
    if reservedroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == reservedroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
    else:
        raise HTTPException(status_code=400, detail="Room ID is required")

    db_reservedroom = ReservedRoom(
        agreedPrice=reservedroom_data.agreedPrice,        booking_id=reservedroom_data.booking,        room_id=reservedroom_data.room        )

    database.add(db_reservedroom)
    database.flush()
    database.refresh(db_reservedroom)




    database.commit()
    database.refresh(db_reservedroom)
    return db_reservedroom


@router.post("/reservedroom/bulk/", response_model=None, tags=["ReservedRoom"])
async def bulk_create_reservedroom(items: list[ReservedRoomCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple ReservedRoom entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.booking:
                raise ValueError("Booking ID is required")
            if not item_data.room:
                raise ValueError("Room ID is required")

            db_reservedroom = ReservedRoom(
                agreedPrice=item_data.agreedPrice,                booking_id=item_data.booking,                room_id=item_data.room            )
            database.add(db_reservedroom)
            database.flush()  # Get ID without committing
            created_items.append(db_reservedroom.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} ReservedRoom entities"
    }


@router.delete("/reservedroom/bulk/", response_model=None, tags=["ReservedRoom"])
async def bulk_delete_reservedroom(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple ReservedRoom entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == item_id).first()
        if db_reservedroom:
            database.delete(db_reservedroom)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} ReservedRoom entities"
    }

@router.put("/reservedroom/{reservedroom_id}/", response_model=None, tags=["ReservedRoom"])
async def update_reservedroom(reservedroom_id: int, reservedroom_data: ReservedRoomCreate, database: Session = Depends(get_db)) -> ReservedRoom:
    db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == reservedroom_id).first()
    if db_reservedroom is None:
        raise HTTPException(status_code=404, detail="ReservedRoom not found")

    setattr(db_reservedroom, 'agreedPrice', reservedroom_data.agreedPrice)
    if reservedroom_data.booking is not None:
        db_booking = database.query(Booking).filter(Booking.id == reservedroom_data.booking).first()
        if not db_booking:
            raise HTTPException(status_code=400, detail="Booking not found")
        setattr(db_reservedroom, 'booking_id', reservedroom_data.booking)
    if reservedroom_data.room is not None:
        db_room = database.query(Room).filter(Room.id == reservedroom_data.room).first()
        if not db_room:
            raise HTTPException(status_code=400, detail="Room not found")
        setattr(db_reservedroom, 'room_id', reservedroom_data.room)
    database.commit()
    database.refresh(db_reservedroom)

    return db_reservedroom


@router.delete("/reservedroom/{reservedroom_id}/", response_model=None, tags=["ReservedRoom"])
async def delete_reservedroom(reservedroom_id: int, database: Session = Depends(get_db)):
    db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == reservedroom_id).first()
    if db_reservedroom is None:
        raise HTTPException(status_code=404, detail="ReservedRoom not found")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_reservedroom = {
        attr.key: getattr(db_reservedroom, attr.key)
        for attr in db_reservedroom.__mapper__.column_attrs
    }
    database.delete(db_reservedroom)
    database.commit()
    return deleted_reservedroom



