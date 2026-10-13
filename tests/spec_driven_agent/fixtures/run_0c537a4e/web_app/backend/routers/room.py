from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/room/", response_model=None, tags=["Room"])
def get_all_room(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Room)
        room_list = query.all()

        # Serialize with relationships included
        result = []
        for room_item in room_list:
            item_dict = room_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)

            # Add many-to-many and one-to-many relationship objects (full details)
            bookingroom_list = database.query(BookingRoom).filter(BookingRoom.room_id == room_item.id).all()
            item_dict['bookingroom'] = []
            for bookingroom_obj in bookingroom_list:
                bookingroom_dict = bookingroom_obj.__dict__.copy()
                bookingroom_dict.pop('_sa_instance_state', None)
                item_dict['bookingroom'].append(bookingroom_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Room).all()


@router.get("/room/count/", response_model=None, tags=["Room"])
def get_count_room(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Room entities"""
    count = database.query(Room).count()
    return {"count": count}


@router.get("/room/paginated/", response_model=None, tags=["Room"])
def get_paginated_room(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Room entities"""
    total = database.query(Room).count()
    room_list = database.query(Room).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": room_list
        }

    result = []
    for room_item in room_list:
        bookingroom_ids = database.query(BookingRoom.id).filter(BookingRoom.room_id == room_item.id).all()
        item_data = {
            "room": room_item,
            "bookingroom_ids": [x[0] for x in bookingroom_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }
@router.get("/room/search/", response_model=None, tags=["Room"])
def search_room(
    capacity: int = None,
    description: str = None,
    id: int = None,
    roomNumber: str = None,
    standardNightlyPrice: float = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Room entities by attributes"""
    query = database.query(Room)

    if capacity is not None:
        query = query.filter(Room.capacity == capacity)
    if description is not None:
        query = query.filter(Room.description.ilike(f"%{description}%"))
    if id is not None:
        query = query.filter(Room.id == id)
    if roomNumber is not None:
        query = query.filter(Room.roomNumber.ilike(f"%{roomNumber}%"))
    if standardNightlyPrice is not None:
        query = query.filter(Room.standardNightlyPrice == standardNightlyPrice)

    results = query.all()
    return results


@router.get("/room/{room_id}/", response_model=None, tags=["Room"])
async def get_room(room_id: int, database: Session = Depends(get_db)) -> Room:
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    bookingroom_ids = database.query(BookingRoom.id).filter(BookingRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookingroom_ids": [x[0] for x in bookingroom_ids]}
    return response_data



@router.post("/room/", response_model=None, tags=["Room"])
async def create_room(room_data: RoomCreate, database: Session = Depends(get_db)) -> Room:


    db_room = Room(
        roomNumber=room_data.roomNumber,        capacity=room_data.capacity,        standardNightlyPrice=room_data.standardNightlyPrice,        description=room_data.description        )

    database.add(db_room)
    database.flush()
    database.refresh(db_room)

    if room_data.bookingroom:
        # Validate that all BookingRoom IDs exist
        for bookingroom_id in room_data.bookingroom:
            db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == bookingroom_id).first()
            if not db_bookingroom:
                raise HTTPException(status_code=400, detail=f"BookingRoom with id {bookingroom_id} not found")

        # Update the related entities with the new foreign key
        database.query(BookingRoom).filter(BookingRoom.id.in_(room_data.bookingroom)).update(
            {BookingRoom.room_id: db_room.id}, synchronize_session=False
        )
        database.flush()



    bookingroom_ids = database.query(BookingRoom.id).filter(BookingRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookingroom_ids": [x[0] for x in bookingroom_ids]    }
    database.commit()
    database.refresh(db_room)
    return response_data


@router.post("/room/bulk/", response_model=None, tags=["Room"])
async def bulk_create_room(items: list[RoomCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Room entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item

            db_room = Room(
                roomNumber=item_data.roomNumber,                capacity=item_data.capacity,                standardNightlyPrice=item_data.standardNightlyPrice,                description=item_data.description            )
            database.add(db_room)
            database.flush()  # Get ID without committing
            created_items.append(db_room.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Room entities"
    }


@router.delete("/room/bulk/", response_model=None, tags=["Room"])
async def bulk_delete_room(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Room entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_room = database.query(Room).filter(Room.id == item_id).first()
        if db_room:
            database.delete(db_room)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Room entities"
    }

@router.put("/room/{room_id}/", response_model=None, tags=["Room"])
async def update_room(room_id: int, room_data: RoomCreate, database: Session = Depends(get_db)) -> Room:
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    setattr(db_room, 'roomNumber', room_data.roomNumber)
    setattr(db_room, 'capacity', room_data.capacity)
    setattr(db_room, 'standardNightlyPrice', room_data.standardNightlyPrice)
    setattr(db_room, 'description', room_data.description)
    if room_data.bookingroom is not None:
        requested_bookingroom = set(room_data.bookingroom)
        current_bookingroom = {
            getattr(item, 'id')
            for item in database.query(BookingRoom).filter(BookingRoom.room_id == db_room.id).all()
        }
        bookingroom_to_detach = current_bookingroom - requested_bookingroom
        if bookingroom_to_detach:
            raise HTTPException(status_code=409, detail="BookingRoom " + ", ".join(str(item) for item in sorted(bookingroom_to_detach)) + " requires a room: reassign it instead of removing it")
        bookingroom_to_attach = requested_bookingroom - current_bookingroom
        for bookingroom_id in bookingroom_to_attach:
            db_bookingroom = database.query(BookingRoom).filter(BookingRoom.id == bookingroom_id).first()
            if not db_bookingroom:
                raise HTTPException(status_code=400, detail="BookingRoom with id " + str(bookingroom_id) + " not found")
        if bookingroom_to_attach:
            database.query(BookingRoom).filter(BookingRoom.id.in_(bookingroom_to_attach)).update(
                {BookingRoom.room_id: db_room.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_room)

    bookingroom_ids = database.query(BookingRoom.id).filter(BookingRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookingroom_ids": [x[0] for x in bookingroom_ids]    }
    return response_data


@router.delete("/room/{room_id}/", response_model=None, tags=["Room"])
async def delete_room(room_id: int, database: Session = Depends(get_db)):
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    related_bookingroom = db_room.bookingroom
    for related in (list(related_bookingroom) if isinstance(related_bookingroom, list) else [item for item in [related_bookingroom] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Room " + str(room_id) + ": BookingRoom " + str(getattr(related, 'id')) + " requires at least 1 room")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_room = {
        attr.key: getattr(db_room, attr.key)
        for attr in db_room.__mapper__.column_attrs
    }
    database.delete(db_room)
    database.commit()
    return deleted_room


@router.get("/room/{room_id}/bookingroom/", response_model=None, tags=["Room Relationships"])
async def get_bookingroom_of_room(room_id: int, database: Session = Depends(get_db)):
    """Get all BookingRoom entities related to this Room through bookingroom"""
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    bookingroom_list = database.query(BookingRoom).filter(BookingRoom.room_id == room_id).all()

    return {
        "room_id": room_id,
        "bookingroom_count": len(bookingroom_list),
        "bookingroom": bookingroom_list
    }


