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
            bookedroom_list = database.query(BookedRoom).filter(BookedRoom.room_id == room_item.id).all()
            item_dict['bookedroom'] = []
            for bookedroom_obj in bookedroom_list:
                bookedroom_dict = bookedroom_obj.__dict__.copy()
                bookedroom_dict.pop('_sa_instance_state', None)
                item_dict['bookedroom'].append(bookedroom_dict)

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
        bookedroom_ids = database.query(BookedRoom.id).filter(BookedRoom.room_id == room_item.id).all()
        item_data = {
            "room": room_item,
            "bookedroom_ids": [x[0] for x in bookedroom_ids]        }
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
    standardPrice: float = None,
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
    if standardPrice is not None:
        query = query.filter(Room.standardPrice == standardPrice)

    results = query.all()
    return results


@router.get("/room/{room_id}/", response_model=None, tags=["Room"])
async def get_room(room_id: int, database: Session = Depends(get_db)) -> Room:
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    bookedroom_ids = database.query(BookedRoom.id).filter(BookedRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookedroom_ids": [x[0] for x in bookedroom_ids]}
    return response_data



@router.post("/room/", response_model=None, tags=["Room"])
async def create_room(room_data: RoomCreate, database: Session = Depends(get_db)) -> Room:


    db_room = Room(
        description=room_data.description,        roomNumber=room_data.roomNumber,        standardPrice=room_data.standardPrice,        capacity=room_data.capacity        )

    database.add(db_room)
    database.flush()
    database.refresh(db_room)

    if room_data.bookedroom:
        # Validate that all BookedRoom IDs exist
        for bookedroom_id in room_data.bookedroom:
            db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == bookedroom_id).first()
            if not db_bookedroom:
                raise HTTPException(status_code=400, detail=f"BookedRoom with id {bookedroom_id} not found")

        # Update the related entities with the new foreign key
        database.query(BookedRoom).filter(BookedRoom.id.in_(room_data.bookedroom)).update(
            {BookedRoom.room_id: db_room.id}, synchronize_session=False
        )
        database.flush()



    bookedroom_ids = database.query(BookedRoom.id).filter(BookedRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookedroom_ids": [x[0] for x in bookedroom_ids]    }
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
                description=item_data.description,                roomNumber=item_data.roomNumber,                standardPrice=item_data.standardPrice,                capacity=item_data.capacity            )
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

    setattr(db_room, 'description', room_data.description)
    setattr(db_room, 'roomNumber', room_data.roomNumber)
    setattr(db_room, 'standardPrice', room_data.standardPrice)
    setattr(db_room, 'capacity', room_data.capacity)
    if room_data.bookedroom is not None:
        requested_bookedroom = set(room_data.bookedroom)
        current_bookedroom = {
            getattr(item, 'id')
            for item in database.query(BookedRoom).filter(BookedRoom.room_id == db_room.id).all()
        }
        bookedroom_to_detach = current_bookedroom - requested_bookedroom
        if bookedroom_to_detach:
            raise HTTPException(status_code=409, detail="BookedRoom " + ", ".join(str(item) for item in sorted(bookedroom_to_detach)) + " requires a room: reassign it instead of removing it")
        bookedroom_to_attach = requested_bookedroom - current_bookedroom
        for bookedroom_id in bookedroom_to_attach:
            db_bookedroom = database.query(BookedRoom).filter(BookedRoom.id == bookedroom_id).first()
            if not db_bookedroom:
                raise HTTPException(status_code=400, detail="BookedRoom with id " + str(bookedroom_id) + " not found")
        if bookedroom_to_attach:
            database.query(BookedRoom).filter(BookedRoom.id.in_(bookedroom_to_attach)).update(
                {BookedRoom.room_id: db_room.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_room)

    bookedroom_ids = database.query(BookedRoom.id).filter(BookedRoom.room_id == db_room.id).all()
    response_data = {
        "room": db_room,
        "bookedroom_ids": [x[0] for x in bookedroom_ids]    }
    return response_data


@router.delete("/room/{room_id}/", response_model=None, tags=["Room"])
async def delete_room(room_id: int, database: Session = Depends(get_db)):
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    related_bookedroom = db_room.bookedroom
    for related in (list(related_bookedroom) if isinstance(related_bookedroom, list) else [item for item in [related_bookedroom] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Room " + str(room_id) + ": BookedRoom " + str(getattr(related, 'id')) + " requires at least 1 room")
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


@router.get("/room/{room_id}/bookedroom/", response_model=None, tags=["Room Relationships"])
async def get_bookedroom_of_room(room_id: int, database: Session = Depends(get_db)):
    """Get all BookedRoom entities related to this Room through bookedroom"""
    db_room = database.query(Room).filter(Room.id == room_id).first()
    if db_room is None:
        raise HTTPException(status_code=404, detail="Room not found")

    bookedroom_list = database.query(BookedRoom).filter(BookedRoom.room_id == room_id).all()

    return {
        "room_id": room_id,
        "bookedroom_count": len(bookedroom_list),
        "bookedroom": bookedroom_list
    }


