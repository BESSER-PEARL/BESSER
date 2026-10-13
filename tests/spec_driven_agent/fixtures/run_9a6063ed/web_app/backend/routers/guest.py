from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/guest/", response_model=None, tags=["Guest"])
def get_all_guest(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Guest)
        guest_list = query.all()

        # Serialize with relationships included
        result = []
        for guest_item in guest_list:
            item_dict = guest_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)

            # Add many-to-many and one-to-many relationship objects (full details)
            booking_list = database.query(Booking).join(guests, Booking.id == guests.c.guests).filter(guests.c.guest == guest_item.id).all()
            item_dict['guests'] = []
            for booking_obj in booking_list:
                booking_dict = booking_obj.__dict__.copy()
                booking_dict.pop('_sa_instance_state', None)
                item_dict['guests'].append(booking_dict)
            booking_list = database.query(Booking).filter(Booking.contact_id == guest_item.id).all()
            item_dict['booking'] = []
            for booking_obj in booking_list:
                booking_dict = booking_obj.__dict__.copy()
                booking_dict.pop('_sa_instance_state', None)
                item_dict['booking'].append(booking_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Guest).all()


@router.get("/guest/count/", response_model=None, tags=["Guest"])
def get_count_guest(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Guest entities"""
    count = database.query(Guest).count()
    return {"count": count}


@router.get("/guest/paginated/", response_model=None, tags=["Guest"])
def get_paginated_guest(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Guest entities"""
    total = database.query(Guest).count()
    guest_list = database.query(Guest).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": guest_list
        }

    result = []
    for guest_item in guest_list:
        booking_ids = database.query(guests.c.guests).filter(guests.c.guest == guest_item.id).all()
        booking_ids = database.query(Booking.id).filter(Booking.contact_id == guest_item.id).all()
        item_data = {
            "guest": guest_item,
            "booking_ids": [x[0] for x in booking_ids],
            "booking_ids": [x[0] for x in booking_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }
@router.get("/guest/search/", response_model=None, tags=["Guest"])
def search_guest(
    email: str = None,
    familyName: str = None,
    firstName: str = None,
    id: int = None,
    phone: str = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Guest entities by attributes"""
    query = database.query(Guest)

    if email is not None:
        query = query.filter(Guest.email.ilike(f"%{email}%"))
    if familyName is not None:
        query = query.filter(Guest.familyName.ilike(f"%{familyName}%"))
    if firstName is not None:
        query = query.filter(Guest.firstName.ilike(f"%{firstName}%"))
    if id is not None:
        query = query.filter(Guest.id == id)
    if phone is not None:
        query = query.filter(Guest.phone.ilike(f"%{phone}%"))

    results = query.all()
    return results


@router.get("/guest/{guest_id}/", response_model=None, tags=["Guest"])
async def get_guest(guest_id: int, database: Session = Depends(get_db)) -> Guest:
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    booking_ids = database.query(guests.c.guests).filter(guests.c.guest == db_guest.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_guest.id).all()
    response_data = {
        "guest": db_guest,
        "booking_ids": [x[0] for x in booking_ids],
        "booking_ids": [x[0] for x in booking_ids]}
    return response_data



@router.post("/guest/", response_model=None, tags=["Guest"])
async def create_guest(guest_data: GuestCreate, database: Session = Depends(get_db)) -> Guest:

    if guest_data.guests:
        for id in guest_data.guests:
            # Entity already validated before creation
            db_booking = database.query(Booking).filter(Booking.id == id).first()
            if not db_booking:
                raise HTTPException(status_code=404, detail=f"Booking with ID {id} not found")

    db_guest = Guest(
        phone=guest_data.phone,        firstName=guest_data.firstName,        email=guest_data.email,        familyName=guest_data.familyName        )

    database.add(db_guest)
    database.flush()
    database.refresh(db_guest)

    if guest_data.booking:
        # Validate that all Booking IDs exist
        for booking_id in guest_data.booking:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail=f"Booking with id {booking_id} not found")

        # Update the related entities with the new foreign key
        database.query(Booking).filter(Booking.id.in_(guest_data.booking)).update(
            {Booking.contact_id: db_guest.id}, synchronize_session=False
        )
        database.flush()

    if guest_data.guests:
        for id in guest_data.guests:
            # Entity already validated before creation
            db_booking = database.query(Booking).filter(Booking.id == id).first()
            # Create the association
            association = guests.insert().values(guest=db_guest.id, guests=db_booking.id)
            database.execute(association)
            database.flush()


    booking_ids = database.query(guests.c.guests).filter(guests.c.guest == db_guest.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_guest.id).all()
    response_data = {
        "guest": db_guest,
        "booking_ids": [x[0] for x in booking_ids],
        "booking_ids": [x[0] for x in booking_ids]    }
    database.commit()
    database.refresh(db_guest)
    return response_data


@router.post("/guest/bulk/", response_model=None, tags=["Guest"])
async def bulk_create_guest(items: list[GuestCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Guest entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item

            db_guest = Guest(
                phone=item_data.phone,                firstName=item_data.firstName,                email=item_data.email,                familyName=item_data.familyName            )
            database.add(db_guest)
            database.flush()  # Get ID without committing
            created_items.append(db_guest.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Guest entities"
    }


@router.delete("/guest/bulk/", response_model=None, tags=["Guest"])
async def bulk_delete_guest(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Guest entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_guest = database.query(Guest).filter(Guest.id == item_id).first()
        if db_guest:
            database.delete(db_guest)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Guest entities"
    }

@router.put("/guest/{guest_id}/", response_model=None, tags=["Guest"])
async def update_guest(guest_id: int, guest_data: GuestCreate, database: Session = Depends(get_db)) -> Guest:
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    setattr(db_guest, 'phone', guest_data.phone)
    setattr(db_guest, 'firstName', guest_data.firstName)
    setattr(db_guest, 'email', guest_data.email)
    setattr(db_guest, 'familyName', guest_data.familyName)
    if guest_data.booking is not None:
        requested_booking = set(guest_data.booking)
        current_booking = {
            getattr(item, 'id')
            for item in database.query(Booking).filter(Booking.contact_id == db_guest.id).all()
        }
        booking_to_detach = current_booking - requested_booking
        if booking_to_detach:
            raise HTTPException(status_code=409, detail="Booking " + ", ".join(str(item) for item in sorted(booking_to_detach)) + " requires a contact: reassign it instead of removing it")
        booking_to_attach = requested_booking - current_booking
        for booking_id in booking_to_attach:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail="Booking with id " + str(booking_id) + " not found")
        if booking_to_attach:
            database.query(Booking).filter(Booking.id.in_(booking_to_attach)).update(
                {Booking.contact_id: db_guest.id}, synchronize_session=False
            )
    if guest_data.guests is not None:
        existing_booking_ids = [assoc.guests for assoc in database.execute(
            guests.select().where(guests.c.guest == db_guest.id))]

        bookings_to_remove = set(existing_booking_ids) - set(guest_data.guests)
        for booking_id in bookings_to_remove:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if db_booking is not None and len(db_booking.guest) - 1 < 1:
                raise HTTPException(status_code=409, detail="Booking " + str(booking_id) + " requires at least 1 guest")
            association = guests.delete().where(
                (guests.c.guest == db_guest.id) & (guests.c.guests == booking_id))
            database.execute(association)

        new_booking_ids = set(guest_data.guests) - set(existing_booking_ids)
        for booking_id in new_booking_ids:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if db_booking is None:
                raise HTTPException(status_code=404, detail="Booking with ID " + str(booking_id) + " not found")
            association = guests.insert().values(guests=db_booking.id, guest=db_guest.id)
            database.execute(association)
    database.commit()
    database.refresh(db_guest)

    booking_ids = database.query(guests.c.guests).filter(guests.c.guest == db_guest.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_guest.id).all()
    response_data = {
        "guest": db_guest,
        "booking_ids": [x[0] for x in booking_ids],
        "booking_ids": [x[0] for x in booking_ids]    }
    return response_data


@router.delete("/guest/{guest_id}/", response_model=None, tags=["Guest"])
async def delete_guest(guest_id: int, database: Session = Depends(get_db)):
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")
    related_guests = db_guest.guests
    for related in (list(related_guests) if isinstance(related_guests, list) else [item for item in [related_guests] if item is not None]):
        remaining = len(related.guest) - 1
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Guest " + str(guest_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 guest")
    related_booking = db_guest.booking
    for related in (list(related_booking) if isinstance(related_booking, list) else [item for item in [related_booking] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Guest " + str(guest_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 contact")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_guest = {
        attr.key: getattr(db_guest, attr.key)
        for attr in db_guest.__mapper__.column_attrs
    }
    database.delete(db_guest)
    database.commit()
    return deleted_guest

@router.post("/guest/{guest_id}/guests/{booking_id}/", response_model=None, tags=["Guest Relationships"])
async def add_guests_to_guest(guest_id: int, booking_id: int, database: Session = Depends(get_db)):
    """Add a Booking to this Guest's guests relationship"""
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    # Check if relationship already exists
    existing = database.query(guests).filter(
        (guests.c.guest == guest_id) &
        (guests.c.guests == booking_id)
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Relationship already exists")

    # Create the association
    association = guests.insert().values(guest=guest_id, guests=booking_id)
    database.execute(association)
    database.commit()

    return {"message": "Booking added to guests successfully"}


@router.delete("/guest/{guest_id}/guests/{booking_id}/", response_model=None, tags=["Guest Relationships"])
async def remove_guests_from_guest(guest_id: int, booking_id: int, database: Session = Depends(get_db)):
    """Remove a Booking from this Guest's guests relationship"""
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    # Check if relationship exists
    existing = database.query(guests).filter(
        (guests.c.guest == guest_id) &
        (guests.c.guests == booking_id)
    ).first()

    if not existing:
        raise HTTPException(status_code=404, detail="Relationship not found")

    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is not None and len(db_booking.guest) - 1 < 1:
        raise HTTPException(status_code=409, detail="Booking " + str(booking_id) + " requires at least 1 guest")

    # Delete the association
    association = guests.delete().where(
        (guests.c.guest == guest_id) &
        (guests.c.guests == booking_id)
    )
    database.execute(association)
    database.commit()

    return {"message": "Booking removed from guests successfully"}


@router.get("/guest/{guest_id}/guests/", response_model=None, tags=["Guest Relationships"])
async def get_guests_of_guest(guest_id: int, database: Session = Depends(get_db)):
    """Get all Booking entities related to this Guest through guests"""
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    booking_ids = database.query(guests.c.guests).filter(guests.c.guest == guest_id).all()
    booking_list = database.query(Booking).filter(Booking.id.in_([id[0] for id in booking_ids])).all()

    return {
        "guest_id": guest_id,
        "guests_count": len(booking_list),
        "guests": booking_list
    }


@router.get("/guest/{guest_id}/booking/", response_model=None, tags=["Guest Relationships"])
async def get_booking_of_guest(guest_id: int, database: Session = Depends(get_db)):
    """Get all Booking entities related to this Guest through booking"""
    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    booking_list = database.query(Booking).filter(Booking.contact_id == guest_id).all()

    return {
        "guest_id": guest_id,
        "booking_count": len(booking_list),
        "booking": booking_list
    }


