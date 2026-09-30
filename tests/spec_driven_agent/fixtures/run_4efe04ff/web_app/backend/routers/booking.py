from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/booking/", response_model=None, tags=["Booking"])
def get_all_booking(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Booking)
        query = query.options(joinedload(Booking.employee))
        query = query.options(joinedload(Booking.bill))
        query = query.options(joinedload(Booking.bill_1))
        query = query.options(joinedload(Booking.contact))
        booking_list = query.all()

        # Serialize with relationships included
        result = []
        for booking_item in booking_list:
            item_dict = booking_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)
            if booking_item.employee:
                related_obj = booking_item.employee
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['employee'] = related_dict
            else:
                item_dict['employee'] = None
            if booking_item.bill:
                related_obj = booking_item.bill
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['bill'] = related_dict
            else:
                item_dict['bill'] = None
            if booking_item.bill_1:
                related_obj = booking_item.bill_1
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['bill_1'] = related_dict
            else:
                item_dict['bill_1'] = None
            if booking_item.contact:
                related_obj = booking_item.contact
                related_dict = related_obj.__dict__.copy()
                related_dict.pop('_sa_instance_state', None)
                item_dict['contact'] = related_dict
            else:
                item_dict['contact'] = None

            # Add many-to-many and one-to-many relationship objects (full details)
            guest_list = database.query(Guest).join(guests, Guest.id == guests.c.guest).filter(guests.c.guests == booking_item.id).all()
            item_dict['guest'] = []
            for guest_obj in guest_list:
                guest_dict = guest_obj.__dict__.copy()
                guest_dict.pop('_sa_instance_state', None)
                item_dict['guest'].append(guest_dict)
            reservedroom_list = database.query(ReservedRoom).filter(ReservedRoom.booking_id == booking_item.id).all()
            item_dict['reservedRooms'] = []
            for reservedroom_obj in reservedroom_list:
                reservedroom_dict = reservedroom_obj.__dict__.copy()
                reservedroom_dict.pop('_sa_instance_state', None)
                item_dict['reservedRooms'].append(reservedroom_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Booking).all()


@router.get("/booking/count/", response_model=None, tags=["Booking"])
def get_count_booking(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Booking entities"""
    count = database.query(Booking).count()
    return {"count": count}


@router.get("/booking/paginated/", response_model=None, tags=["Booking"])
def get_paginated_booking(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Booking entities"""
    total = database.query(Booking).count()
    booking_list = database.query(Booking).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": booking_list
        }

    result = []
    for booking_item in booking_list:
        guest_ids = database.query(guests.c.guest).filter(guests.c.guests == booking_item.id).all()
        reservedRooms_ids = database.query(ReservedRoom.id).filter(ReservedRoom.booking_id == booking_item.id).all()
        item_data = {
            "booking": booking_item,
            "guest_ids": [x[0] for x in guest_ids],
            "reservedRooms_ids": [x[0] for x in reservedRooms_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }
@router.get("/booking/search/", response_model=None, tags=["Booking"])
def search_booking(
    bookingNumber: str = None,
    id: int = None,
    totalPrice: float = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Booking entities by attributes"""
    query = database.query(Booking)

    if bookingNumber is not None:
        query = query.filter(Booking.bookingNumber.ilike(f"%{bookingNumber}%"))
    if id is not None:
        query = query.filter(Booking.id == id)
    if totalPrice is not None:
        query = query.filter(Booking.totalPrice == totalPrice)

    results = query.all()
    return results


@router.get("/booking/{booking_id}/", response_model=None, tags=["Booking"])
async def get_booking(booking_id: int, database: Session = Depends(get_db)) -> Booking:
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    guest_ids = database.query(guests.c.guest).filter(guests.c.guests == db_booking.id).all()
    reservedRooms_ids = database.query(ReservedRoom.id).filter(ReservedRoom.booking_id == db_booking.id).all()
    response_data = {
        "booking": db_booking,
        "guest_ids": [x[0] for x in guest_ids],
        "reservedRooms_ids": [x[0] for x in reservedRooms_ids]}
    return response_data



@router.post("/booking/", response_model=None, tags=["Booking"])
async def create_booking(booking_data: BookingCreate, database: Session = Depends(get_db)) -> Booking:

    if booking_data.employee is not None:
        db_employee = database.query(Employee).filter(Employee.id == booking_data.employee).first()
        if not db_employee:
            raise HTTPException(status_code=400, detail="Employee not found")
    else:
        raise HTTPException(status_code=400, detail="Employee ID is required")
    if booking_data.contact is not None:
        db_contact = database.query(Person).filter(Person.id == booking_data.contact).first()
        if not db_contact:
            raise HTTPException(status_code=400, detail="Person not found")
    else:
        raise HTTPException(status_code=400, detail="Person ID is required")
    if not booking_data.guest or len(booking_data.guest) < 1:
        raise HTTPException(status_code=400, detail="At least 1 Guest(s) required")
    if booking_data.guest:
        for id in booking_data.guest:
            # Entity already validated before creation
            db_guest = database.query(Guest).filter(Guest.id == id).first()
            if not db_guest:
                raise HTTPException(status_code=404, detail=f"Guest with ID {id} not found")

    db_booking = Booking(
        departureDate=booking_data.departureDate,        commercialStatus=booking_data.commercialStatus.value,        bookingNumber=booking_data.bookingNumber,        physicalStatus=booking_data.physicalStatus.value,        arrivalDate=booking_data.arrivalDate,        totalPrice=booking_data.totalPrice,        employee_id=booking_data.employee,        contact_id=booking_data.contact        )

    database.add(db_booking)
    database.flush()
    database.refresh(db_booking)

    if booking_data.reservedRooms:
        # Validate that all ReservedRoom IDs exist
        for reservedroom_id in booking_data.reservedRooms:
            db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == reservedroom_id).first()
            if not db_reservedroom:
                raise HTTPException(status_code=400, detail=f"ReservedRoom with id {reservedroom_id} not found")

        # Update the related entities with the new foreign key
        database.query(ReservedRoom).filter(ReservedRoom.id.in_(booking_data.reservedRooms)).update(
            {ReservedRoom.booking_id: db_booking.id}, synchronize_session=False
        )
        database.flush()

    if booking_data.guest:
        for id in booking_data.guest:
            # Entity already validated before creation
            db_guest = database.query(Guest).filter(Guest.id == id).first()
            # Create the association
            association = guests.insert().values(guests=db_booking.id, guest=db_guest.id)
            database.execute(association)
            database.flush()


    guest_ids = database.query(guests.c.guest).filter(guests.c.guests == db_booking.id).all()
    reservedRooms_ids = database.query(ReservedRoom.id).filter(ReservedRoom.booking_id == db_booking.id).all()
    response_data = {
        "booking": db_booking,
        "guest_ids": [x[0] for x in guest_ids],
        "reservedRooms_ids": [x[0] for x in reservedRooms_ids]    }
    database.commit()
    database.refresh(db_booking)
    return response_data


@router.post("/booking/bulk/", response_model=None, tags=["Booking"])
async def bulk_create_booking(items: list[BookingCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Booking entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item
            if not item_data.employee:
                raise ValueError("Employee ID is required")
            if not item_data.contact:
                raise ValueError("Person ID is required")

            db_booking = Booking(
                departureDate=item_data.departureDate,                commercialStatus=item_data.commercialStatus.value,                bookingNumber=item_data.bookingNumber,                physicalStatus=item_data.physicalStatus.value,                arrivalDate=item_data.arrivalDate,                totalPrice=item_data.totalPrice,                employee_id=item_data.employee,                contact_id=item_data.contact            )
            database.add(db_booking)
            database.flush()  # Get ID without committing
            created_items.append(db_booking.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Booking entities"
    }


@router.delete("/booking/bulk/", response_model=None, tags=["Booking"])
async def bulk_delete_booking(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Booking entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_booking = database.query(Booking).filter(Booking.id == item_id).first()
        if db_booking:
            database.delete(db_booking)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Booking entities"
    }

@router.put("/booking/{booking_id}/", response_model=None, tags=["Booking"])
async def update_booking(booking_id: int, booking_data: BookingCreate, database: Session = Depends(get_db)) -> Booking:
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    setattr(db_booking, 'departureDate', booking_data.departureDate)
    setattr(db_booking, 'commercialStatus', booking_data.commercialStatus.value)
    setattr(db_booking, 'bookingNumber', booking_data.bookingNumber)
    setattr(db_booking, 'physicalStatus', booking_data.physicalStatus.value)
    setattr(db_booking, 'arrivalDate', booking_data.arrivalDate)
    setattr(db_booking, 'totalPrice', booking_data.totalPrice)
    if booking_data.employee is not None:
        db_employee = database.query(Employee).filter(Employee.id == booking_data.employee).first()
        if not db_employee:
            raise HTTPException(status_code=400, detail="Employee not found")
        setattr(db_booking, 'employee_id', booking_data.employee)
    if booking_data.contact is not None:
        db_contact = database.query(Person).filter(Person.id == booking_data.contact).first()
        if not db_contact:
            raise HTTPException(status_code=400, detail="Person not found")
        setattr(db_booking, 'contact_id', booking_data.contact)
    if booking_data.reservedRooms is not None:
        requested_reservedRooms = set(booking_data.reservedRooms)
        current_reservedRooms = {
            getattr(item, 'id')
            for item in database.query(ReservedRoom).filter(ReservedRoom.booking_id == db_booking.id).all()
        }
        reservedRooms_to_detach = current_reservedRooms - requested_reservedRooms
        if reservedRooms_to_detach:
            raise HTTPException(status_code=409, detail="ReservedRoom " + ", ".join(str(item) for item in sorted(reservedRooms_to_detach)) + " requires a booking: reassign it instead of removing it")
        reservedRooms_to_attach = requested_reservedRooms - current_reservedRooms
        for reservedroom_id in reservedRooms_to_attach:
            db_reservedroom = database.query(ReservedRoom).filter(ReservedRoom.id == reservedroom_id).first()
            if not db_reservedroom:
                raise HTTPException(status_code=400, detail="ReservedRoom with id " + str(reservedroom_id) + " not found")
        if reservedRooms_to_attach:
            database.query(ReservedRoom).filter(ReservedRoom.id.in_(reservedRooms_to_attach)).update(
                {ReservedRoom.booking_id: db_booking.id}, synchronize_session=False
            )
    if booking_data.guest is not None:
        if len(booking_data.guest) < 1:
            raise HTTPException(status_code=400, detail="At least 1 Guest(s) required")
        existing_guest_ids = [assoc.guest for assoc in database.execute(
            guests.select().where(guests.c.guests == db_booking.id))]

        guests_to_remove = set(existing_guest_ids) - set(booking_data.guest)
        for guest_id in guests_to_remove:
            association = guests.delete().where(
                (guests.c.guests == db_booking.id) & (guests.c.guest == guest_id))
            database.execute(association)

        new_guest_ids = set(booking_data.guest) - set(existing_guest_ids)
        for guest_id in new_guest_ids:
            db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
            if db_guest is None:
                raise HTTPException(status_code=404, detail="Guest with ID " + str(guest_id) + " not found")
            association = guests.insert().values(guest=db_guest.id, guests=db_booking.id)
            database.execute(association)
    database.commit()
    database.refresh(db_booking)

    guest_ids = database.query(guests.c.guest).filter(guests.c.guests == db_booking.id).all()
    reservedRooms_ids = database.query(ReservedRoom.id).filter(ReservedRoom.booking_id == db_booking.id).all()
    response_data = {
        "booking": db_booking,
        "guest_ids": [x[0] for x in guest_ids],
        "reservedRooms_ids": [x[0] for x in reservedRooms_ids]    }
    return response_data


@router.delete("/booking/{booking_id}/", response_model=None, tags=["Booking"])
async def delete_booking(booking_id: int, database: Session = Depends(get_db)):
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")
    related_reservedRooms = db_booking.reservedRooms
    for related in (list(related_reservedRooms) if isinstance(related_reservedRooms, list) else [item for item in [related_reservedRooms] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Booking " + str(booking_id) + ": ReservedRoom " + str(getattr(related, 'id')) + " requires at least 1 booking")
    related_bill = db_booking.bill
    for related in (list(related_bill) if isinstance(related_bill, list) else [item for item in [related_bill] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Booking " + str(booking_id) + ": Bill " + str(getattr(related, 'id')) + " requires at least 1 booking_1")
    related_bill_1 = db_booking.bill_1
    for related in (list(related_bill_1) if isinstance(related_bill_1, list) else [item for item in [related_bill_1] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Booking " + str(booking_id) + ": Bill " + str(getattr(related, 'id')) + " requires at least 1 booking")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_booking = {
        attr.key: getattr(db_booking, attr.key)
        for attr in db_booking.__mapper__.column_attrs
    }
    database.delete(db_booking)
    database.commit()
    return deleted_booking

@router.post("/booking/{booking_id}/guest/{guest_id}/", response_model=None, tags=["Booking Relationships"])
async def add_guest_to_booking(booking_id: int, guest_id: int, database: Session = Depends(get_db)):
    """Add a Guest to this Booking's guest relationship"""
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    db_guest = database.query(Guest).filter(Guest.id == guest_id).first()
    if db_guest is None:
        raise HTTPException(status_code=404, detail="Guest not found")

    # Check if relationship already exists
    existing = database.query(guests).filter(
        (guests.c.guests == booking_id) &
        (guests.c.guest == guest_id)
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Relationship already exists")

    # Create the association
    association = guests.insert().values(guests=booking_id, guest=guest_id)
    database.execute(association)
    database.commit()

    return {"message": "Guest added to guest successfully"}


@router.delete("/booking/{booking_id}/guest/{guest_id}/", response_model=None, tags=["Booking Relationships"])
async def remove_guest_from_booking(booking_id: int, guest_id: int, database: Session = Depends(get_db)):
    """Remove a Guest from this Booking's guest relationship"""
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    # Check if relationship exists
    existing = database.query(guests).filter(
        (guests.c.guests == booking_id) &
        (guests.c.guest == guest_id)
    ).first()

    if not existing:
        raise HTTPException(status_code=404, detail="Relationship not found")

    if len(db_booking.guest) - 1 < 1:
        raise HTTPException(status_code=409, detail="Booking " + str(booking_id) + " requires at least 1 guest")

    # Delete the association
    association = guests.delete().where(
        (guests.c.guests == booking_id) &
        (guests.c.guest == guest_id)
    )
    database.execute(association)
    database.commit()

    return {"message": "Guest removed from guest successfully"}


@router.get("/booking/{booking_id}/guest/", response_model=None, tags=["Booking Relationships"])
async def get_guest_of_booking(booking_id: int, database: Session = Depends(get_db)):
    """Get all Guest entities related to this Booking through guest"""
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    guest_ids = database.query(guests.c.guest).filter(guests.c.guests == booking_id).all()
    guest_list = database.query(Guest).filter(Guest.id.in_([id[0] for id in guest_ids])).all()

    return {
        "booking_id": booking_id,
        "guest_count": len(guest_list),
        "guest": guest_list
    }


@router.get("/booking/{booking_id}/reservedRooms/", response_model=None, tags=["Booking Relationships"])
async def get_reservedRooms_of_booking(booking_id: int, database: Session = Depends(get_db)):
    """Get all ReservedRoom entities related to this Booking through reservedRooms"""
    db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
    if db_booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")

    reservedRooms_list = database.query(ReservedRoom).filter(ReservedRoom.booking_id == booking_id).all()

    return {
        "booking_id": booking_id,
        "reservedRooms_count": len(reservedRooms_list),
        "reservedRooms": reservedRooms_list
    }


