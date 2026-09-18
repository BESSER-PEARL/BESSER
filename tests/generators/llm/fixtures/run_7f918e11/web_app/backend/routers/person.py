from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/person/", response_model=None, tags=["Person"])
def get_all_person(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Person)
        person_list = query.all()

        # Serialize with relationships included
        result = []
        for person_item in person_list:
            item_dict = person_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)

            # Add many-to-many and one-to-many relationship objects (full details)
            booking_list = database.query(Booking).filter(Booking.contact_id == person_item.id).all()
            item_dict['booking'] = []
            for booking_obj in booking_list:
                booking_dict = booking_obj.__dict__.copy()
                booking_dict.pop('_sa_instance_state', None)
                item_dict['booking'].append(booking_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Person).all()


@router.get("/person/count/", response_model=None, tags=["Person"])
def get_count_person(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Person entities"""
    count = database.query(Person).count()
    return {"count": count}


@router.get("/person/paginated/", response_model=None, tags=["Person"])
def get_paginated_person(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Person entities"""
    total = database.query(Person).count()
    person_list = database.query(Person).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": person_list
        }

    result = []
    for person_item in person_list:
        booking_ids = database.query(Booking.id).filter(Booking.contact_id == person_item.id).all()
        item_data = {
            "person": person_item,
            "booking_ids": [x[0] for x in booking_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }
@router.get("/person/search/", response_model=None, tags=["Person"])
def search_person(
    email: str = None,
    familyName: str = None,
    firstName: str = None,
    id: int = None,
    phone: str = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Person entities by attributes"""
    query = database.query(Person)

    if email is not None:
        query = query.filter(Person.email.ilike(f"%{email}%"))
    if familyName is not None:
        query = query.filter(Person.familyName.ilike(f"%{familyName}%"))
    if firstName is not None:
        query = query.filter(Person.firstName.ilike(f"%{firstName}%"))
    if id is not None:
        query = query.filter(Person.id == id)
    if phone is not None:
        query = query.filter(Person.phone.ilike(f"%{phone}%"))

    results = query.all()
    return results


@router.get("/person/{person_id}/", response_model=None, tags=["Person"])
async def get_person(person_id: int, database: Session = Depends(get_db)) -> Person:
    db_person = database.query(Person).filter(Person.id == person_id).first()
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_person.id).all()
    response_data = {
        "person": db_person,
        "booking_ids": [x[0] for x in booking_ids]}
    return response_data



@router.post("/person/", response_model=None, tags=["Person"])
async def create_person(person_data: PersonCreate, database: Session = Depends(get_db)) -> Person:


    db_person = Person(
        email=person_data.email,        familyName=person_data.familyName,        firstName=person_data.firstName,        phone=person_data.phone        )

    database.add(db_person)
    database.flush()
    database.refresh(db_person)

    if person_data.booking:
        # Validate that all Booking IDs exist
        for booking_id in person_data.booking:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail=f"Booking with id {booking_id} not found")

        # Update the related entities with the new foreign key
        database.query(Booking).filter(Booking.id.in_(person_data.booking)).update(
            {Booking.contact_id: db_person.id}, synchronize_session=False
        )
        database.flush()



    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_person.id).all()
    response_data = {
        "person": db_person,
        "booking_ids": [x[0] for x in booking_ids]    }
    database.commit()
    database.refresh(db_person)
    return response_data


@router.post("/person/bulk/", response_model=None, tags=["Person"])
async def bulk_create_person(items: list[PersonCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Person entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item

            db_person = Person(
                email=item_data.email,                familyName=item_data.familyName,                firstName=item_data.firstName,                phone=item_data.phone            )
            database.add(db_person)
            database.flush()  # Get ID without committing
            created_items.append(db_person.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Person entities"
    }


@router.delete("/person/bulk/", response_model=None, tags=["Person"])
async def bulk_delete_person(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Person entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_person = database.query(Person).filter(Person.id == item_id).first()
        if db_person:
            database.delete(db_person)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Person entities"
    }

@router.put("/person/{person_id}/", response_model=None, tags=["Person"])
async def update_person(person_id: int, person_data: PersonCreate, database: Session = Depends(get_db)) -> Person:
    db_person = database.query(Person).filter(Person.id == person_id).first()
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    setattr(db_person, 'email', person_data.email)
    setattr(db_person, 'familyName', person_data.familyName)
    setattr(db_person, 'firstName', person_data.firstName)
    setattr(db_person, 'phone', person_data.phone)
    if person_data.booking is not None:
        requested_booking = set(person_data.booking)
        current_booking = {
            getattr(item, 'id')
            for item in database.query(Booking).filter(Booking.contact_id == db_person.id).all()
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
                {Booking.contact_id: db_person.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_person)

    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_person.id).all()
    response_data = {
        "person": db_person,
        "booking_ids": [x[0] for x in booking_ids]    }
    return response_data


@router.delete("/person/{person_id}/", response_model=None, tags=["Person"])
async def delete_person(person_id: int, database: Session = Depends(get_db)):
    db_person = database.query(Person).filter(Person.id == person_id).first()
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    related_booking = db_person.booking
    for related in (list(related_booking) if isinstance(related_booking, list) else [item for item in [related_booking] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Person " + str(person_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 contact")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_person = {
        attr.key: getattr(db_person, attr.key)
        for attr in db_person.__mapper__.column_attrs
    }
    database.delete(db_person)
    database.commit()
    return deleted_person


@router.get("/person/{person_id}/booking/", response_model=None, tags=["Person Relationships"])
async def get_booking_of_person(person_id: int, database: Session = Depends(get_db)):
    """Get all Booking entities related to this Person through booking"""
    db_person = database.query(Person).filter(Person.id == person_id).first()
    if db_person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    booking_list = database.query(Booking).filter(Booking.contact_id == person_id).all()

    return {
        "person_id": person_id,
        "booking_count": len(booking_list),
        "booking": booking_list
    }


