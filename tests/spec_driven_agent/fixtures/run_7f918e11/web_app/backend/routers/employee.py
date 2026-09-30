from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()


@router.get("/employee/", response_model=None, tags=["Employee"])
def get_all_employee(detailed: bool = False, database: Session = Depends(get_db)) -> list:
    # Use detailed=true to get entities with eagerly loaded relationships (for tables with lookup columns)
    if detailed:
        # Eagerly load all relationships to avoid N+1 queries
        query = database.query(Employee)
        employee_list = query.all()

        # Serialize with relationships included
        result = []
        for employee_item in employee_list:
            item_dict = employee_item.__dict__.copy()
            item_dict.pop('_sa_instance_state', None)

            # Add many-to-one relationships (foreign keys for lookup columns)

            # Add many-to-many and one-to-many relationship objects (full details)
            booking_list = database.query(Booking).filter(Booking.employee_id == employee_item.id).all()
            item_dict['handledBy'] = []
            for booking_obj in booking_list:
                booking_dict = booking_obj.__dict__.copy()
                booking_dict.pop('_sa_instance_state', None)
                item_dict['handledBy'].append(booking_dict)
            booking_list = database.query(Booking).filter(Booking.contact_id == employee_item.id).all()
            item_dict['booking'] = []
            for booking_obj in booking_list:
                booking_dict = booking_obj.__dict__.copy()
                booking_dict.pop('_sa_instance_state', None)
                item_dict['booking'].append(booking_dict)

            result.append(item_dict)
        return result
    else:
        # Default: return flat entities (faster for charts/widgets without lookup columns)
        return database.query(Employee).all()


@router.get("/employee/count/", response_model=None, tags=["Employee"])
def get_count_employee(database: Session = Depends(get_db)) -> dict:
    """Get the total count of Employee entities"""
    count = database.query(Employee).count()
    return {"count": count}


@router.get("/employee/paginated/", response_model=None, tags=["Employee"])
def get_paginated_employee(skip: int = 0, limit: int = 100, detailed: bool = False, database: Session = Depends(get_db)) -> dict:
    """Get paginated list of Employee entities"""
    total = database.query(Employee).count()
    employee_list = database.query(Employee).offset(skip).limit(limit).all()
    # By default, return flat entities (for charts/widgets)
    # Use detailed=true to get entities with relationships
    if not detailed:
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": employee_list
        }

    result = []
    for employee_item in employee_list:
        handledBy_ids = database.query(Booking.id).filter(Booking.employee_id == employee_item.id).all()
        booking_ids = database.query(Booking.id).filter(Booking.contact_id == employee_item.id).all()
        item_data = {
            "employee": employee_item,
            "handledBy_ids": [x[0] for x in handledBy_ids],            "booking_ids": [x[0] for x in booking_ids]        }
        result.append(item_data)
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": result
    }
@router.get("/employee/search/", response_model=None, tags=["Employee"])
def search_employee(
    email: str = None,
    familyName: str = None,
    firstName: str = None,
    id: int = None,
    phone: str = None,
    database: Session = Depends(get_db)
) -> list:
    """Search Employee entities by attributes"""
    query = database.query(Employee)

    if email is not None:
        query = query.filter(Employee.email.ilike(f"%{email}%"))
    if familyName is not None:
        query = query.filter(Employee.familyName.ilike(f"%{familyName}%"))
    if firstName is not None:
        query = query.filter(Employee.firstName.ilike(f"%{firstName}%"))
    if id is not None:
        query = query.filter(Employee.id == id)
    if phone is not None:
        query = query.filter(Employee.phone.ilike(f"%{phone}%"))

    results = query.all()
    return results


@router.get("/employee/{employee_id}/", response_model=None, tags=["Employee"])
async def get_employee(employee_id: int, database: Session = Depends(get_db)) -> Employee:
    db_employee = database.query(Employee).filter(Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    handledBy_ids = database.query(Booking.id).filter(Booking.employee_id == db_employee.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_employee.id).all()
    response_data = {
        "employee": db_employee,
        "handledBy_ids": [x[0] for x in handledBy_ids],        "booking_ids": [x[0] for x in booking_ids]}
    return response_data



@router.post("/employee/", response_model=None, tags=["Employee"])
async def create_employee(employee_data: EmployeeCreate, database: Session = Depends(get_db)) -> Employee:


    db_employee = Employee(
        email=employee_data.email,        familyName=employee_data.familyName,        firstName=employee_data.firstName,        phone=employee_data.phone        )

    database.add(db_employee)
    database.flush()
    database.refresh(db_employee)

    if employee_data.handledBy:
        # Validate that all Booking IDs exist
        for booking_id in employee_data.handledBy:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail=f"Booking with id {booking_id} not found")

        # Update the related entities with the new foreign key
        database.query(Booking).filter(Booking.id.in_(employee_data.handledBy)).update(
            {Booking.employee_id: db_employee.id}, synchronize_session=False
        )
        database.flush()
    if employee_data.booking:
        # Validate that all Booking IDs exist
        for booking_id in employee_data.booking:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail=f"Booking with id {booking_id} not found")

        # Update the related entities with the new foreign key
        database.query(Booking).filter(Booking.id.in_(employee_data.booking)).update(
            {Booking.contact_id: db_employee.id}, synchronize_session=False
        )
        database.flush()



    handledBy_ids = database.query(Booking.id).filter(Booking.employee_id == db_employee.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_employee.id).all()
    response_data = {
        "employee": db_employee,
        "handledBy_ids": [x[0] for x in handledBy_ids],        "booking_ids": [x[0] for x in booking_ids]    }
    database.commit()
    database.refresh(db_employee)
    return response_data


@router.post("/employee/bulk/", response_model=None, tags=["Employee"])
async def bulk_create_employee(items: list[EmployeeCreate], database: Session = Depends(get_db)) -> dict:
    """Create multiple Employee entities at once"""
    created_items = []
    errors = []

    for idx, item_data in enumerate(items):
        try:
            # Basic validation for each item

            db_employee = Employee(
                email=item_data.email,                familyName=item_data.familyName,                firstName=item_data.firstName,                phone=item_data.phone            )
            database.add(db_employee)
            database.flush()  # Get ID without committing
            created_items.append(db_employee.id)
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        database.rollback()
        raise HTTPException(status_code=400, detail={"message": "Bulk creation failed", "errors": errors})

    database.commit()
    return {
        "created_count": len(created_items),
        "created_ids": created_items,
        "message": f"Successfully created {len(created_items)} Employee entities"
    }


@router.delete("/employee/bulk/", response_model=None, tags=["Employee"])
async def bulk_delete_employee(ids: list[int], database: Session = Depends(get_db)) -> dict:
    """Delete multiple Employee entities at once"""
    deleted_count = 0
    not_found = []

    for item_id in ids:
        db_employee = database.query(Employee).filter(Employee.id == item_id).first()
        if db_employee:
            database.delete(db_employee)
            deleted_count += 1
        else:
            not_found.append(item_id)

    database.commit()

    return {
        "deleted_count": deleted_count,
        "not_found": not_found,
        "message": f"Successfully deleted {deleted_count} Employee entities"
    }

@router.put("/employee/{employee_id}/", response_model=None, tags=["Employee"])
async def update_employee(employee_id: int, employee_data: EmployeeCreate, database: Session = Depends(get_db)) -> Employee:
    db_employee = database.query(Employee).filter(Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    setattr(db_employee, 'email', employee_data.email)
    setattr(db_employee, 'familyName', employee_data.familyName)
    setattr(db_employee, 'firstName', employee_data.firstName)
    setattr(db_employee, 'phone', employee_data.phone)
    if employee_data.handledBy is not None:
        requested_handledBy = set(employee_data.handledBy)
        current_handledBy = {
            getattr(item, 'id')
            for item in database.query(Booking).filter(Booking.employee_id == db_employee.id).all()
        }
        handledBy_to_detach = current_handledBy - requested_handledBy
        if handledBy_to_detach:
            raise HTTPException(status_code=409, detail="Booking " + ", ".join(str(item) for item in sorted(handledBy_to_detach)) + " requires a employee: reassign it instead of removing it")
        handledBy_to_attach = requested_handledBy - current_handledBy
        for booking_id in handledBy_to_attach:
            db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
            if not db_booking:
                raise HTTPException(status_code=400, detail="Booking with id " + str(booking_id) + " not found")
        if handledBy_to_attach:
            database.query(Booking).filter(Booking.id.in_(handledBy_to_attach)).update(
                {Booking.employee_id: db_employee.id}, synchronize_session=False
            )
    if employee_data.booking is not None:
        requested_booking = set(employee_data.booking)
        current_booking = {
            getattr(item, 'id')
            for item in database.query(Booking).filter(Booking.contact_id == db_employee.id).all()
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
                {Booking.contact_id: db_employee.id}, synchronize_session=False
            )
    database.commit()
    database.refresh(db_employee)

    handledBy_ids = database.query(Booking.id).filter(Booking.employee_id == db_employee.id).all()
    booking_ids = database.query(Booking.id).filter(Booking.contact_id == db_employee.id).all()
    response_data = {
        "employee": db_employee,
        "handledBy_ids": [x[0] for x in handledBy_ids],        "booking_ids": [x[0] for x in booking_ids]    }
    return response_data


@router.delete("/employee/{employee_id}/", response_model=None, tags=["Employee"])
async def delete_employee(employee_id: int, database: Session = Depends(get_db)):
    db_employee = database.query(Employee).filter(Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    related_handledBy = db_employee.handledBy
    for related in (list(related_handledBy) if isinstance(related_handledBy, list) else [item for item in [related_handledBy] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Employee " + str(employee_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 employee")
    related_booking = db_employee.booking
    for related in (list(related_booking) if isinstance(related_booking, list) else [item for item in [related_booking] if item is not None]):
        remaining = 0
        if remaining < 1:
            raise HTTPException(status_code=409, detail="Cannot delete Employee " + str(employee_id) + ": Booking " + str(getattr(related, 'id')) + " requires at least 1 contact")
    # Snapshot the columns before deleting: cascading the association-class
    # links loads relationship collections whose back-references would make
    # the JSON encoder recurse endlessly on the live object.
    deleted_employee = {
        attr.key: getattr(db_employee, attr.key)
        for attr in db_employee.__mapper__.column_attrs
    }
    database.delete(db_employee)
    database.commit()
    return deleted_employee


@router.get("/employee/{employee_id}/handledBy/", response_model=None, tags=["Employee Relationships"])
async def get_handledBy_of_employee(employee_id: int, database: Session = Depends(get_db)):
    """Get all Booking entities related to this Employee through handledBy"""
    db_employee = database.query(Employee).filter(Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    handledBy_list = database.query(Booking).filter(Booking.employee_id == employee_id).all()

    return {
        "employee_id": employee_id,
        "handledBy_count": len(handledBy_list),
        "handledBy": handledBy_list
    }

@router.get("/employee/{employee_id}/booking/", response_model=None, tags=["Employee Relationships"])
async def get_booking_of_employee(employee_id: int, database: Session = Depends(get_db)):
    """Get all Booking entities related to this Employee through booking"""
    db_employee = database.query(Employee).filter(Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    booking_list = database.query(Booking).filter(Booking.contact_id == employee_id).all()

    return {
        "employee_id": employee_id,
        "booking_count": len(booking_list),
        "booking": booking_list
    }


