from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()

############################################
#   Booking Method Endpoints
############################################





@router.post("/booking/methods/registerDeparture/", response_model=None, tags=["Booking Methods"])
async def booking_registerDeparture(
    database: Session = Depends(get_db)
):
    """
    Execute the registerDeparture class method on Booking.
    This method operates on all Booking entities or performs class-level operations.
    """
    try:
        # Booking.registerDeparture: capture stdout to include print outputs in the response
        import io
        import sys
        _registerDeparture_stdout = io.StringIO()
        sys.stdout = _registerDeparture_stdout



        # Booking.registerDeparture: check if booking exists and is in CHECKED_IN state
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        if db_booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Check if booking is already checked out or cancelled
        if db_booking.physicalStatus == "CHECKED_OUT" or db_booking.commercialStatus == "CANCELLED":
            raise HTTPException(status_code=400, detail="Booking is already checked out or cancelled")
        
        # Only allow registration if current status is CHECKED_IN
        if db_booking.physicalStatus != "CHECKED_IN":
            raise HTTPException(status_code=400, detail="Booking must be checked in before checking out")
        
        # Update the physical status to CHECKED_OUT
        db_booking.physicalStatus = "CHECKED_OUT"
        database.commit()
        
        sys.stdout = sys.__stdout__
        return {"success": True, "message": "Booking checked out successfully"}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerDeparture failed: {str(e)}")







@router.post("/booking/methods/computeAmountOwed/", response_model=None, tags=["Booking Methods"])
async def booking_computeAmountOwed(
    database: Session = Depends(get_db)
):
    """
    Execute the computeAmountOwed class method on Booking.
    This method operates on all Booking entities or performs class-level operations.
    """
    try:
        # Booking.computeAmountOwed: capture stdout to include print outputs in the response
        import io
        import sys
        _computeAmountOwed_stdout = io.StringIO()
        sys.stdout = _computeAmountOwed_stdout



        # Booking.computeAmountOwed: calculate total amount owed
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        if db_booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Calculate the number of nights
        nights = (db_booking.departureDate - db_booking.arrivalDate).days
        if nights <= 0:
            nights = 1  # At least one night
        
        # Calculate total amount from booking rooms
        total_amount = 0.0
        for bookingroom in db_booking.bookingRooms:
            total_amount += bookingroom.agreedPrice * nights
            total_amount += bookingroom.extraCharges
        
        sys.stdout = sys.__stdout__
        return {"totalAmountOwed": total_amount}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.computeAmountOwed failed: {str(e)}")







@router.post("/booking/methods/cancel/", response_model=None, tags=["Booking Methods"])
async def booking_cancel(
    database: Session = Depends(get_db)
):
    """
    Execute the cancel class method on Booking.
    This method operates on all Booking entities or performs class-level operations.
    """
    try:
        # Booking.cancel: capture stdout to include print outputs in the response
        import io
        import sys
        _cancel_stdout = io.StringIO()
        sys.stdout = _cancel_stdout



        # Booking.cancel: check if booking exists
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        if db_booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Check if booking is already cancelled
        if db_booking.commercialStatus == "CANCELLED":
            raise HTTPException(status_code=400, detail="Booking is already cancelled")
        
        # Update the commercial status to CANCELLED
        db_booking.commercialStatus = "CANCELLED"
        
        # Release all associated BookingRoom entries
        for bookingroom in db_booking.bookingRooms:
            bookingroom.booking_id = None
        
        database.commit()
        
        sys.stdout = sys.__stdout__
        return {"success": True, "message": "Booking cancelled successfully"}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.cancel failed: {str(e)}")







@router.post("/booking/methods/produceBill/", response_model=None, tags=["Booking Methods"])
async def booking_produceBill(
    database: Session = Depends(get_db)
):
    """
    Execute the produceBill class method on Booking.
    This method operates on all Booking entities or performs class-level operations.
    """
    try:
        # Booking.produceBill: capture stdout to include print outputs in the response
        import io
        import sys
        _produceBill_stdout = io.StringIO()
        sys.stdout = _produceBill_stdout



        # Booking.produceBill: check if booking exists
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        if db_booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Check if booking already has a bill
        if db_booking.bill is not None:
            raise HTTPException(status_code=400, detail="Booking already has a bill")
        
        # Compute the total amount owed
        total_amount_owed = computeAmountOwed(db_booking.id, database)
        
        # Create a new bill
        db_bill = Bill(
            issuedDate=dt_date.today(),
            totalAmountDue=total_amount_owed,
            settled=False,
            booking_id=db_booking.id
        )
        
        database.add(db_bill)
        database.commit()
        
        sys.stdout = sys.__stdout__
        return {"success": True, "message": "Bill produced successfully", "bill_id": db_bill.id}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.produceBill failed: {str(e)}")







@router.post("/booking/methods/registerArrival/", response_model=None, tags=["Booking Methods"])
async def booking_registerArrival(
    database: Session = Depends(get_db)
):
    """
    Execute the registerArrival class method on Booking.
    This method operates on all Booking entities or performs class-level operations.
    """
    try:
        # Booking.registerArrival: capture stdout to include print outputs in the response
        import io
        import sys
        _registerArrival_stdout = io.StringIO()
        sys.stdout = _registerArrival_stdout



        # Booking.registerArrival: check if booking exists and is not already arrived
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        if db_booking is None:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Check if booking is already cancelled
        if db_booking.commercialStatus == "CANCELLED":
            raise HTTPException(status_code=400, detail="Cancelled bookings cannot be checked in")
        
        # Only allow registration if current status is NOT_ARRIVED
        if db_booking.physicalStatus != "NOT_ARRIVED":
            raise HTTPException(status_code=400, detail="Booking is already checked in or checked out")
        
        # Update the physical status to CHECKED_IN
        db_booking.physicalStatus = "CHECKED_IN"
        database.commit()
        
        sys.stdout = sys.__stdout__
        return {"success": True, "message": "Booking checked in successfully"}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerArrival failed: {str(e)}")


