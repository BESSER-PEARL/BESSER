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

        # Get the booking from the database
        # Since this is a class method, we need to determine which booking to operate on
        # For now, we'll assume we're operating on a single booking, but this needs to be defined
        # In a real implementation, we might pass a booking ID in the request body
        # For now, we'll just use a placeholder
        booking_id = 1  # This should be passed in the request
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        
        if not db_booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Check if the booking is already cancelled
        if db_booking.commercialStatus == BookingCommercialStatus.CANCELLED:
            return {"success": False, "message": "Booking is already cancelled"}

        # Check if the booking is awaiting payment
        if db_booking.commercialStatus != BookingCommercialStatus.AWAITING_PAYMENT:
            return {"success": False, "message": "Booking is not in AWAITING_PAYMENT state"}

        # Cancel the booking
        db_booking.commercialStatus = BookingCommercialStatus.CANCELLED
        database.commit()
        
        # Return success status
        return {"success": True, "message": "Booking cancelled successfully"}
    
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.cancel failed: {str(e)}")







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

        # Get the booking from the database
        # Since this is a class method, we need to determine which booking to operate on
        # For now, we'll assume we're operating on a single booking, but this needs to be defined
        # In a real implementation, we might pass a booking ID in the request body
        # For now, we'll just use a placeholder
        booking_id = 1  # This should be passed in the request
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        
        if not db_booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Check if the booking is already checked in
        if db_booking.physicalStatus == BookingPhysicalStatus.CHECKED_IN:
            return {"success": False, "message": "Booking is already checked in"}

        # Check if the booking is not arrived yet
        if db_booking.physicalStatus != BookingPhysicalStatus.NOT_ARRIVED:
            return {"success": False, "message": "Booking is not in NOT_ARRIVED state"}

        # Register arrival
        db_booking.physicalStatus = BookingPhysicalStatus.CHECKED_IN
        database.commit()
        
        # Return success status
        return {"success": True, "message": "Arrival registered successfully"}
    
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerArrival failed: {str(e)}")







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

        # Get the booking from the database
        # Since this is a class method, we need to determine which booking to operate on
        # For now, we'll assume we're operating on a single booking, but this needs to be defined
        # In a real implementation, we might pass a booking ID in the request body
        # For now, we'll just use a placeholder
        booking_id = 1  # This should be passed in the request
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        
        if not db_booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Calculate the total amount owed
        total_amount = 0
        
        # Calculate based on booked rooms
        for booked_room in db_booking.bookedRooms:
            # Calculate number of days
            days = (db_booking.departureDate - db_booking.arrivalDate).days
            if days <= 0:
                days = 1  # At least one day
            
            # Add room cost
            total_amount += booked_room.get_total_cost(days)

        # Update the booking's total price
        db_booking.totalPrice = total_amount
        database.commit()
        
        # Return the total amount
        return {"totalAmount": total_amount, "message": "Amount computed successfully"}
    
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.computeAmountOwed failed: {str(e)}")







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

        # Get the booking from the database
        # Since this is a class method, we need to determine which booking to operate on
        # For now, we'll assume we're operating on a single booking, but this needs to be defined
        # In a real implementation, we might pass a booking ID in the request body
        # For now, we'll just use a placeholder
        booking_id = 1  # This should be passed in the request
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        
        if not db_booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Check if the booking is already checked out
        if db_booking.physicalStatus == BookingPhysicalStatus.CHECKED_OUT:
            return {"success": False, "message": "Booking is already checked out"}

        # Check if the booking is checked in
        if db_booking.physicalStatus != BookingPhysicalStatus.CHECKED_IN:
            return {"success": False, "message": "Booking is not in CHECKED_IN state"}

        # Register departure
        db_booking.physicalStatus = BookingPhysicalStatus.CHECKED_OUT
        database.commit()
        
        # Return success status
        return {"success": True, "message": "Departure registered successfully"}
    
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerDeparture failed: {str(e)}")







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

        # Get the booking from the database
        # Since this is a class method, we need to determine which booking to operate on
        # For now, we'll assume we're operating on a single booking, but this needs to be defined
        # In a real implementation, we might pass a booking ID in the request body
        # For now, we'll just use a placeholder
        booking_id = 1  # This should be passed in the request
        db_booking = database.query(Booking).filter(Booking.id == booking_id).first()
        
        if not db_booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        # Check if a bill already exists for this booking
        if db_booking.bill:
            raise HTTPException(status_code=400, detail="A bill already exists for this booking")

        # Calculate the total amount owed
        total_amount = db_booking.computeAmountOwed()

        # Create a new bill
        new_bill = Bill(
            billNumber=1,  # This should be generated
            issuedDate=dt_date.today(),
            totalAmount=total_amount,
            settled=False,
            billBooking_id=db_booking.id
        )
        
        database.add(new_bill)
        database.commit()
        
        # Return success status
        return {"success": True, "message": "Bill produced successfully", "bill_id": new_bill.id}
    
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.produceBill failed: {str(e)}")


