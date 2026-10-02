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



        # Booking.cancel: no body in the model - be honest: 501, never a fake "executed" success.
        sys.stdout = sys.__stdout__
        raise HTTPException(
            status_code=501,
            detail="Method 'cancel' of Booking is modeled but has no implementation",
        )
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.cancel failed: {str(e)}")







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



        # Booking.computeAmountOwed: no body in the model - be honest: 501, never a fake "executed" success.
        sys.stdout = sys.__stdout__
        raise HTTPException(
            status_code=501,
            detail="Method 'computeAmountOwed' of Booking is modeled but has no implementation",
        )
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.computeAmountOwed failed: {str(e)}")







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



        # Booking.produceBill: no body in the model - be honest: 501, never a fake "executed" success.
        sys.stdout = sys.__stdout__
        raise HTTPException(
            status_code=501,
            detail="Method 'produceBill' of Booking is modeled but has no implementation",
        )
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



        # Booking.registerArrival: no body in the model - be honest: 501, never a fake "executed" success.
        sys.stdout = sys.__stdout__
        raise HTTPException(
            status_code=501,
            detail="Method 'registerArrival' of Booking is modeled but has no implementation",
        )
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerArrival failed: {str(e)}")







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



        # Booking.registerDeparture: no body in the model - be honest: 501, never a fake "executed" success.
        sys.stdout = sys.__stdout__
        raise HTTPException(
            status_code=501,
            detail="Method 'registerDeparture' of Booking is modeled but has no implementation",
        )
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Booking.registerDeparture failed: {str(e)}")


