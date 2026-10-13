from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from pydantic_classes import *
from sql_alchemy import *
from database import get_db
from bal_stdlib import *

router = APIRouter()

############################################
#   Bill Method Endpoints
############################################





@router.post("/bill/methods/registerPayment/", response_model=None, tags=["Bill Methods"])
async def bill_registerPayment(
    database: Session = Depends(get_db)
):
    """
    Execute the registerPayment class method on Bill.
    This method operates on all Bill entities or performs class-level operations.
    """
    try:
        # Bill.registerPayment: capture stdout to include print outputs in the response
        import io
        import sys
        _registerPayment_stdout = io.StringIO()
        sys.stdout = _registerPayment_stdout



        # Bill.registerPayment: register the payment of the bill.
        # The settled status is updated to True.
        # The booking's commercialStatus is updated to CONFIRMED.
        # This is a placeholder implementation.
        sys.stdout = sys.__stdout__
        # For demonstration, we return a success message.
        return {"message": "Payment registered successfully"}
    except HTTPException:
        sys.stdout = sys.__stdout__
        raise
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Bill.registerPayment failed: {str(e)}")


