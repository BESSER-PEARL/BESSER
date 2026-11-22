import re
from datetime import datetime, date, time
from typing import Any, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator


############################################
# Enumerations are defined here
############################################

class BookingCommercialStatus(Enum):
    CANCELLED = "CANCELLED"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    CONFIRMED = "CONFIRMED"

class BookingPhysicalStatus(Enum):
    NOT_ARRIVED = "NOT_ARRIVED"
    CHECKED_OUT = "CHECKED_OUT"
    CHECKED_IN = "CHECKED_IN"

############################################
# Classes are defined here
############################################
class PersonCreate(BaseModel):
    firstName: str
    familyName: str
    phoneNumber: str
    emailAddress: str
    booking: Optional[List[int]] = None  # 1:N Relationship

    @field_validator('emailAddress')
    @classmethod
    def validate_emailAddress_1(cls, v):
        """OCL Constraint: validEmail"""
        if not (re.match(r'.+@.+\.[A-Za-z]{2,}', v) is not None):
            raise ValueError("emailAddress must match '.+@.+\\.[A-Za-z]{2,}'")
        return v
    @field_validator('phoneNumber')
    @classmethod
    def validate_phoneNumber_2(cls, v):
        """OCL Constraint: validPhone"""
        if not (re.match(r'\+?[0-9]{7,15}', v) is not None):
            raise ValueError("phoneNumber must match '\\+?[0-9]{7,15}'")
        return v

class EmployeeCreate(PersonCreate):
    handledBy: Optional[List[int]] = None  # 1:N Relationship


class BillCreate(BaseModel):
    totalAmountDue: float
    settled: bool
    issuedDate: date
    booking: int  # 1:1 Relationship (mandatory)


class BookingRoomCreate(BaseModel):
    agreedPrice: float
    extraCharges: float = 0.0
    booking: int  # N:1 Relationship (mandatory)
    room: int  # N:1 Relationship (mandatory)


class BookingCreate(BaseModel):
    arrivalDate: date
    totalPrice: float
    commercialStatus: BookingCommercialStatus
    departureDate: date
    physicalStatus: BookingPhysicalStatus
    bookingRooms: Optional[List[int]] = None  # 1:N Relationship
    guest: List[int]  # N:M Relationship
    employee: int  # N:1 Relationship (mandatory)
    contact: int  # N:1 Relationship (mandatory)

    @model_validator(mode='after')
    def validate_arrivalBeforeDeparture(self):
        """OCL Constraint: arrivalBeforeDeparture"""
        if not (self.arrivalDate <= self.departureDate):
            raise ValueError("Constraint 'arrivalBeforeDeparture' violated: self.arrivalDate <= self.departureDate")
        return self

class RoomCreate(BaseModel):
    roomNumber: str
    capacity: int
    standardNightlyPrice: float
    description: str
    bookingroom: Optional[List[int]] = None  # 1:N Relationship


class GuestCreate(PersonCreate):
    guests: Optional[List[int]] = None  # N:M Relationship (optional)


