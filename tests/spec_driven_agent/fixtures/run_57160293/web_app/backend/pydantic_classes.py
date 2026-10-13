import re
from datetime import datetime, date, time
from typing import Any, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator

from abc import ABC, abstractmethod

############################################
# Enumerations are defined here
############################################

class BookingCommercialStatus(Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"

class BookingPhysicalStatus(Enum):
    CHECKED_IN = "CHECKED_IN"
    CHECKED_OUT = "CHECKED_OUT"
    NOT_YET_ARRIVED = "NOT_YET_ARRIVED"

############################################
# Classes are defined here
############################################
class ReservedRoomCreate(BaseModel):
    extraCharges: float
    agreedPrice: float
    room: int  # N:1 Relationship (mandatory)
    booking: int  # N:1 Relationship (mandatory)


class BookingCreate(BaseModel):
    totalPrice: float
    commercialStatus: BookingCommercialStatus
    physicalStatus: BookingPhysicalStatus
    arrivalDate: date
    departureDate: date
    reservedRooms: Optional[List[int]] = None  # 1:N Relationship
    contact: int  # N:1 Relationship (mandatory)
    guest: List[int]  # N:M Relationship
    employee: int  # N:1 Relationship (mandatory)

    @model_validator(mode='after')
    def validate_arrivalBeforeDeparture(self):
        """OCL Constraint: arrivalBeforeDeparture"""
        if not (self.arrivalDate <= self.departureDate):
            raise ValueError("Constraint 'arrivalBeforeDeparture' violated: self.arrivalDate <= self.departureDate")
        return self

class RoomCreate(BaseModel):
    roomNumber: int
    standardNightlyPrice: float
    description: str
    capacity: int
    reservedroom: Optional[List[int]] = None  # 1:N Relationship


class BillCreate(BaseModel):
    isSettled: bool
    issuedDate: date
    totalAmountDue: float
    booking: int  # 1:1 Relationship (mandatory)


class PersonCreate(BaseModel):
    familyName: str
    firstName: str
    phone: str
    email: str
    booking: Optional[List[int]] = None  # 1:N Relationship

    @field_validator('email')
    @classmethod
    def validate_email_1(cls, v):
        """OCL Constraint: validEmail"""
        if not (re.match(r'.+@.+\.[A-Za-z]{2,}$', v) is not None):
            raise ValueError("email must match '.+@.+\\.[A-Za-z]{2,}$'")
        return v
    @field_validator('phone')
    @classmethod
    def validate_phone_2(cls, v):
        """OCL Constraint: validPhone"""
        if not (re.match(r'^\+?\d{7,15}$', v) is not None):
            raise ValueError("phone must match '^\\+?\\d{7,15}$'")
        return v

class EmployeeCreate(PersonCreate):
    handledBy: Optional[List[int]] = None  # 1:N Relationship


class GuestCreate(PersonCreate):
    guests: Optional[List[int]] = None  # N:M Relationship (optional)


