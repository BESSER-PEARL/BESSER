import re
from datetime import datetime, date, time
from typing import Any, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator

from abc import ABC, abstractmethod

############################################
# Enumerations are defined here
############################################

class BookingPhysicalStatus(Enum):
    CHECKED_OUT = "CHECKED_OUT"
    CHECKED_IN = "CHECKED_IN"
    NOT_ARRIVED = "NOT_ARRIVED"

class BookingCommercialStatus(Enum):
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    CANCELLED = "CANCELLED"
    CONFIRMED = "CONFIRMED"

############################################
# Classes are defined here
############################################
class BillCreate(BaseModel):
    issuedDate: date
    settled: bool
    totalAmountDue: float
    booking: int  # 1:1 Relationship (mandatory)

    @model_validator(mode='after')
    def validate_totalAmountDue(self):
        """OCL Constraint: totalAmountDue must be computed from agreed room prices, stay length, and extra charges"""
        # This is a placeholder. The actual validation is done in the backend.
        # The totalAmountDue is computed, not manually entered.
        return self


class ReservedRoomCreate(BaseModel):
    agreedPrice: float
    booking: int  # N:1 Relationship (mandatory)
    room: int  # N:1 Relationship (mandatory)


class BookingCreate(BaseModel):
    departureDate: date
    commercialStatus: BookingCommercialStatus
    physicalStatus: BookingPhysicalStatus
    arrivalDate: date
    employee: int  # N:1 Relationship (mandatory)
    contact: int  # N:1 Relationship (mandatory)
    guest: List[int]  # N:M Relationship
    reservedRooms: Optional[List[int]] = None  # 1:N Relationship
    totalAmountDue: Optional[float] = None  # Derived field, not client-supplied

    @model_validator(mode='after')
    def validate_arrivalBeforeDeparture(self):
        """OCL Constraint: arrivalBeforeDeparture"""
        if not (self.arrivalDate <= self.departureDate):
            raise ValueError("Constraint 'arrivalBeforeDeparture' violated: self.arrivalDate <= self.departureDate")
        return self
    # NOTE: OCL constraint 'atLeastOneRoom' involves collections/relationships and is not enforced by this Create model.
    # NOTE: OCL constraint 'reservedRooms_at_least_1' involves collections/relationships and is not enforced by this Create model.

class RoomCreate(BaseModel):
    capacity: int
    description: str
    roomNumber: str
    standardPrice: float
    reservedroom: Optional[List[int]] = None  # 1:N Relationship


class PersonCreate(BaseModel):
    email: str
    familyName: str
    firstName: str
    phone: str
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
        if not (re.match(r'\+?\d{7,15}', v) is not None):
            raise ValueError("phone must match '\\+?\\d{7,15}'")
        return v

class EmployeeCreate(PersonCreate):
    handledBy: Optional[List[int]] = None  # 1:N Relationship


class GuestCreate(PersonCreate):
    guests: Optional[List[int]] = None  # N:M Relationship (optional)


