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
    CHECKED_IN = "CHECKED_IN"
    NOT_ARRIVED = "NOT_ARRIVED"
    CHECKED_OUT = "CHECKED_OUT"

class BookingCommercialStatus(Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"

############################################
# Classes are defined here
############################################
class BillCreate(BaseModel):
    issuedDate: date
    totalAmount: float
    billNumber: int
    settled: bool
    billBooking: int  # 1:1 Relationship (mandatory)


class BookedRoomCreate(BaseModel):
    agreedPrice: float
    booking: int  # N:1 Relationship (mandatory)
    booking_1: int  # N:1 Relationship (mandatory)
    room: int  # N:1 Relationship (mandatory)
    extraCharges: Optional[float] = 0.0  # Additional charges per room



class BookingCreate(BaseModel):
    commercialStatus: BookingCommercialStatus
    bookingNumber: int
    physicalStatus: BookingPhysicalStatus
    arrivalDate: date
    totalPrice: float
    departureDate: date
    contact: int  # N:1 Relationship (mandatory)
    guest: List[int]  # N:M Relationship
    employee: int  # N:1 Relationship (mandatory)
    bookedroom: Optional[List[int]] = None  # 1:N Relationship
    bookedRooms: Optional[List[int]] = None  # 1:N Relationship

    @model_validator(mode='after')
    def validate_arrivalBeforeDeparture(self):
        """OCL Constraint: arrivalBeforeDeparture"""
        if not (self.arrivalDate <= self.departureDate):
            raise ValueError("Constraint 'arrivalBeforeDeparture' violated: self.arrivalDate <= self.departureDate")
        return self

class RoomCreate(BaseModel):
    description: str
    roomNumber: str
    standardPrice: float
    capacity: int
    bookedroom: Optional[List[int]] = None  # 1:N Relationship


class PersonCreate(BaseModel):
    phone: str
    firstName: str
    email: str
    familyName: str
    booking: Optional[List[int]] = None  # 1:N Relationship

    @field_validator('email')
    @classmethod
    def validate_email_1(cls, v):
        """OCL Constraint: validEmail"""
        if not (re.match(r'.+@.+\.[A-Za-z]{2,}', v) is not None):
            raise ValueError("email must match '.+@.+\\.[A-Za-z]{2,}'")
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


