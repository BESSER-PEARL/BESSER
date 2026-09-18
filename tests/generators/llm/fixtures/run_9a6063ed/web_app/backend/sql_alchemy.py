import enum
import os
# Both spellings are bound on purpose. Generated code uses the trailing-
# underscore aliases so a modelled class named List, Optional or DateTime
# cannot shadow them. The PLAIN names are imported too because this file is
# routinely edited afterwards by an LLM, which writes ordinary Python and has
# no reason to know about our aliasing convention. Observed in generated apps
# on 2026-09-11, each one an app that would not import:
#
#     Mapped_[Optional[int]]        undefined name 'Optional'
#     mapped_column(DateTime())     undefined name 'DateTime'
#
# Binding only the names we had seen fail left the same trap set for every
# other alias, so every aliased name below is bound both ways.
from typing import List, Optional, List as List_, Optional as Optional_
from sqlalchemy import (
    create_engine, Enum,
    Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, Interval,
    PickleType, String, Table, Text, Time,
    Column as Column_, ForeignKey as ForeignKey_, Table as Table_,
    Text as Text_, Boolean as Boolean_, String as String_, Date as Date_,
    Time as Time_, DateTime as DateTime_, Float as Float_, Integer as Integer_,
    Interval as Interval_, PickleType as PickleType_,
)
from sqlalchemy.ext.declarative import AbstractConcreteBase
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, Mapped as Mapped_, mapped_column,
    relationship
)
from datetime import datetime as dt_datetime, time as dt_time, date as dt_date, timedelta as dt_timedelta
from uuid import uuid4 as _uuid4


def _new_str_id() -> str:
    """Server-side surrogate for string primary keys named ``id``.

    A string PK has no autoincrement; without a default every INSERT dies
    on NOT NULL (the id is in nobody's hands: create schemas rightly
    exclude the server-owned ``id``, so the server must mint it).
    """
    return _uuid4().hex

class Base(DeclarativeBase):
    pass

# Definitions of Enumerations
class BookingPhysicalStatus(enum.Enum):
    CHECKED_IN = "CHECKED_IN"
    NOT_ARRIVED = "NOT_ARRIVED"
    CHECKED_OUT = "CHECKED_OUT"

class BookingCommercialStatus(enum.Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"


# Tables definition for many-to-many relationships
guests = Table_(
    "guests",
    Base.metadata,
    Column_("guests", ForeignKey_("booking.id"), primary_key=True),
    Column_("guest", ForeignKey_("guest.id"), primary_key=True),
)

# Tables definition
class Bill(Base):
    __tablename__ = "bill"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    billNumber: Mapped_[int] = mapped_column(Integer_)
    issuedDate: Mapped_[dt_date] = mapped_column(Date_)
    totalAmount: Mapped_[float] = mapped_column(Float_)
    settled: Mapped_[bool] = mapped_column(Boolean_)
    billBooking_id: Mapped_[int] = mapped_column(ForeignKey_("booking.id"), unique=True)

    def registerPayment(self):
        """Register payment for this bill, transitioning the associated Booking to CONFIRMED if not already settled."""
        if self.settled:
            return False  # Already settled, no change
        self.settled = True
        # Update the associated Booking's commercialStatus to CONFIRMED
        if self.billBooking:
            self.billBooking.commercialStatus = BookingCommercialStatus.CONFIRMED
        return True


class BookedRoom(Base):
    __tablename__ = "bookedroom"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    agreedPrice: Mapped_[float] = mapped_column(Float_)
    createdAt: Mapped_[dt_datetime] = mapped_column(DateTime_, default=dt_datetime.now)
    booking_id: Mapped_[int] = mapped_column(ForeignKey_("booking.id"))
    booking_1_id: Mapped_[int] = mapped_column(ForeignKey_("booking.id"))
    room_id: Mapped_[int] = mapped_column(ForeignKey_("room.id"))
    extraCharges: Mapped_[float] = mapped_column(Float_, default=0.0)  # Additional charges per room

    def get_total_cost(self, days: int) -> float:
        """Calculate total cost for this booked room over the given number of days."""
        return self.agreedPrice * days + self.extraCharges

class Booking(Base):
    __tablename__ = "booking"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    bookingNumber: Mapped_[int] = mapped_column(Integer_)
    arrivalDate: Mapped_[dt_date] = mapped_column(Date_)
    departureDate: Mapped_[dt_date] = mapped_column(Date_)
    commercialStatus: Mapped_[BookingCommercialStatus] = mapped_column(Enum(BookingCommercialStatus))
    physicalStatus: Mapped_[BookingPhysicalStatus] = mapped_column(Enum(BookingPhysicalStatus))
    totalPrice: Mapped_[float] = mapped_column(Float_)
    contact_id: Mapped_[int] = mapped_column(ForeignKey_("person.id"))
    employee_id: Mapped_[int] = mapped_column(ForeignKey_("employee.id"))

class Room(Base):
    __tablename__ = "room"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    roomNumber: Mapped_[str] = mapped_column(String_(100))
    capacity: Mapped_[int] = mapped_column(Integer_)
    description: Mapped_[str] = mapped_column(String_(100))
    standardPrice: Mapped_[float] = mapped_column(Float_)

class Person(Base):
    __tablename__ = "person"
    id: Mapped_[int] = mapped_column(Integer_, primary_key=True)
    firstName: Mapped_[str] = mapped_column(String_(100))
    familyName: Mapped_[str] = mapped_column(String_(100))
    phone: Mapped_[str] = mapped_column(String_(100))
    email: Mapped_[str] = mapped_column(String_(100))
    type_spec: Mapped_[str] = mapped_column(String_(50))
    __mapper_args__ = {
        "polymorphic_identity": "person",
        "polymorphic_on": "type_spec",
    }

class Employee(Person):
    __tablename__ = "employee"
    id: Mapped_[int] = mapped_column(ForeignKey_("person.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "employee",
    }

class Guest(Person):
    __tablename__ = "guest"
    id: Mapped_[int] = mapped_column(ForeignKey_("person.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "guest",
    }


#--- Relationships of the bill table

Bill.billBooking: Mapped_["Booking"] = relationship("Booking", back_populates="bill", uselist=False, foreign_keys=[Bill.billBooking_id])

#--- Relationships of the bookedroom table

BookedRoom.booking: Mapped_["Booking"] = relationship("Booking", back_populates="bookedRooms", uselist=False, foreign_keys=[BookedRoom.booking_id])

BookedRoom.booking_1: Mapped_["Booking"] = relationship("Booking", back_populates="bookedroom", uselist=False, foreign_keys=[BookedRoom.booking_1_id])

BookedRoom.room: Mapped_["Room"] = relationship("Room", back_populates="bookedroom", uselist=False, foreign_keys=[BookedRoom.room_id])

#--- Relationships of the booking table
Booking.guest: Mapped_[List_["Guest"]] = relationship("Guest", secondary=guests, back_populates="guests")
Booking.bookedRooms: Mapped_[List_["BookedRoom"]] = relationship("BookedRoom", back_populates="booking", foreign_keys=[BookedRoom.booking_id])

Booking.bill: Mapped_["Bill"] = relationship("Bill", back_populates="billBooking", uselist=False, foreign_keys=[Bill.billBooking_id])

Booking.contact: Mapped_["Person"] = relationship("Person", back_populates="booking", uselist=False, foreign_keys=[Booking.contact_id])

Booking.employee: Mapped_["Employee"] = relationship("Employee", back_populates="handledBy", uselist=False, foreign_keys=[Booking.employee_id])
Booking.bookedroom: Mapped_[List_["BookedRoom"]] = relationship("BookedRoom", back_populates="booking_1", foreign_keys=[BookedRoom.booking_1_id])

#--- Relationships of the room table
Room.bookedroom: Mapped_[List_["BookedRoom"]] = relationship("BookedRoom", back_populates="room", foreign_keys=[BookedRoom.room_id])

#--- Relationships of the person table
Person.booking: Mapped_[List_["Booking"]] = relationship("Booking", back_populates="contact", foreign_keys=[Booking.contact_id])

#--- Relationships of the employee table
Employee.handledBy: Mapped_[List_["Booking"]] = relationship("Booking", back_populates="employee", foreign_keys=[Booking.employee_id])

#--- Relationships of the guest table
Guest.guests: Mapped_[List_["Booking"]] = relationship("Booking", secondary=guests, back_populates="guest")

# Database connection (override the default with the DATABASE_URL environment variable)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/Class_Diagram.db")  # SQLite connection
engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    # Create tables in the database only when this module is executed directly,
    # so importing it never touches the database as a side effect.
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs("data", exist_ok=True)  # folder for the default SQLite database
    Base.metadata.create_all(engine, checkfirst=True)