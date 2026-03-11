# Mutant Operator: Updated Operation Signature: loyaltyPoints
from datetime import datetime, date, time
from enum import Enum

class RoomType(Enum):
    SINGLE = 'SINGLE'
    DOUBLE = 'DOUBLE'
    SUITE = 'SUITE'
    DELUXE = 'DELUXE'

class Reservation:

    def __init__(self, reservationId: str, checkInDate: str, checkOutDate: str, totalCost: float, guest: 'Guest'=None, room: 'Room'=None):
        self.reservationId = reservationId
        self.checkInDate = checkInDate
        self.checkOutDate = checkOutDate
        self.totalCost = totalCost
        self.guest = guest
        self.room = room

    @property
    def reservationId(self) -> str:
        return self.__reservationId

    @reservationId.setter
    def reservationId(self, reservationId: str):
        self.__reservationId = reservationId

    @property
    def totalCost(self) -> float:
        return self.__totalCost

    @totalCost.setter
    def totalCost(self, totalCost: float):
        self.__totalCost = totalCost

    @property
    def checkInDate(self) -> str:
        return self.__checkInDate

    @checkInDate.setter
    def checkInDate(self, checkInDate: str):
        self.__checkInDate = checkInDate

    @property
    def checkOutDate(self) -> str:
        return self.__checkOutDate

    @checkOutDate.setter
    def checkOutDate(self, checkOutDate: str):
        self.__checkOutDate = checkOutDate

    @property
    def room(self):
        return self.__room

    @room.setter
    def room(self, value):
        old_value = getattr(self, f'_Reservation__room', None)
        self.__room = value
        if old_value is not None:
            if hasattr(old_value, 'bookings'):
                opp_val = getattr(old_value, 'bookings', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'bookings'):
                opp_val = getattr(value, 'bookings', None)
                if opp_val is None:
                    setattr(value, 'bookings', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def guest(self):
        return self.__guest

    @guest.setter
    def guest(self, value):
        old_value = getattr(self, f'_Reservation__guest', None)
        self.__guest = value
        if old_value is not None:
            if hasattr(old_value, 'reservations'):
                opp_val = getattr(old_value, 'reservations', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'reservations'):
                opp_val = getattr(value, 'reservations', None)
                if opp_val is None:
                    setattr(value, 'reservations', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def createReservation(self):
        print(f'Reservation {self.reservationId} created')
        print(f'Check-in: {self.checkInDate}, Check-out: {self.checkOutDate}')
        print(f'Total cost: ${self.totalCost:.2f}')

    def calculateNights(self):
        nights = 3
        print(f'Number of nights: {nights}')
        return nights

    def cancelReservation(self):
        print(f'Reservation {self.reservationId} has been cancelled')
        print(f'Refund amount: ${self.totalCost:.2f}')

class Room:

    def __init__(self, roomNumber: str, roomType: str, pricePerNight: float, available: bool, bookings: set['Reservation']=None):
        self.roomNumber = roomNumber
        self.roomType = roomType
        self.pricePerNight = pricePerNight
        self.available = available
        self.bookings = bookings if bookings is not None else set()

    @property
    def pricePerNight(self) -> float:
        return self.__pricePerNight

    @pricePerNight.setter
    def pricePerNight(self, pricePerNight: float):
        self.__pricePerNight = pricePerNight

    @property
    def roomNumber(self) -> str:
        return self.__roomNumber

    @roomNumber.setter
    def roomNumber(self, roomNumber: str):
        self.__roomNumber = roomNumber

    @property
    def available(self) -> bool:
        return self.__available

    @available.setter
    def available(self, available: bool):
        self.__available = available

    @property
    def roomType(self) -> str:
        return self.__roomType

    @roomType.setter
    def roomType(self, roomType: str):
        self.__roomType = roomType

    @property
    def bookings(self):
        return self.__bookings

    @bookings.setter
    def bookings(self, value):
        old_value = getattr(self, f'_Room__bookings', None)
        self.__bookings = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'room'):
                    opp_val = getattr(item, 'room', None)
                    if opp_val == self:
                        setattr(item, 'room', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'room'):
                    opp_val = getattr(item, 'room', None)
                    setattr(item, 'room', self)

    def calculateCost(self, nights):
        total = self.pricePerNight * nights
        print(f'Cost for {nights} night(s) in room {self.roomNumber}: ${total:.2f}')
        return total

    def getRoomDetails(self):
        status = 'Available' if self.available else 'Occupied'
        return f'Room {self.roomNumber} - {self.roomType} (${self.pricePerNight}/night) - {status}'

    def setAvailable(self, status):
        self.available = status
        status_text = 'available' if status else 'occupied'
        print(f'Room {self.roomNumber} is now {status_text}')

class Guest:

    def __init__(self, guestId: str, name: str, email: str, loyaltyPoints: int, reservations: set['Reservation']=None):
        self.guestId = guestId
        self.name = name
        self.email = email
        self.loyaltyPoints = loyaltyPoints
        self.reservations = reservations if reservations is not None else set()

    @property
    def loyaltyPoints(self) -> int:
        return self.__loyaltyPoints

    @loyaltyPoints.setter
    def loyaltyPoints(self_m, loyaltyPoints_m: int):
        self.__loyaltyPoints = loyaltyPoints

    @property
    def email(self) -> str:
        return self.__email

    @email.setter
    def email(self, email: str):
        self.__email = email

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def guestId(self) -> str:
        return self.__guestId

    @guestId.setter
    def guestId(self, guestId: str):
        self.__guestId = guestId

    @property
    def reservations(self):
        return self.__reservations

    @reservations.setter
    def reservations(self, value):
        old_value = getattr(self, f'_Guest__reservations', None)
        self.__reservations = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'guest'):
                    opp_val = getattr(item, 'guest', None)
                    if opp_val == self:
                        setattr(item, 'guest', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'guest'):
                    opp_val = getattr(item, 'guest', None)
                    setattr(item, 'guest', self)

    def addLoyaltyPoints(self, points):
        self.loyaltyPoints += points
        print(f'{points} loyalty points added to {self.name}')
        print(f'Total loyalty points: {self.loyaltyPoints}')

    def getGuestInfo(self):
        return f'Guest: {self.name} (ID: {self.guestId})\nLoyalty Points: {self.loyaltyPoints}'

    def checkIn(self):
        print(f'Guest {self.name} checked in')
        print(f'ID: {self.guestId}, Email: {self.email}')