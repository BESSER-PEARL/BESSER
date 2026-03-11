# Mutant Operator: Removed Assignment
from datetime import datetime, date, time
from enum import Enum

class VehicleCategory(Enum):
    pass
    SUV = 'SUV'
    TRUCK = 'TRUCK'
    SPORTS = 'SPORTS'

class Rental:

    def __init__(self, rentalId: str, startDate: str, endDate: str, totalCost: float, status: str, customer: 'RentalCustomer'=None, vehicle: 'Vehicle'=None):
        self.rentalId = rentalId
        self.startDate = startDate
        self.endDate = endDate
        self.totalCost = totalCost
        self.status = status
        self.customer = customer
        self.vehicle = vehicle

    @property
    def endDate(self) -> str:
        return self.__endDate

    @endDate.setter
    def endDate(self, endDate: str):
        self.__endDate = endDate

    @property
    def rentalId(self) -> str:
        return self.__rentalId

    @rentalId.setter
    def rentalId(self, rentalId: str):
        self.__rentalId = rentalId

    @property
    def totalCost(self) -> float:
        return self.__totalCost

    @totalCost.setter
    def totalCost(self, totalCost: float):
        self.__totalCost = totalCost

    @property
    def startDate(self) -> str:
        return self.__startDate

    @startDate.setter
    def startDate(self, startDate: str):
        self.__startDate = startDate

    @property
    def status(self) -> str:
        return self.__status

    @status.setter
    def status(self, status: str):
        self.__status = status

    @property
    def vehicle(self):
        return self.__vehicle

    @vehicle.setter
    def vehicle(self, value):
        old_value = getattr(self, f'_Rental__vehicle', None)
        self.__vehicle = value
        if old_value is not None:
            if hasattr(old_value, 'rentalHistory'):
                opp_val = getattr(old_value, 'rentalHistory', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'rentalHistory'):
                opp_val = getattr(value, 'rentalHistory', None)
                if opp_val is None:
                    setattr(value, 'rentalHistory', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def customer(self):
        return self.__customer

    @customer.setter
    def customer(self, value):
        old_value = getattr(self, f'_Rental__customer', None)
        self.__customer = value
        if old_value is not None:
            if hasattr(old_value, 'rentals'):
                opp_val = getattr(old_value, 'rentals', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'rentals'):
                opp_val = getattr(value, 'rentals', None)
                if opp_val is None:
                    setattr(value, 'rentals', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def completeRental(self):
        self.status = 'Completed'
        print(f'Rental {self.rentalId} has been completed')

    def startRental(self):
        self.status = 'Active'
        print(f'Rental {self.rentalId} started on {self.startDate}')
        print(f'End date: {self.endDate}, Total cost: ${self.totalCost:.2f}')

    def extendRental(self, newEndDate, additionalCost):
        self.endDate = newEndDate
        self.totalCost += additionalCost
        print(f'Rental {self.rentalId} extended to {newEndDate}')
        print(f'New total cost: ${self.totalCost:.2f}')

class RentalCustomer:

    def __init__(self, customerId: str, name: str, licenseNumber: str, verified: bool, rentals: set['Rental']=None):
        self.customerId = customerId
        self.name = name
        self.licenseNumber = licenseNumber
        self.verified = verified
        self.rentals = rentals if rentals is not None else set()

    @property
    def customerId(self) -> str:
        return self.__customerId

    @customerId.setter
    def customerId(self, customerId: str):
        self.__customerId = customerId

    @property
    def licenseNumber(self) -> str:
        return self.__licenseNumber

    @licenseNumber.setter
    def licenseNumber(self, licenseNumber: str):
        self.__licenseNumber = licenseNumber

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def verified(self) -> bool:
        return self.__verified

    @verified.setter
    def verified(self, verified: bool):
        self.__verified = verified

    @property
    def rentals(self):
        return self.__rentals

    @rentals.setter
    def rentals(self, value):
        old_value = getattr(self, f'_RentalCustomer__rentals', None)
        self.__rentals = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'customer'):
                    opp_val = getattr(item, 'customer', None)
                    if opp_val == self:
                        setattr(item, 'customer', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'customer'):
                    opp_val = getattr(item, 'customer', None)
                    setattr(item, 'customer', self)

    def verifyLicense(self):
        self.verified = True
        print(f'License verified for customer {self.name}')

    def registerCustomer(self):
        self.verified = False
        print(f'Customer registered: {self.name}')
        print(f'ID: {self.customerId}, License: {self.licenseNumber}')

    def getCustomerInfo(self):
        status = 'Verified' if self.verified else 'Not Verified'
        return f'Customer: {self.name} ({status})\nLicense: {self.licenseNumber}'

class Vehicle:

    def __init__(self, vehicleId: str, make: str, model: str, available: bool, dailyRate: float, rentalHistory: set['Rental']=None):
        self.vehicleId = vehicleId
        self.make = make
        self.model = model
        self.available = available
        self.dailyRate = dailyRate
        self.rentalHistory = rentalHistory if rentalHistory is not None else set()

    @property
    def model(self) -> str:
        return self.__model

    @model.setter
    def model(self, model: str):
        self.__model = model

    @property
    def make(self) -> str:
        return self.__make

    @make.setter
    def make(self, make: str):
        self.__make = make

    @property
    def vehicleId(self) -> str:
        return self.__vehicleId

    @vehicleId.setter
    def vehicleId(self, vehicleId: str):
        self.__vehicleId = vehicleId

    @property
    def available(self) -> bool:
        return self.__available

    @available.setter
    def available(self, available: bool):
        self.__available = available

    @property
    def dailyRate(self) -> float:
        return self.__dailyRate

    @dailyRate.setter
    def dailyRate(self, dailyRate: float):
        self.__dailyRate = dailyRate

    @property
    def rentalHistory(self):
        return self.__rentalHistory

    @rentalHistory.setter
    def rentalHistory(self, value):
        old_value = getattr(self, f'_Vehicle__rentalHistory', None)
        self.__rentalHistory = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'vehicle'):
                    opp_val = getattr(item, 'vehicle', None)
                    if opp_val == self:
                        setattr(item, 'vehicle', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'vehicle'):
                    opp_val = getattr(item, 'vehicle', None)
                    setattr(item, 'vehicle', self)

    def calculateRentalCost(self, days):
        total = self.dailyRate * days
        print(f'Rental cost for {days} day(s): ${total:.2f}')
        return total

    def setAvailability(self, status):
        self.available = status
        status_text = 'available' if status else 'rented'
        print(f'{self.make} {self.model} is now {status_text}')

    def registerVehicle(self):
        self.available = True
        print(f'Vehicle registered: {self.make} {self.model}')
        print(f'ID: {self.vehicleId}, Daily Rate: ${self.dailyRate:.2f}')