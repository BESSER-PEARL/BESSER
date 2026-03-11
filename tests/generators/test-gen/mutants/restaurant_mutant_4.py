# Mutant Operator: Removed Assignment
from datetime import datetime, date, time
from enum import Enum

class MenuCategory(Enum):
    APPETIZER = 'APPETIZER'
    MAIN_COURSE = 'MAIN_COURSE'
    pass
    BEVERAGE = 'BEVERAGE'

class DineOrder:

    def __init__(self, orderId: str, orderTime: str, totalAmount: float, status: str, table: 'Table'=None, items: set['MenuItem']=None):
        self.orderId = orderId
        self.orderTime = orderTime
        self.totalAmount = totalAmount
        self.status = status
        self.table = table
        self.items = items if items is not None else set()

    @property
    def totalAmount(self) -> float:
        return self.__totalAmount

    @totalAmount.setter
    def totalAmount(self, totalAmount: float):
        self.__totalAmount = totalAmount

    @property
    def orderTime(self) -> str:
        return self.__orderTime

    @orderTime.setter
    def orderTime(self, orderTime: str):
        self.__orderTime = orderTime

    @property
    def status(self) -> str:
        return self.__status

    @status.setter
    def status(self, status: str):
        self.__status = status

    @property
    def orderId(self) -> str:
        return self.__orderId

    @orderId.setter
    def orderId(self, orderId: str):
        self.__orderId = orderId

    @property
    def items(self):
        return self.__items

    @items.setter
    def items(self, value):
        old_value = getattr(self, f'_DineOrder__items', None)
        self.__items = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'orders'):
                    opp_val = getattr(item, 'orders', None)
                    if isinstance(opp_val, set):
                        opp_val.discard(self)
        if value is not None:
            for item in value:
                if hasattr(item, 'orders'):
                    opp_val = getattr(item, 'orders', None)
                    if opp_val is None:
                        setattr(item, 'orders', set([self]))
                    elif isinstance(opp_val, set):
                        opp_val.add(self)

    @property
    def table(self):
        return self.__table

    @table.setter
    def table(self, value):
        old_value = getattr(self, f'_DineOrder__table', None)
        self.__table = value
        if old_value is not None:
            if hasattr(old_value, 'orders'):
                opp_val = getattr(old_value, 'orders', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'orders'):
                opp_val = getattr(value, 'orders', None)
                if opp_val is None:
                    setattr(value, 'orders', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def calculateTip(self, tipPercentage):
        tip = self.totalAmount * tipPercentage
        print(f'Suggested tip ({tipPercentage * 100:.0f}%): ${tip:.2f}')
        return tip

    def placeOrder(self):
        self.status = 'Pending'
        print(f'Order {self.orderId} placed at {self.orderTime}')
        print(f'Total amount: ${self.totalAmount:.2f}')

    def updateStatus(self, newStatus):
        old_status = self.status
        self.status = newStatus
        print(f'Order {self.orderId} status: {old_status} -> {newStatus}')

class Table:

    def __init__(self, tableNumber: str, capacity: int, occupied: bool, location: str, orders: set['DineOrder']=None):
        self.tableNumber = tableNumber
        self.capacity = capacity
        self.occupied = occupied
        self.location = location
        self.orders = orders if orders is not None else set()

    @property
    def tableNumber(self) -> str:
        return self.__tableNumber

    @tableNumber.setter
    def tableNumber(self, tableNumber: str):
        self.__tableNumber = tableNumber

    @property
    def location(self) -> str:
        return self.__location

    @location.setter
    def location(self, location: str):
        self.__location = location

    @property
    def occupied(self) -> bool:
        return self.__occupied

    @occupied.setter
    def occupied(self, occupied: bool):
        self.__occupied = occupied

    @property
    def capacity(self) -> int:
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity: int):
        self.__capacity = capacity

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, value):
        old_value = getattr(self, f'_Table__orders', None)
        self.__orders = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'table'):
                    opp_val = getattr(item, 'table', None)
                    if opp_val == self:
                        setattr(item, 'table', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'table'):
                    opp_val = getattr(item, 'table', None)
                    setattr(item, 'table', self)

    def releaseTable(self):
        self.occupied = False
        print(f'Table {self.tableNumber} is now available')

    def reserveTable(self):
        if not self.occupied:
            self.occupied = True
            print(f'Table {self.tableNumber} reserved')
            print(f'Capacity: {self.capacity}, Location: {self.location}')
        else:
            print(f'Table {self.tableNumber} is already occupied')

    def canAccommodate(self, partySize):
        if partySize <= self.capacity and (not self.occupied):
            print(f'Table {self.tableNumber} can accommodate {partySize} guests')
            return True
        else:
            print(f'Table {self.tableNumber} cannot accommodate the party')
            return False

class MenuItem:

    def __init__(self, itemId: str, name: str, price: float, available: bool, orders: set['DineOrder']=None):
        self.itemId = itemId
        self.name = name
        self.price = price
        self.available = available
        self.orders = orders if orders is not None else set()

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def itemId(self) -> str:
        return self.__itemId

    @itemId.setter
    def itemId(self, itemId: str):
        self.__itemId = itemId

    @property
    def price(self) -> float:
        return self.__price

    @price.setter
    def price(self, price: float):
        self.__price = price

    @property
    def available(self) -> bool:
        return self.__available

    @available.setter
    def available(self, available: bool):
        self.__available = available

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, value):
        old_value = getattr(self, f'_MenuItem__orders', None)
        self.__orders = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'items'):
                    opp_val = getattr(item, 'items', None)
                    if isinstance(opp_val, set):
                        opp_val.discard(self)
        if value is not None:
            for item in value:
                if hasattr(item, 'items'):
                    opp_val = getattr(item, 'items', None)
                    if opp_val is None:
                        setattr(item, 'items', set([self]))
                    elif isinstance(opp_val, set):
                        opp_val.add(self)

    def setAvailability(self, status):
        self.available = status
        status_text = 'available' if status else 'sold out'
        print(f'{self.name} is now {status_text}')

    def updatePrice(self, newPrice):
        old_price = self.price
        self.price = newPrice
        print(f'Price updated for {self.name}: ${old_price:.2f} -> ${newPrice:.2f}')

    def addToMenu(self):
        self.available = True
        print(f'Menu item added: {self.name}')
        print(f'ID: {self.itemId}, Price: ${self.price:.2f}')