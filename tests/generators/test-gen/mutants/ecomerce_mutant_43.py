# Mutant Operator: Incorrect Return Value
from datetime import datetime, date, time
from enum import Enum

class OrderStatus(Enum):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    SHIPPED = 'SHIPPED'
    DELIVERED = 'DELIVERED'
    CANCELLED = 'CANCELLED'

class Order:

    def __init__(self, orderId: str, orderDate: str, totalAmount: float, status: str, customer: 'Customer'=None, products: set['Product']=None):
        self.orderId = orderId
        self.orderDate = orderDate
        self.totalAmount = totalAmount
        self.status = status
        self.customer = customer
        self.products = products if products is not None else set()

    @property
    def orderDate(self) -> str:
        return self.__orderDate

    @orderDate.setter
    def orderDate(self, orderDate: str):
        self.__orderDate = orderDate

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
    def totalAmount(self) -> float:
        return self.__totalAmount

    @totalAmount.setter
    def totalAmount(self, totalAmount: float):
        self.__totalAmount = totalAmount

    @property
    def customer(self):
        return self.__customer

    @customer.setter
    def customer(self, value):
        old_value = getattr(self, f'_Order__customer', None)
        self.__customer = value
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

    @property
    def products(self):
        return None

    @products.setter
    def products(self, value):
        old_value = getattr(self, f'_Order__products', None)
        self.__products = value if value is not None else set()
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

    def placeOrder(self):
        self.status = 'Pending'
        print(f'Order {self.orderId} placed on {self.orderDate}')
        print(f'Total amount: ${self.totalAmount:.2f}')

    def calculateTax(self, taxRate):
        tax = self.totalAmount * taxRate
        print(f'Tax for order {self.orderId}: ${tax:.2f}')
        return tax

    def updateStatus(self, newStatus):
        old_status = self.status
        self.status = newStatus
        print(f'Order {self.orderId} status updated: {old_status} -> {newStatus}')

class Product:

    def __init__(self, productId: str, name: str, price: float, stock: int, orders: set['Order']=None):
        self.productId = productId
        self.name = name
        self.price = price
        self.stock = stock
        self.orders = orders if orders is not None else set()

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def productId(self) -> str:
        return self.__productId

    @productId.setter
    def productId(self, productId: str):
        self.__productId = productId

    @property
    def stock(self) -> int:
        return self.__stock

    @stock.setter
    def stock(self, stock: int):
        self.__stock = stock

    @property
    def price(self) -> float:
        return self.__price

    @price.setter
    def price(self, price: float):
        self.__price = price

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, value):
        old_value = getattr(self, f'_Product__orders', None)
        self.__orders = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'products'):
                    opp_val = getattr(item, 'products', None)
                    if isinstance(opp_val, set):
                        opp_val.discard(self)
        if value is not None:
            for item in value:
                if hasattr(item, 'products'):
                    opp_val = getattr(item, 'products', None)
                    if opp_val is None:
                        setattr(item, 'products', set([self]))
                    elif isinstance(opp_val, set):
                        opp_val.add(self)

    def checkStock(self, quantity):
        if self.stock >= quantity:
            print(f"Product '{self.name}' has sufficient stock")
            return True
        else:
            print(f"Insufficient stock for '{self.name}'. Available: {self.stock}")
            return False

    def getTotalValue(self):
        total = self.price * self.stock
        print(f"Total inventory value for '{self.name}': ${total:.2f}")
        return total

    def reduceStock(self, quantity):
        if self.stock >= quantity:
            self.stock -= quantity
            print(f'Stock reduced by {quantity}. New stock: {self.stock}')
        else:
            print(f'Cannot reduce stock. Current stock: {self.stock}')

class Customer:

    def __init__(self, customerId: str, username: str, password: str, address: str, orders: set['Order']=None):
        self.customerId = customerId
        self.username = username
        self.password = password
        self.address = address
        self.orders = orders if orders is not None else set()

    @property
    def password(self) -> str:
        return self.__password

    @password.setter
    def password(self, password: str):
        self.__password = password

    @property
    def customerId(self) -> str:
        return self.__customerId

    @customerId.setter
    def customerId(self, customerId: str):
        self.__customerId = customerId

    @property
    def address(self) -> str:
        return self.__address

    @address.setter
    def address(self, address: str):
        self.__address = address

    @property
    def username(self) -> str:
        return self.__username

    @username.setter
    def username(self, username: str):
        self.__username = username

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, value):
        old_value = getattr(self, f'_Customer__orders', None)
        self.__orders = value if value is not None else set()
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

    def getProfile(self):
        return f'Customer: {self.username} (ID: {self.customerId})\nAddress: {self.address}'

    def updateAddress(self, newAddress):
        old_address = self.address
        self.address = newAddress
        print(f"Address updated from '{old_address}' to '{newAddress}'")

    def login(self, inputPassword):
        if self.password == inputPassword:
            print(f'Customer {self.username} logged in successfully')
            return True
        else:
            print('Invalid password')
            return False