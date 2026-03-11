# Mutant Operator: Removed Assignment
from datetime import datetime, date, time
from enum import Enum

class BookStatus(Enum):
    AVAILABLE = 'AVAILABLE'
    CHECKED_OUT = 'CHECKED_OUT'
    RESERVED = 'RESERVED'

class Loan:

    def __init__(self, dueDate: str, returned: bool, loanId: str, loanDate: str, member: 'Member'=None, book: 'Book'=None):
        self.dueDate = dueDate
        self.returned = returned
        pass
        self.loanDate = loanDate
        self.member = member
        self.book = book

    @property
    def dueDate(self) -> str:
        return self.__dueDate

    @dueDate.setter
    def dueDate(self, dueDate: str):
        self.__dueDate = dueDate

    @property
    def returned(self) -> bool:
        return self.__returned

    @returned.setter
    def returned(self, returned: bool):
        self.__returned = returned

    @property
    def loanId(self) -> str:
        return self.__loanId

    @loanId.setter
    def loanId(self, loanId: str):
        self.__loanId = loanId

    @property
    def loanDate(self) -> str:
        return self.__loanDate

    @loanDate.setter
    def loanDate(self, loanDate: str):
        self.__loanDate = loanDate

    @property
    def book(self):
        return self.__book

    @book.setter
    def book(self, value):
        old_value = getattr(self, f'_Loan__book', None)
        self.__book = value
        if old_value is not None:
            if hasattr(old_value, 'loanHistory'):
                opp_val = getattr(old_value, 'loanHistory', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'loanHistory'):
                opp_val = getattr(value, 'loanHistory', None)
                if opp_val is None:
                    setattr(value, 'loanHistory', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def member(self):
        return self.__member

    @member.setter
    def member(self, value):
        old_value = getattr(self, f'_Loan__member', None)
        self.__member = value
        if old_value is not None:
            if hasattr(old_value, 'loans'):
                opp_val = getattr(old_value, 'loans', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'loans'):
                opp_val = getattr(value, 'loans', None)
                if opp_val is None:
                    setattr(value, 'loans', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def returnBook(self):
        self.returned = True
        print(f'Loan {self.loanId} marked as returned')

    def createLoan(self):
        self.returned = False
        print(f'Loan {self.loanId} created on {self.loanDate}')
        print(f'Due date: {self.dueDate}')

    def isOverdue(self, currentDate):
        if not self.returned and currentDate > self.dueDate:
            print(f'Loan {self.loanId} is overdue!')
            return True
        return False

class Member:

    def __init__(self, memberId: str, name: str, email: str, maxBooks: int, loans: set['Loan']=None):
        self.memberId = memberId
        self.name = name
        self.email = email
        self.maxBooks = maxBooks
        self.loans = loans if loans is not None else set()

    @property
    def memberId(self) -> str:
        return self.__memberId

    @memberId.setter
    def memberId(self, memberId: str):
        self.__memberId = memberId

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def email(self) -> str:
        return self.__email

    @email.setter
    def email(self, email: str):
        self.__email = email

    @property
    def maxBooks(self) -> int:
        return self.__maxBooks

    @maxBooks.setter
    def maxBooks(self, maxBooks: int):
        self.__maxBooks = maxBooks

    @property
    def loans(self):
        return self.__loans

    @loans.setter
    def loans(self, value):
        old_value = getattr(self, f'_Member__loans', None)
        self.__loans = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'member'):
                    opp_val = getattr(item, 'member', None)
                    if opp_val == self:
                        setattr(item, 'member', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'member'):
                    opp_val = getattr(item, 'member', None)
                    setattr(item, 'member', self)

    def register(self):
        print(f'Member {self.name} registered with ID: {self.memberId}')
        print(f'Email: {self.email}')

    def updateEmail(self, newEmail):
        self.email = newEmail
        print(f'Email updated to {newEmail} for member {self.name}')

    def canBorrow(self, currentLoans):
        if currentLoans < self.maxBooks:
            return True
        else:
            print(f'Member {self.name} has reached maximum loan limit')
            return False

class Book:

    def __init__(self, isbn: str, title: str, author: str, available: bool, loanHistory: set['Loan']=None):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.available = available
        self.loanHistory = loanHistory if loanHistory is not None else set()

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str):
        self.__title = title

    @property
    def isbn(self) -> str:
        return self.__isbn

    @isbn.setter
    def isbn(self, isbn: str):
        self.__isbn = isbn

    @property
    def available(self) -> bool:
        return self.__available

    @available.setter
    def available(self, available: bool):
        self.__available = available

    @property
    def author(self) -> str:
        return self.__author

    @author.setter
    def author(self, author: str):
        self.__author = author

    @property
    def loanHistory(self):
        return self.__loanHistory

    @loanHistory.setter
    def loanHistory(self, value):
        old_value = getattr(self, f'_Book__loanHistory', None)
        self.__loanHistory = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'book'):
                    opp_val = getattr(item, 'book', None)
                    if opp_val == self:
                        setattr(item, 'book', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'book'):
                    opp_val = getattr(item, 'book', None)
                    setattr(item, 'book', self)

    def getInfo(self):
        return f'{self.title} by {self.author} (ISBN: {self.isbn})'

    def checkAvailability(self):
        return self.available

    def setAvailability(self, status):
        self.available = status
        print(f"Book '{self.title}' availability set to {status}")