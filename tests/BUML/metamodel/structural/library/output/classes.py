from datetime import datetime, date, time

############################################
# Definition of Classes
############################################

class Author:

    def __init__(self, name: str, email: str, publishes: set["Book"]):
        self.name = name
        self.email = email
        self.publishes = publishes
        
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
    def publishes(self):
        return self.__publishes

    @publishes.setter
    def publishes(self, publishes):
        self.__publishes = publishes


class Book:

    def __init__(self, title: str, pages: int, release: date, locatedIn: "Library", writtenBy: set["Author"]):
        self.title = title
        self.pages = pages
        self.release = release
        self.locatedIn = locatedIn
        self.writtenBy = writtenBy
        
    @property
    def pages(self) -> int:
        return self.__pages

    @pages.setter
    def pages(self, pages: int):
        self.__pages = pages

    @property
    def release(self) -> date:
        return self.__release

    @release.setter
    def release(self, release: date):
        self.__release = release

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str):
        self.__title = title

    @property
    def locatedIn(self):
        return self.__locatedIn

    @locatedIn.setter
    def locatedIn(self, locatedIn):
        self.__locatedIn = locatedIn

    @property
    def writtenBy(self):
        return self.__writtenBy

    @writtenBy.setter
    def writtenBy(self, writtenBy):
        self.__writtenBy = writtenBy


class Library:

    def __init__(self, name: str, address: str, has: set["Book"]):
        self.name = name
        self.address = address
        self.has = has
        
    @property
    def address(self) -> str:
        return self.__address

    @address.setter
    def address(self, address: str):
        self.__address = address

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def has(self):
        return self.__has

    @has.setter
    def has(self, has):
        self.__has = has

