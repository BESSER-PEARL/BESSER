
from datetime import datetime, date, time


class Book:
    def __init__(self, pages: int, release: datetime, title: str, writtenBy: set["Author"], locatedIn: "Library"):
        self.pages = pages
        self.release = release
        self.title = title
        self.writtenBy = writtenBy 
        self.locatedIn = locatedIn 
        
    @property
    def pages(self) -> int:
        return self.__pages
    
    @pages.setter
    def pages(self, pages: int):
        self.__pages = pages
    
    @property
    def release(self) -> datetime:
        return self.__release
    
    @release.setter
    def release(self, release: datetime):
        self.__release = release
    
    @property
    def title(self) -> str:
        return self.__title
    
    @title.setter
    def title(self, title: str):
        self.__title = title
    
    @property
    def writtenBy(self):
        return self.__writtenBy
    
    @writtenBy.setter
    def writtenBy(self, writtenBy):
        self.__writtenBy = writtenBy
    
    @property
    def locatedIn(self):
        return self.__locatedIn
    
    @locatedIn.setter
    def locatedIn(self, locatedIn):
        self.__locatedIn = locatedIn
    
class Author:
    def __init__(self, name: str, email: str, publishes: set["Book"]):
        self.name = name
        self.email = email
        self.publishes = publishes 
        
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
    def publishes(self):
        return self.__publishes
    
    @publishes.setter
    def publishes(self, publishes):
        self.__publishes = publishes
    
class Library:
    def __init__(self, address: str, name: str, has: set["Book"]):
        self.address = address
        self.name = name
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
    