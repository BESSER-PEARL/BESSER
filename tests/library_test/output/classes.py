
from datetime import datetime, date, time


class Book:
    def __init__(self, release: datetime, title: str, pages: int, writedBy: set["Author"], locatedIn: "Library"):
        self.release = release
        self.title = title
        self.pages = pages
        self.writedBy = writedBy 
        self.locatedIn = locatedIn 
        
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
    def pages(self) -> int:
        return self.__pages
    
    @pages.setter
    def pages(self, pages: int):
        self.__pages = pages
    
    @property
    def writedBy(self):
        return self.__writedBy
    
    @writedBy.setter
    def writedBy(self, writedBy):
        self.__writedBy = writedBy
    
    @property
    def locatedIn(self):
        return self.__locatedIn
    
    @locatedIn.setter
    def locatedIn(self, locatedIn):
        self.__locatedIn = locatedIn
    
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
    
class Author:
    def __init__(self, email: str, name: str, publishes: set["Book"]):
        self.email = email
        self.name = name
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
    