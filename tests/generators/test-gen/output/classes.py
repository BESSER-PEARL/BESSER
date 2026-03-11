from datetime import datetime, date, time
from enum import Enum

############################################
# Definition of Enumerations
############################################

class Genre(Enum):
    FICTION = "FICTION"
    NONFICTION = "NONFICTION"
    SCIENCE = "SCIENCE"


############################################
# Definition of Classes
############################################

class Author:

    def __init__(self, name: str, email: str, books: set["Book"] = None):
        self.name = name
        self.email = email
        self.books = books if books is not None else set()
        
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
    def books(self):
        return self.__books

    @books.setter
    def books(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Author__books", None)
        self.__books = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "author"):
                    opp_val = getattr(item, "author", None)
                    
                    if opp_val == self:
                        setattr(item, "author", None)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "author"):
                    opp_val = getattr(item, "author", None)
                    
                    setattr(item, "author", self)
                    

class Book:

    def __init__(self, title: str, pages: int, rating: float, inPrint: bool, author: "Author" = None):
        self.title = title
        self.pages = pages
        self.rating = rating
        self.inPrint = inPrint
        self.author = author
        
    @property
    def rating(self) -> float:
        return self.__rating

    @rating.setter
    def rating(self, rating: float):
        self.__rating = rating

    @property
    def inPrint(self) -> bool:
        return self.__inPrint

    @inPrint.setter
    def inPrint(self, inPrint: bool):
        self.__inPrint = inPrint

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
    def author(self):
        return self.__author

    @author.setter
    def author(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Book__author", None)
        self.__author = value
        
        # Remove self from old opposite end
        if old_value is not None:
            if hasattr(old_value, "books"):
                opp_val = getattr(old_value, "books", None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
                
        # Add self to new opposite end
        if value is not None:
            if hasattr(value, "books"):
                opp_val = getattr(value, "books", None)
                if opp_val is None:
                    setattr(value, "books", set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def getSummary(self, maxLength: int):
        # TODO: Implement getSummary method
        pass

    def isAvailable(self):
        # TODO: Implement isAvailable method
        pass
