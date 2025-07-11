from datetime import datetime, date, time

############################################
# Definition of Classes
############################################

class Author:

    def __init__(self, name: str, email: str, publishes: set["Book"] = None):
        self.name = name
        self.email = email
        self.publishes = publishes if publishes is not None else set()
        
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
    def publishes(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Author__publishes", None)
        self.__publishes = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "writtenBy"):
                    opp_val = getattr(item, "writtenBy", None)
                    
                    if isinstance(opp_val, set):
                        opp_val.discard(self)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "writtenBy"):
                    opp_val = getattr(item, "writtenBy", None)
                    
                    if opp_val is None:
                        setattr(item, "writtenBy", set([self]))
                    elif isinstance(opp_val, set):
                        opp_val.add(self)
                    

class Book:

    def __init__(self, title: str, pages: int, release: date, locatedIn: "Library" = None, writtenBy: set["Author"] = None):
        self.title = title
        self.pages = pages
        self.release = release
        self.locatedIn = locatedIn
        self.writtenBy = writtenBy if writtenBy is not None else set()
        
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
    def release(self) -> date:
        return self.__release

    @release.setter
    def release(self, release: date):
        self.__release = release

    @property
    def writtenBy(self):
        return self.__writtenBy

    @writtenBy.setter
    def writtenBy(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Book__writtenBy", None)
        self.__writtenBy = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "publishes"):
                    opp_val = getattr(item, "publishes", None)
                    
                    if isinstance(opp_val, set):
                        opp_val.discard(self)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "publishes"):
                    opp_val = getattr(item, "publishes", None)
                    
                    if opp_val is None:
                        setattr(item, "publishes", set([self]))
                    elif isinstance(opp_val, set):
                        opp_val.add(self)
                    

    @property
    def locatedIn(self):
        return self.__locatedIn

    @locatedIn.setter
    def locatedIn(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Book__locatedIn", None)
        self.__locatedIn = value
        
        # Remove self from old opposite end
        if old_value is not None:
            if hasattr(old_value, "has"):
                opp_val = getattr(old_value, "has", None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
                
        # Add self to new opposite end
        if value is not None:
            if hasattr(value, "has"):
                opp_val = getattr(value, "has", None)
                if opp_val is None:
                    setattr(value, "has", set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

class Library:

    def __init__(self, name: str, address: str, has: set["Book"] = None):
        self.name = name
        self.address = address
        self.has = has if has is not None else set()
        
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
    def has(self, value):
        # Bidirectional consistency
        old_value = getattr(self, f"_Library__has", None)
        self.__has = value if value is not None else set()
        
        # Remove self from old opposite end
        if old_value is not None:
            for item in old_value:
                if hasattr(item, "locatedIn"):
                    opp_val = getattr(item, "locatedIn", None)
                    
                    if opp_val == self:
                        setattr(item, "locatedIn", None)
                    
        # Add self to new opposite end
        if value is not None:
            for item in value:
                if hasattr(item, "locatedIn"):
                    opp_val = getattr(item, "locatedIn", None)
                    
                    setattr(item, "locatedIn", self)
                    
