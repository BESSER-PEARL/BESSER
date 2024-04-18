from datetime import datetime
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class Author(BaseModel):
    name: str
    email: str
    id: int  # the id created
    books: Set["Book"]  # N:M Relationship
 

class Book(BaseModel):
    pages: int
    release: datetime
    title: str
    id: int  # the id created
    library: "Library"  # N:1 Relationship
 

class Library(BaseModel):
    name: str
    address: str
    id: int  # the id created
 

