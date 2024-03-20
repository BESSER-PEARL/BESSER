from datetime import datetime
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class Book(BaseModel):
    pages: int
    release: datetime
    title: str
    id: int  # the id created
    library: "Library"  # N:1 Relationship
    book_author_assoc: Set["Author"]  # N:M Relationship
 

class Author(BaseModel):
    name: str
    email: str
    id: int  # the id created
 

class Library(BaseModel):
    name: str
    address: str
    id: int  # the id created
 

