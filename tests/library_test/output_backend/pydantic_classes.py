from datetime import datetime, date
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class BookCreate(BaseModel):
    pages: int
    title: str
    release: datetime
    id: int  # id created
    library_id: int
    authors: Set["AuthorCreate"]  # N:M Relationship

 

class AuthorCreate(BaseModel):
    name: str
    email: str
    id: int  # id created
    books: Set["BookCreate"]  # N:M Relationship

 

class LibraryCreate(BaseModel):
    address: str
    name: str
    id: int  # id created

 

