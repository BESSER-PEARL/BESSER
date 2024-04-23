from datetime import datetime, date
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class BookCreate(BaseModel):
    release: datetime
    pages: int
    title: str
    authors_id: List[int]
    library_id: int

 

class LibraryCreate(BaseModel):
    address: str
    name: str

 

class AuthorCreate(BaseModel):
    email: str
    name: str
    books_id: List[int]

 

