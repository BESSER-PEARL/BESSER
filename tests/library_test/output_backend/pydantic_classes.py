from datetime import datetime, date
from typing import List, Optional, Union,Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class BookCreate(BaseModel):
    title: str
    release: datetime
    pages: int
    authors_id: List[int]
    library_id: int

 

class AuthorCreate(BaseModel):
    email: str
    name: str
    books_id: List[int]

 

class LibraryCreate(BaseModel):
    name: str
    address: str

 

