from datetime import datetime, date
from typing import List, Optional, Union,Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class BookCreate(BaseModel):
    pages: int
    release: datetime
    title: str
    authors_id: List[int]
    library_id: int

class AuthorCreate(BaseModel):
    name: str
    email: str
    books_id: List[int]

class LibraryCreate(BaseModel):
    address: str
    name: str
