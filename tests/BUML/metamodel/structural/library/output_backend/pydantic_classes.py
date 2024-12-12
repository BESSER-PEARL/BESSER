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
    authors: Optional[List[Union["AuthorCreate", int]]] = None  # N:M Relationship
    library_id: int

class AuthorCreate(BaseModel):
    name: str
    email: str
    books: Optional[List[Union["BookCreate", int]]] = None  # N:M Relationship

class LibraryCreate(BaseModel):
    address: str
    name: str
