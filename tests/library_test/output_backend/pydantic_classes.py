from datetime import datetime
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class AuthorCreate(BaseModel):
    name: str
    email: str
 

class BookCreate(BaseModel):
    release: datetime
    title: str
    pages: int
 

class LibraryCreate(BaseModel):
    name: str
    address: str
 

