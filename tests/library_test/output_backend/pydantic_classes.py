from datetime import datetime
from typing import List, Set
from pydantic import BaseModel

############################################
#
# The classes are defined here
#
############################################

class AuthorCreate(BaseModel):
    email: str
    name: str
    book_author_assoc: Set["BookCreate"]  # N:M Relationship
 

class BookCreate(BaseModel):
    pages: int
    title: str
    release: datetime
    library: "LibraryCreate"  # N:1 Relationship
 

class LibraryCreate(BaseModel):
    name: str
    address: str
 

