from datetime import datetime, date
from typing import List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel


############################################
# Enumerations are defined here
############################################

############################################
# Classes are defined here
############################################
class LibraryCreate(BaseModel):
    address: str
    name: str
    has: Optional[List[int]] = None  # 1:N Relationship

class BookCreate(BaseModel):
    release: date
    pages: int
    title: str
    locatedIn: int  # N:1 Relationship (mandatory)
    writtenBy: List[int]  # N:M Relationship

class AuthorCreate(BaseModel):
    name: str
    email: str
    publishes: List[int]  # N:M Relationship

