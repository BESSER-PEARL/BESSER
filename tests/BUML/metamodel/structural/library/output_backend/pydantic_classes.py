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
class AuthorCreate(BaseModel):
    name: str
    email: str
    publishes: List[int]

class LibraryCreate(BaseModel):
    address: str
    name: str

class BookCreate(BaseModel):
    title: str
    pages: int
    release: date
    writtenBy: List[int]
    locatedIn: int  # N:1 Relationship

