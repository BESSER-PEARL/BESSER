from datetime import datetime, date, time
from typing import Any, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator


############################################
# Enumerations are defined here
############################################

############################################
# Classes are defined here
############################################
class AuthorCreate(BaseModel):
    name: str
    email: str
    publishes: List[int]  # N:M Relationship


class BookCreate(BaseModel):
    title: str
    pages: int
    release: date
    locatedIn: int  # N:1 Relationship (mandatory)
    writtenBy: List[int]  # N:M Relationship


class LibraryCreate(BaseModel):
    address: str
    name: str
    has: Optional[List[int]] = None  # 1:N Relationship


