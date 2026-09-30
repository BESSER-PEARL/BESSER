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
    email: str
    name: str
    publishes: List[int]  # N:M Relationship


class BookCreate(BaseModel):
    release: date
    pages: int
    title: str
    writtenBy: List[int]  # N:M Relationship
    locatedIn: int  # N:1 Relationship (mandatory)


class LibraryCreate(BaseModel):
    name: str
    address: str
    has: Optional[List[int]] = None  # 1:N Relationship


