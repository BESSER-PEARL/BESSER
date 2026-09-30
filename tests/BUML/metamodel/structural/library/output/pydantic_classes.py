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
class Author(BaseModel):
    email: str
    name: str
    id: int  # id created
    publishes: List[int]  # N:M Relationship


class Book(BaseModel):
    release: date
    pages: int
    title: str
    id: int  # id created
    writtenBy: List[int]  # N:M Relationship
    locatedIn: "Library"  # N:1 Relationship


class Library(BaseModel):
    name: str
    address: str
    id: int  # id created
    has: List["Book"]  # 1:N Relationship


