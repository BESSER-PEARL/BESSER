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
class Unit(BaseModel):
    id: int
    size: int
    id: int  # id created

class Employee(BaseModel):
    id: int
    salary: int
    id: int  # id created

class Department(Unit):
    id: int
    name: str
    id: int  # id created

