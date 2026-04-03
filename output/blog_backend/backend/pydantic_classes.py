from datetime import datetime, date, time
from typing import Any, List, Optional, Union, Set
from enum import Enum
from pydantic import BaseModel, field_validator


############################################
# Enumerations are defined here
############################################

class UserRole(Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    MODERATOR = "MODERATOR"

class PostStatus(Enum):
    PUBLISHED = "PUBLISHED"
    DRAFT = "DRAFT"
    ARCHIVED = "ARCHIVED"

############################################
# Classes are defined here
############################################
class CommentCreate(BaseModel):
    text: str
    created_at: datetime
    id: int  # the id
    commenter: int  # N:1 Relationship (mandatory)
    post: int  # N:1 Relationship (mandatory)


class PostCreate(BaseModel):
    created_at: datetime
    content: str
    title: str
    id: int  # the id
    comments: Optional[List[int]] = None  # 1:N Relationship
    author: int  # N:1 Relationship (mandatory)


class UserCreate(BaseModel):
    bio: Optional[str] = None
    id: int  # the id
    email: str
    is_active: bool
    username: str
    password_hash: str
    comments: Optional[List[int]] = None  # 1:N Relationship
    posts: Optional[List[int]] = None  # 1:N Relationship


