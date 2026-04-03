import enum
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, ForeignKey, Table, Text, Boolean, String, Date, 
    Time, DateTime, Float, Integer, Enum
)
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, mapped_column, relationship
)
from datetime import datetime as dt_datetime, time as dt_time, date as dt_date

class Base(DeclarativeBase):
    pass

# Definitions of Enumerations
class UserRole(enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    MODERATOR = "MODERATOR"

class PostStatus(enum.Enum):
    PUBLISHED = "PUBLISHED"
    DRAFT = "DRAFT"
    ARCHIVED = "ARCHIVED"


# Tables definition for many-to-many relationships

# Tables definition
class Comment(Base):
    __tablename__ = "comment"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[dt_datetime] = mapped_column(DateTime)
    commenter_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    post_id: Mapped[int] = mapped_column(ForeignKey("post.id"))

class Post(Base):
    __tablename__ = "post"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(100))
    content: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[dt_datetime] = mapped_column(DateTime)
    author_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(100))
    username: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(100))
    bio: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean)


#--- Relationships of the comment table
Comment.commenter: Mapped["User"] = relationship("User", back_populates="comments", foreign_keys=[Comment.commenter_id])
Comment.post: Mapped["Post"] = relationship("Post", back_populates="comments", foreign_keys=[Comment.post_id])

#--- Relationships of the post table
Post.comments: Mapped[List["Comment"]] = relationship("Comment", back_populates="post", foreign_keys=[Comment.post_id])
Post.author: Mapped["User"] = relationship("User", back_populates="posts", foreign_keys=[Post.author_id])

#--- Relationships of the user table
User.posts: Mapped[List["Post"]] = relationship("Post", back_populates="author", foreign_keys=[Post.author_id])
User.comments: Mapped[List["Comment"]] = relationship("Comment", back_populates="commenter", foreign_keys=[Comment.commenter_id])

# Database connection
DATABASE_URL = "sqlite:///BlogPlatform.db"  # SQLite connection
engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine, checkfirst=True)