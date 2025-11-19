import enum
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, ForeignKey, Table, Text, Boolean, String, Date, 
    Time, DateTime, Float, Integer, Enum
)
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, mapped_column, relationship
)
from datetime import datetime, time, date

class Base(DeclarativeBase):
    pass



# Tables definition for many-to-many relationships
book_author_assoc = Table(
    "book_author_assoc",
    Base.metadata,
    Column("writtenBy", ForeignKey("author.id"), primary_key=True),
    Column("publishes", ForeignKey("book.id"), primary_key=True),
)

# Tables definition
class Author(Base):
    __tablename__ = "author"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100))

class Book(Base):
    __tablename__ = "book"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(100))
    pages: Mapped[int] = mapped_column(Integer)
    release: Mapped[date] = mapped_column(Date)
    locatedIn_id: Mapped[int] = mapped_column(ForeignKey("library.id"), nullable=False)

class Library(Base):
    __tablename__ = "library"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    address: Mapped[str] = mapped_column(String(100))


#--- Relationships of the author table
Author.publishes: Mapped[List["Book"]] = relationship("Book", secondary=book_author_assoc, back_populates="writtenBy")

#--- Relationships of the book table
Book.writtenBy: Mapped[List["Author"]] = relationship("Author", secondary=book_author_assoc, back_populates="publishes")
Book.locatedIn: Mapped["Library"] = relationship("Library", back_populates="has", foreign_keys=[Book.locatedIn_id])

#--- Relationships of the library table
Library.has: Mapped[List["Book"]] = relationship("Book", back_populates="locatedIn", foreign_keys=[Book.locatedIn_id])

# Database connection
DATABASE_URL = "sqlite:///Library_model.db"  # SQLite connection
engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine, checkfirst=True)