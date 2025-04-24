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
    Column("publishes", ForeignKey("book.id"), primary_key=True),
    Column("writtenBy", ForeignKey("author.id"), primary_key=True),
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

class Library(Base):
    __tablename__ = "library"
    id: Mapped[int] = mapped_column(primary_key=True)
    address: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100))


#--- Foreign keys and relationships of the author table
Author.publishes: Mapped[List["Book"]] = relationship("Book", secondary=book_author_assoc, back_populates="writtenBy")

#--- Foreign keys and relationships of the book table
Book.locatedIn_id: Mapped["Library"] = mapped_column(ForeignKey("library.id"), nullable=False)
Book.locatedIn: Mapped["Library"] = relationship("Library", back_populates="has")
Book.writtenBy: Mapped[List["Author"]] = relationship("Author", secondary=book_author_assoc, back_populates="publishes")

#--- Foreign keys and relationships of the library table
Library.has: Mapped[List["Book"]] = relationship("Book", back_populates="locatedIn")

# Database connection

DATABASE_URL = "sqlite:///Library_model.db"  # SQLite connection

engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine)