from typing import List
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy.orm import column_property
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import Boolean, String, Date, Time, DateTime, Float, Integer
from datetime import datetime, time, date

class Base(DeclarativeBase):
    pass

# Tables definition for many-to-many relationships
book_author_assoc = Table(
    "book_author_assoc",
    Base.metadata,
    Column("author_id", ForeignKey("author.id"), primary_key=True),
    Column("book_id", ForeignKey("book.id"), primary_key=True),
)

# Tables definition
class Book(Base):
    
    __tablename__ = "book"
    id: Mapped[int] = mapped_column(primary_key=True)
    release: Mapped[datetime] = mapped_column(DateTime)
    title: Mapped[str] = mapped_column(String(100))
    pages: Mapped[int] = mapped_column(Integer)

class Author(Base):
    
    __tablename__ = "author"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100))

class Library(Base):
    
    __tablename__ = "library"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    address: Mapped[str] = mapped_column(String(100))


#--- Foreign keys and relationships of the book table
Book.library_id: Mapped["Library"] = mapped_column(ForeignKey("library.id"), nullable=False)
Book.locatedIn: Mapped["Library"] = relationship("Library", back_populates="has")
Book.writedBy: Mapped[List["Author"]] = relationship("Author", secondary=book_author_assoc, back_populates="publishes")

#--- Foreign keys and relationships of the author table
Author.publishes: Mapped[List["Book"]] = relationship("Book", secondary=book_author_assoc, back_populates="writedBy")

#--- Foreign keys and relationships of the library table
Library.has: Mapped[List["Book"]] = relationship("Book", back_populates="locatedIn")