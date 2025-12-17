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
author_book = Table(
    "author_book",
    Base.metadata,
    Column("publishes", ForeignKey("book.id"), primary_key=True),
    Column("writtenBy", ForeignKey("author.id"), primary_key=True),
)

# Tables definition
class Library(Base):
    __tablename__ = "library"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    address: Mapped[str] = mapped_column(String(100))

    def has_address(self):
        has_it = self.address is not None and len(self.address) > 0
        print(f"Library '{self.name}' has address: {has_it}")
        return has_it


    def info(self):
        print(f"Library: {self.name}")
        print(f"Address: {self.address}")
        return {'name': self.name, 'address': self.address}


    @staticmethod
    def count_libraries(session):
        count = session.query(Library).count()
        print(f"Total libraries in database: {count}")
        return count


    @staticmethod
    def find_by_name(session, name):
        library = session.query(Library).filter(Library.name.ilike(f'%{name}%')).first()
        if library:
            print(f"Found library: {library.name}")
        else:
            print(f"Library with name containing '{name}' not found")
        return library


class Book(Base):
    __tablename__ = "book"
    id: Mapped[int] = mapped_column(primary_key=True)
    pages: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(100))
    release: Mapped[date] = mapped_column(Date)
    locatedIn_id: Mapped[int] = mapped_column(ForeignKey("library.id"))

    def is_old(self, years=10):
        from datetime import datetime
        current_year = datetime.now().year
        book_age = current_year - self.release.year
        is_old = book_age >= years
        print(f"Book '{self.title}' published {book_age} years ago. Is old ({years}+ years): {is_old}")
        return is_old


    def hello_world(self):
        print(f"Hello from {self.title}!")
        return self.title


    @staticmethod
    def count_long_books(session, min_pages=300):
        count = session.query(Book).filter(Book.pages > min_pages).count()
        print(f"Found {count} books with more than {min_pages} pages")
        return count


    @staticmethod
    def get_longest(session):
        return session.query(Book).order_by(Book.pages.desc()).first()


    def is_long(self, min_pages=300):
        is_long = self.pages > min_pages
        print(f"Book '{self.title}' has {self.pages} pages. Is long ({min_pages}+): {is_long}")
        return is_long


    def calculate_discount(self, discount):
        discounted = int(self.pages * (1 - discount))
        print(f"Original pages: {self.pages}, After {discount*100}% discount: {discounted}")
        return discounted


    @staticmethod
    def average_pages(session):
        from sqlalchemy import func
        avg = session.query(func.avg(Book.pages)).scalar()
        avg = float(avg) if avg else 0
        print(f"Average pages across all books: {avg:.2f}")
        return avg


class Author(Base):
    __tablename__ = "author"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100))


#--- Relationships of the library table
Library.has: Mapped[List["Book"]] = relationship("Book", back_populates="locatedIn", foreign_keys=[Book.locatedIn_id])

#--- Relationships of the book table
Book.writtenBy: Mapped[List["Author"]] = relationship("Author", secondary=author_book, back_populates="publishes")
Book.locatedIn: Mapped["Library"] = relationship("Library", back_populates="has", foreign_keys=[Book.locatedIn_id])

#--- Relationships of the author table
Author.publishes: Mapped[List["Book"]] = relationship("Book", secondary=author_book, back_populates="writtenBy")

# Database connection
DATABASE_URL = "sqlite:///Class_Diagram.db"  # SQLite connection
engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine, checkfirst=True)