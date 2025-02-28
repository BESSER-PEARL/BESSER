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
name_assoc = Table(
    "name_assoc",
    Base.metadata,
    Column("attr_assoc1", ForeignKey("name1.id"), primary_key=True),
    Column("attr_assoc2", ForeignKey("name2.id"), primary_key=True),
)

# Tables definition
class name2(Base):
    __tablename__ = "name2"
    id: Mapped[int] = mapped_column(primary_key=True)
    attr2: Mapped[int] = mapped_column(Integer)

class name1(Base):
    __tablename__ = "name1"
    id: Mapped[int] = mapped_column(primary_key=True)
    attr1: Mapped[int] = mapped_column(Integer)


#--- Foreign keys and relationships of the name2 table
name2.attr_assoc1: Mapped[List["name1"]] = relationship("name1", secondary=name_assoc, back_populates="attr_assoc2")

#--- Foreign keys and relationships of the name1 table
name1.attr_assoc2: Mapped[List["name2"]] = relationship("name2", secondary=name_assoc, back_populates="attr_assoc1")

# Database connection

DATABASE_URL = "sqlite:///Name.db"  # SQLite connection

engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine)