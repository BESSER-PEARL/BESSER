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
composition = Table(
    "composition",
    Base.metadata,
    Column("lifecyclestage_id", ForeignKey("lifecyclestage.id"), primary_key=True),
    Column("rawmaterial_id", ForeignKey("rawmaterial.id"), primary_key=True),
)
stage = Table(
    "stage",
    Base.metadata,
    Column("productpassport_id", ForeignKey("productpassport.id"), primary_key=True),
    Column("lifecyclestage_id", ForeignKey("lifecyclestage.id"), primary_key=True),
)

# Tables definition
class Reparation(Base):
    
    __tablename__ = "reparation"
    id: Mapped[int] = mapped_column(primary_key=True)
    date_set: Mapped[date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(String(100))

class LifecycleStage(Base):
    
    __tablename__ = "lifecyclestage"
    id: Mapped[int] = mapped_column(primary_key=True)
    start: Mapped[date] = mapped_column(Date)
    end: Mapped[date] = mapped_column(Date)
    type_spec: Mapped[str]
    __mapper_args__ = {
        "polymorphic_identity": "lifecyclestage",
        "polymorphic_on": "type_spec",
    }

class Design(LifecycleStage):
        
    __tablename__ = "design"
    id: Mapped[int] = mapped_column(ForeignKey("lifecyclestage.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "design",
    }

class Use(LifecycleStage):
        
    __tablename__ = "use"
    id: Mapped[int] = mapped_column(ForeignKey("lifecyclestage.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "use",
    }

class Manufacture(LifecycleStage):
        
    __tablename__ = "manufacture"
    id: Mapped[int] = mapped_column(ForeignKey("lifecyclestage.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "manufacture",
    }

class Distribution(LifecycleStage):
        
    __tablename__ = "distribution"
    id: Mapped[int] = mapped_column(ForeignKey("lifecyclestage.id"), primary_key=True)
    __mapper_args__ = {
        "polymorphic_identity": "distribution",
    }

class RawMaterial(Base):
    
    __tablename__ = "rawmaterial"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

class ProductPassport(Base):
    
    __tablename__ = "productpassport"
    id: Mapped[int] = mapped_column(primary_key=True)
    product_name: Mapped[str] = mapped_column(String(100))
    brand: Mapped[str] = mapped_column(String(100))
    code: Mapped[str] = mapped_column(String(100))


#--- Foreign keys and relationships of the reparation table
Reparation.use_id: Mapped["Use"] = mapped_column(ForeignKey("use.id"), nullable=False)
Reparation.reparations: Mapped["Use"] = relationship("Use", back_populates="reparations")

#--- Foreign keys and relationships of the lifecyclestage table
LifecycleStage.stage: Mapped[List["ProductPassport"]] = relationship("ProductPassport", secondary=stage, back_populates="stage")
LifecycleStage.composition: Mapped[List["RawMaterial"]] = relationship("RawMaterial", secondary=composition, back_populates="composition")

#--- Foreign keys and relationships of the use table
Use.reparations: Mapped[List["Reparation"]] = relationship("Reparation", back_populates="reparations")

#--- Foreign keys and relationships of the rawmaterial table
RawMaterial.composition: Mapped[List["LifecycleStage"]] = relationship("LifecycleStage", secondary=composition, back_populates="composition")

#--- Foreign keys and relationships of the productpassport table
ProductPassport.stage: Mapped[List["LifecycleStage"]] = relationship("LifecycleStage", secondary=stage, back_populates="stage")