import pickle
from typing import Union
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, create_engine, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.types import JSON
import json

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test232323.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define the SQLAlchemy model
class Property(Base):
    __tablename__ = 'properties'

    id = Column(Integer, primary_key=True)
    storage_name = Column(String)
    name = Column(String, nullable=False)
    class_id = Column(Integer, ForeignKey('classes.id', ondelete="CASCADE"))  # Foreign key to Class
    class_ = relationship("Class", back_populates="properties")
    property_type_id = Column(Integer, ForeignKey('primitive_data_types.id'))
    property_type = relationship("PrimitiveDataType", backref="properties")
    multiplicity = Column(JSON, default="[1, 1]")  # Store as JSON array
    visibility = Column(String, default='public')
    is_composite = Column(Boolean, default=False)
    is_navigable = Column(Boolean, default=False)
    is_aggregation = Column(Boolean, default=False)
    is_id = Column(Boolean, default=False)
    is_read_only = Column(Boolean, default=False)

    def set_multiplicity(self, lower, upper):
        """Helper method to set the multiplicity as a list (JSON serializable)."""
        self.multiplicity = json.dumps([lower, upper])

    def get_multiplicity(self):
        """Helper method to retrieve the multiplicity as a tuple."""
        lower, upper = json.loads(self.multiplicity)
        return (lower, upper)


class PrimitiveDataType(Base):
    __tablename__ = 'primitive_data_types'
    id = Column(Integer, primary_key=True)
    storage_name = Column(String)
    name = Column(String, nullable=False)


class Class(Base):
    __tablename__ = 'classes'
    id = Column(Integer, primary_key=True)
    storage_name = Column(String)
    name = Column(String, nullable=False)
    is_abstract = Column(Boolean, default=False)
    is_read_only = Column(Boolean, default=False)
    # Define a relationship to Property
    properties = relationship("Property", back_populates="class_", cascade="all, delete")




Base.metadata.create_all(bind=engine)


# Define the Pydantic model for request data
class PropertyCreate(BaseModel):
    storage_name: str = None
    name: str
    property_type_id: int
    multiplicity: tuple[Union[int, str], Union[int, str]] = (1, 1)
    visibility: str = 'public'
    is_composite: bool = False
    is_navigable: bool = False
    is_aggregation: bool = False
    is_id: bool = False
    is_read_only: bool = False


class PrimitiveDataTypeCreate(BaseModel):
    storage_name: str = None
    name: str


class ClassCreate(BaseModel):
    storage_name: str = None
    name: str
    property_ids: list[int] = None
    is_abstract: bool = False
    is_read_only: bool = False



app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


@app.post("/primitive_data_types/")
def create_primitive_data_type(primitive_data_type: PrimitiveDataTypeCreate, db: Session = Depends(get_db)):
    db_primitive_data_type = PrimitiveDataType(storage_name=primitive_data_type.storage_name,
                                               name=primitive_data_type.name)
    db.add(db_primitive_data_type)
    db.commit()
    db.refresh(db_primitive_data_type)
    return {"id": db_primitive_data_type.id, "message": "Primitive Data Type stored successfully"}


@app.post("/properties/")
def create_property(property_data: PropertyCreate, db: Session = Depends(get_db)):
    # Assuming property_type in PropertyCreate is supposed to be property_type_id
    db_property = Property(
        storage_name=property_data.storage_name,
        name=property_data.name,
        property_type_id=property_data.property_type_id,  # Use ID directly
        multiplicity=json.dumps(property_data.multiplicity),  # Ensure it's JSON serialized
        visibility=property_data.visibility,
        is_composite=property_data.is_composite,
        is_navigable=property_data.is_navigable,
        is_aggregation=property_data.is_aggregation,
        is_id=property_data.is_id,
        is_read_only=property_data.is_read_only
    )
    db.add(db_property)
    db.commit()
    db.refresh(db_property)

    return {"id": db_property.id, "message": "Property stored successfully"}


@app.get("/properties/{property_id}")
def get_property(property_id: int, db: Session = Depends(get_db)):
    db_property = db.query(Property).filter(Property.id == property_id).first()
    if db_property is not None:
        return {"property": str(db_property)}
    else:
        raise HTTPException(status_code=404, detail="Property not found")


@app.post("/classes/")
def create_class_with_existing_properties(class_data: ClassCreate, db: Session = Depends(get_db)):
    new_class = Class(
        storage_name=class_data.storage_name,
        name=class_data.name,
        is_abstract=class_data.is_abstract,
        is_read_only=class_data.is_read_only
    )
    db.add(new_class)
    db.flush()  # Ensure 'new_class.id' is available immediately

    # Link existing properties by updating their class_id
    for property_id in class_data.property_ids:
        existing_property = db.query(Property).filter(Property.id == property_id).one_or_none()
        if existing_property:
            existing_property.class_id = new_class.id
        else:
            print(f"No property found with ID {property_id}")

    '''# Create new properties and link them to the class
    for prop in new_properties:
        new_property = Property(
            storage_name=prop.get('storage_name', ''),
            name=prop['name'],
            property_type_id=prop['property_type_id'],
            multiplicity=json.dumps(prop['multiplicity']),
            visibility=prop.get('visibility', 'public'),
            is_composite=prop.get('is_composite', False),
            is_navigable=prop.get('is_navigable', False),
            is_aggregation=prop.get('is_aggregation', False),
            is_id=prop.get('is_id', False),
            is_read_only=prop.get('is_read_only', False),
            class_id=new_class.id
        )
        db.add(new_property)'''

    db.commit()
    return {"id": new_class.id, "message": "Class stored successfully"}

# Modify the endpoint to handle the association between Class and Property




# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

