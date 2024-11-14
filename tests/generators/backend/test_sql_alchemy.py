import pytest
import os
import importlib.util
import sys
from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, MetaData, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    PrimitiveDataType, Multiplicity, BinaryAssociation

output_file = 'output/sql_alchemy.py'

def generation():
    class1 = Class(name="name1", attributes={
        Property(name="attr1", type=PrimitiveDataType("int")),
    })
    class2 = Class(name="name2", attributes={
        Property(name="attr2", type=PrimitiveDataType("int"))
    })
    association = BinaryAssociation(name="name_assoc", ends={
        Property(name="attr_assoc1", owner=class2, type=class1, multiplicity=Multiplicity(1, "*")),
        Property(name="attr_assoc2", owner=class1, type=class2, multiplicity=Multiplicity(1, "*"))
    })

    domain_model = DomainModel(name="Name", types={class1, class2}, associations={association}, packages=None, constraints=None)

    sqlalchemy_model = SQLAlchemyGenerator(model=domain_model)
    sqlalchemy_model.generate()

generation()

spec = importlib.util.spec_from_file_location("sql_alchemy", "output/sql_alchemy.py")
sql_alchemy = importlib.util.module_from_spec(spec)
sys.modules["sql_alchemy"] = sql_alchemy
spec.loader.exec_module(sql_alchemy)



from sql_alchemy import Base, name1, name2, name_assoc
import shutil

# Fixture for database setup
@pytest.fixture(scope='module')
def setup_database():
    # Create an in-memory SQLite database
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    yield Session
    Session.remove()
    Base.metadata.drop_all(engine)

def test_table_creation(setup_database):
    Session = setup_database
    session = Session()
    assert session.query(name1).count() == 0
    assert session.query(name2).count() == 0
    session.close()

def test_insert_and_query(setup_database):
    Session = setup_database
    session = Session()
    entity1 = name1(attr1=100)
    entity2 = name2(attr2=200)
    entity1.attr_assoc2.append(entity2)
    session.add(entity1)
    session.add(entity2)
    session.commit()

    assert session.query(name1).first().attr_assoc2[0].attr2 == 200
    assert session.query(name2).first().attr_assoc1[0].attr1 == 100
    session.close()

def test_insert_and_relationship_creation(setup_database):
    Session = setup_database
    session = Session()


    # Validate the relationship creation
    assert session.query(name1).count() == 1
    assert session.query(name2).count() == 1
    assert len(session.query(name1).first().attr_assoc2) == 1
    session.close()

def test_delete_relationship(setup_database):
    Session = setup_database
    session = Session()

    '''
    n1 = name1(attr1=100)    
    n2 = name2(attr2=200)
    n1.attr_assoc2.append(n2)  # Link name1 and name2 through the association table
    session.add(n1)
    session.add(n2)
    session.commit()
    '''
    n1 = session.query(name1).first()
    n2 = session.query(name2).first()

    if n1 and n2:
        n1.attr_assoc2.remove(n2)  # Remove the relationship
        session.commit()

        # Verify that the association table reflects the relationship removal
        assert len(n1.attr_assoc2) == 0
        assert len(n2.attr_assoc1) == 0

        # Double-check that no entries remain in the association table
        association_count = session.execute(
            select(func.count()).select_from(name_assoc)  # Updated to correct syntax
        ).scalar()
        assert association_count == 0
    else:
        pytest.fail("Entities not found; ensure data setup is correct.")

    session.close()
    os.remove(output_file)
    # Delete the folder and file after importing
    shutil.rmtree('output')