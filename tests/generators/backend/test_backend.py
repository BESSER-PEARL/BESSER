import os
import shutil
import subprocess
import pytest
import requests
import time  # Added to use time.sleep for a brief pause
from datetime import datetime
from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, MetaData, select, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session, Mapped, mapped_column
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multiprocessing import Process
from besser.generators.backend import BackendGenerator
from besser.BUML.metamodel.structural import DomainModel, Class, Property, PrimitiveDataType, Multiplicity, BinaryAssociation


BASE_URL = "http://localhost:8000"

# Add this decorator to filter out the specific deprecation warning
pytestmark = pytest.mark.filterwarnings(
    "ignore:The 'app' shortcut is now deprecated.:DeprecationWarning"
)

def run_tests():
    pytest.main([__file__])

def test_file_generation():
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

    domain_model = DomainModel(name="Name", types={class1, class2}, associations={association})

    backend = BackendGenerator(model=domain_model, output_dir=".")
    backend.generate()

def test_get_all_name1():
    from main_api import app
    client = TestClient(app)
    response = client.get(f"{BASE_URL}/name1/")
    assert response.status_code == 200

def test_create_name1():
    from main_api import app
    client = TestClient(app)
    data = {
        "attr1": 1,
        "name2s_id": [] 
    }
    response = client.post(f"{BASE_URL}/name1/", json=data)
    assert response.status_code == 200

def test_get_name1():
    from main_api import app
    client = TestClient(app)
    name1_id = 1  # Adjust this ID according to your requirements
    response = client.get(f"{BASE_URL}/name1/{name1_id}/")
    assert response.status_code == 200

def test_update_name1():
    from main_api import app
    client = TestClient(app)
    name1_id = 1
    client.get(f"{BASE_URL}/name1/{name1_id}/")
    data = {
        "attr1": 11,
        "name2s_id": []
    }
    response = client.put(f"{BASE_URL}/name1/{name1_id}/", json=data)
    assert response.status_code == 200

def test_delete_name1():
    from main_api import app
    client = TestClient(app)
    name1_id = 1
    client.get(f"{BASE_URL}/name1/{name1_id}/")
    response = client.delete(f"{BASE_URL}/name1/{name1_id}/")
    assert response.status_code == 200
    client.close()
    delete_files()

def delete_files():
    os.remove("main_api.py")
    os.remove("pydantic_classes.py")
    os.remove("sql_alchemy.py")

if __name__ == "__main__":
    run_tests()
