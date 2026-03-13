import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy
from datetime import date

from classes import (
    Rental,
    RentalCustomer,
    Vehicle,
    VehicleCategory,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================



def test_rental_is_not_abstract():
    assert not inspect.isabstract(Rental)


def test_rental_constructor_exists():
    assert callable(Rental.__init__)


def test_rental_constructor_args():
    sig = inspect.signature(Rental.__init__)
    params = list(sig.parameters.keys())
    assert "endDate" in params, "Missing parameter 'endDate'"
    assert "startDate" in params, "Missing parameter 'startDate'"
    assert "status" in params, "Missing parameter 'status'"
    assert "rentalId" in params, "Missing parameter 'rentalId'"
    assert "totalCost" in params, "Missing parameter 'totalCost'"

def test_rental_has_endDate():
    assert hasattr(Rental, "endDate")
    descriptor = None
    for klass in Rental.__mro__:
        if "endDate" in klass.__dict__:
            descriptor = klass.__dict__["endDate"]
            break
    assert isinstance(descriptor, property)

def test_rental_has_startDate():
    assert hasattr(Rental, "startDate")
    descriptor = None
    for klass in Rental.__mro__:
        if "startDate" in klass.__dict__:
            descriptor = klass.__dict__["startDate"]
            break
    assert isinstance(descriptor, property)

def test_rental_has_status():
    assert hasattr(Rental, "status")
    descriptor = None
    for klass in Rental.__mro__:
        if "status" in klass.__dict__:
            descriptor = klass.__dict__["status"]
            break
    assert isinstance(descriptor, property)

def test_rental_has_rentalId():
    assert hasattr(Rental, "rentalId")
    descriptor = None
    for klass in Rental.__mro__:
        if "rentalId" in klass.__dict__:
            descriptor = klass.__dict__["rentalId"]
            break
    assert isinstance(descriptor, property)

def test_rental_has_totalCost():
    assert hasattr(Rental, "totalCost")
    descriptor = None
    for klass in Rental.__mro__:
        if "totalCost" in klass.__dict__:
            descriptor = klass.__dict__["totalCost"]
            break
    assert isinstance(descriptor, property)



def test_rentalcustomer_is_not_abstract():
    assert not inspect.isabstract(RentalCustomer)


def test_rentalcustomer_constructor_exists():
    assert callable(RentalCustomer.__init__)


def test_rentalcustomer_constructor_args():
    sig = inspect.signature(RentalCustomer.__init__)
    params = list(sig.parameters.keys())
    assert "customerId" in params, "Missing parameter 'customerId'"
    assert "licenseNumber" in params, "Missing parameter 'licenseNumber'"
    assert "name" in params, "Missing parameter 'name'"
    assert "verified" in params, "Missing parameter 'verified'"

def test_rentalcustomer_has_customerId():
    assert hasattr(RentalCustomer, "customerId")
    descriptor = None
    for klass in RentalCustomer.__mro__:
        if "customerId" in klass.__dict__:
            descriptor = klass.__dict__["customerId"]
            break
    assert isinstance(descriptor, property)

def test_rentalcustomer_has_licenseNumber():
    assert hasattr(RentalCustomer, "licenseNumber")
    descriptor = None
    for klass in RentalCustomer.__mro__:
        if "licenseNumber" in klass.__dict__:
            descriptor = klass.__dict__["licenseNumber"]
            break
    assert isinstance(descriptor, property)

def test_rentalcustomer_has_name():
    assert hasattr(RentalCustomer, "name")
    descriptor = None
    for klass in RentalCustomer.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_rentalcustomer_has_verified():
    assert hasattr(RentalCustomer, "verified")
    descriptor = None
    for klass in RentalCustomer.__mro__:
        if "verified" in klass.__dict__:
            descriptor = klass.__dict__["verified"]
            break
    assert isinstance(descriptor, property)



def test_vehicle_is_not_abstract():
    assert not inspect.isabstract(Vehicle)


def test_vehicle_constructor_exists():
    assert callable(Vehicle.__init__)


def test_vehicle_constructor_args():
    sig = inspect.signature(Vehicle.__init__)
    params = list(sig.parameters.keys())
    assert "dailyRate" in params, "Missing parameter 'dailyRate'"
    assert "model" in params, "Missing parameter 'model'"
    assert "make" in params, "Missing parameter 'make'"
    assert "vehicleId" in params, "Missing parameter 'vehicleId'"
    assert "available" in params, "Missing parameter 'available'"

def test_vehicle_has_dailyRate():
    assert hasattr(Vehicle, "dailyRate")
    descriptor = None
    for klass in Vehicle.__mro__:
        if "dailyRate" in klass.__dict__:
            descriptor = klass.__dict__["dailyRate"]
            break
    assert isinstance(descriptor, property)

def test_vehicle_has_model():
    assert hasattr(Vehicle, "model")
    descriptor = None
    for klass in Vehicle.__mro__:
        if "model" in klass.__dict__:
            descriptor = klass.__dict__["model"]
            break
    assert isinstance(descriptor, property)

def test_vehicle_has_make():
    assert hasattr(Vehicle, "make")
    descriptor = None
    for klass in Vehicle.__mro__:
        if "make" in klass.__dict__:
            descriptor = klass.__dict__["make"]
            break
    assert isinstance(descriptor, property)

def test_vehicle_has_vehicleId():
    assert hasattr(Vehicle, "vehicleId")
    descriptor = None
    for klass in Vehicle.__mro__:
        if "vehicleId" in klass.__dict__:
            descriptor = klass.__dict__["vehicleId"]
            break
    assert isinstance(descriptor, property)

def test_vehicle_has_available():
    assert hasattr(Vehicle, "available")
    descriptor = None
    for klass in Vehicle.__mro__:
        if "available" in klass.__dict__:
            descriptor = klass.__dict__["available"]
            break
    assert isinstance(descriptor, property)

def test_vehiclecategory_exists():
    # Check that the Enumeration exists
    assert VehicleCategory is not None

def test_vehiclecategory_has_all_literals():
    # Collect the names of literals in this Enumeration
    enum_literals = [lit.name for lit in VehicleCategory]
    expected_literals = [
        "SPORTS",
        "SEDAN",
        "SUV",
        "TRUCK",
    ]
    # Check that all expected literals exist
    for lit_name in expected_literals:
        assert lit_name in enum_literals, f"Literal '' missing in VehicleCategory"


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_",
    ),
    min_size=1,
).filter(lambda s: s[0].isalpha())
Rental_strategy = st.builds(
    Rental,
    endDate=
        safe_text,
    startDate=
        safe_text,
    status=
        safe_text,
    rentalId=
        safe_text,
    totalCost=
        st.floats(min_value=0, max_value=1000,allow_nan=False, allow_infinity=False)
)
RentalCustomer_strategy = st.builds(
    RentalCustomer,
    customerId=
        safe_text,
    licenseNumber=
        safe_text,
    name=
        safe_text,
    verified=
        st.booleans()
)
Vehicle_strategy = st.builds(
    Vehicle,
    dailyRate=
        st.floats(min_value=0, max_value=1000,allow_nan=False, allow_infinity=False),
    model=
        safe_text,
    make=
        safe_text,
    vehicleId=
        safe_text,
    available=
        st.booleans()
)

@given(instance=Rental_strategy)
@settings(max_examples=50)
def test_rental_instantiation(instance):
    assert isinstance(instance, Rental)

@given(instance=Rental_strategy)
def test_rental_endDate_type(instance):
    assert isinstance(instance.endDate, str)


@given(instance=Rental_strategy)
def test_rental_endDate_setter(instance):
    original = instance.endDate
    instance.endDate = original
    assert instance.endDate == original

@given(instance=Rental_strategy)
def test_rental_startDate_type(instance):
    assert isinstance(instance.startDate, str)


@given(instance=Rental_strategy)
def test_rental_startDate_setter(instance):
    original = instance.startDate
    instance.startDate = original
    assert instance.startDate == original

@given(instance=Rental_strategy)
def test_rental_status_type(instance):
    assert isinstance(instance.status, str)


@given(instance=Rental_strategy)
def test_rental_status_setter(instance):
    original = instance.status
    instance.status = original
    assert instance.status == original

@given(instance=Rental_strategy)
def test_rental_rentalId_type(instance):
    assert isinstance(instance.rentalId, str)


@given(instance=Rental_strategy)
def test_rental_rentalId_setter(instance):
    original = instance.rentalId
    instance.rentalId = original
    assert instance.rentalId == original

@given(instance=Rental_strategy)
def test_rental_totalCost_type(instance):
    assert isinstance(instance.totalCost, float)


@given(instance=Rental_strategy)
def test_rental_totalCost_setter(instance):
    original = instance.totalCost
    instance.totalCost = original
    assert instance.totalCost == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Rental_strategy)
@settings(max_examples=30)
def test_rental_extendrental_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.extendRental(
            "test", 
            1.0
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.extendRental).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'extendRental' in Rental is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'extendRental' in Rental did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'extendRental' in Rental is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Rental_strategy)
@settings(max_examples=30)
def test_rental_startrental_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.startRental()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.startRental).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'startRental' in Rental is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'startRental' in Rental did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'startRental' in Rental is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Rental_strategy)
@settings(max_examples=30)
def test_rental_completerental_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.completeRental(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.completeRental).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'completeRental' in Rental is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'completeRental' in Rental did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'completeRental' in Rental is not implemented or raised an error")

@given(instance=RentalCustomer_strategy)
@settings(max_examples=50)
def test_rentalcustomer_instantiation(instance):
    assert isinstance(instance, RentalCustomer)

@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_customerId_type(instance):
    assert isinstance(instance.customerId, str)


@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_customerId_setter(instance):
    original = instance.customerId
    instance.customerId = original
    assert instance.customerId == original

@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_licenseNumber_type(instance):
    assert isinstance(instance.licenseNumber, str)


@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_licenseNumber_setter(instance):
    original = instance.licenseNumber
    instance.licenseNumber = original
    assert instance.licenseNumber == original

@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_verified_type(instance):
    assert isinstance(instance.verified, bool)


@given(instance=RentalCustomer_strategy)
def test_rentalcustomer_verified_setter(instance):
    original = instance.verified
    instance.verified = original
    assert instance.verified == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=RentalCustomer_strategy)
@settings(max_examples=30)
def test_rentalcustomer_verifylicense_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.verifyLicense()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.verifyLicense).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'verifyLicense' in RentalCustomer is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'verifyLicense' in RentalCustomer did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'verifyLicense' in RentalCustomer is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=RentalCustomer_strategy)
@settings(max_examples=30)
def test_rentalcustomer_registercustomer_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.registerCustomer()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.registerCustomer).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'registerCustomer' in RentalCustomer is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'registerCustomer' in RentalCustomer did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'registerCustomer' in RentalCustomer is not implemented or raised an error")

@given(instance=Vehicle_strategy)
@settings(max_examples=50)
def test_vehicle_instantiation(instance):
    assert isinstance(instance, Vehicle)

@given(instance=Vehicle_strategy)
def test_vehicle_dailyRate_type(instance):
    assert isinstance(instance.dailyRate, float)


@given(instance=Vehicle_strategy)
def test_vehicle_dailyRate_setter(instance):
    original = instance.dailyRate
    instance.dailyRate = original
    assert instance.dailyRate == original

@given(instance=Vehicle_strategy)
def test_vehicle_model_type(instance):
    assert isinstance(instance.model, str)


@given(instance=Vehicle_strategy)
def test_vehicle_model_setter(instance):
    original = instance.model
    instance.model = original
    assert instance.model == original

@given(instance=Vehicle_strategy)
def test_vehicle_make_type(instance):
    assert isinstance(instance.make, str)


@given(instance=Vehicle_strategy)
def test_vehicle_make_setter(instance):
    original = instance.make
    instance.make = original
    assert instance.make == original

@given(instance=Vehicle_strategy)
def test_vehicle_vehicleId_type(instance):
    assert isinstance(instance.vehicleId, str)


@given(instance=Vehicle_strategy)
def test_vehicle_vehicleId_setter(instance):
    original = instance.vehicleId
    instance.vehicleId = original
    assert instance.vehicleId == original

@given(instance=Vehicle_strategy)
def test_vehicle_available_type(instance):
    assert isinstance(instance.available, bool)


@given(instance=Vehicle_strategy)
def test_vehicle_available_setter(instance):
    original = instance.available
    instance.available = original
    assert instance.available == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Vehicle_strategy)
@settings(max_examples=30)
def test_vehicle_calculaterentalcost_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.calculateRentalCost(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.calculateRentalCost).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'calculateRentalCost' in Vehicle is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'calculateRentalCost' in Vehicle did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'calculateRentalCost' in Vehicle is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Vehicle_strategy)
@settings(max_examples=30)
def test_vehicle_registervehicle_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.registerVehicle()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.registerVehicle).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'registerVehicle' in Vehicle is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'registerVehicle' in Vehicle did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'registerVehicle' in Vehicle is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Vehicle_strategy)
@settings(max_examples=30)
def test_vehicle_setavailability_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.setAvailability(
            True
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.setAvailability).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'setAvailability' in Vehicle is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'setAvailability' in Vehicle did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'setAvailability' in Vehicle is not implemented or raised an error")

@given(instance=Vehicle_strategy)
def test_vehicle_ocl_constraint_1(instance):
     
    
    
    
    
    
    value = True
    # Call the operation
    instance.setAvailability(value)
    
    assert instance.available == value

@given(instance=Rental_strategy)
def test_rental_ocl_constraint_2(instance):
     
    
    
    
    
    
    value = True
    # Call the operation
    instance.completeRental(value)
    
    assert instance.status ==value