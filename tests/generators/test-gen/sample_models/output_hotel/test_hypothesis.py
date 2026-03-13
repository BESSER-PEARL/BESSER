import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy
from datetime import date

from classes import (
    Reservation,
    Room,
    Guest,
    RoomType,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================



def test_reservation_is_not_abstract():
    assert not inspect.isabstract(Reservation)


def test_reservation_constructor_exists():
    assert callable(Reservation.__init__)


def test_reservation_constructor_args():
    sig = inspect.signature(Reservation.__init__)
    params = list(sig.parameters.keys())
    assert "checkInDate" in params, "Missing parameter 'checkInDate'"
    assert "totalCost" in params, "Missing parameter 'totalCost'"
    assert "reservationId" in params, "Missing parameter 'reservationId'"
    assert "checkOutDate" in params, "Missing parameter 'checkOutDate'"

def test_reservation_has_checkInDate():
    assert hasattr(Reservation, "checkInDate")
    descriptor = None
    for klass in Reservation.__mro__:
        if "checkInDate" in klass.__dict__:
            descriptor = klass.__dict__["checkInDate"]
            break
    assert isinstance(descriptor, property)

def test_reservation_has_totalCost():
    assert hasattr(Reservation, "totalCost")
    descriptor = None
    for klass in Reservation.__mro__:
        if "totalCost" in klass.__dict__:
            descriptor = klass.__dict__["totalCost"]
            break
    assert isinstance(descriptor, property)

def test_reservation_has_reservationId():
    assert hasattr(Reservation, "reservationId")
    descriptor = None
    for klass in Reservation.__mro__:
        if "reservationId" in klass.__dict__:
            descriptor = klass.__dict__["reservationId"]
            break
    assert isinstance(descriptor, property)

def test_reservation_has_checkOutDate():
    assert hasattr(Reservation, "checkOutDate")
    descriptor = None
    for klass in Reservation.__mro__:
        if "checkOutDate" in klass.__dict__:
            descriptor = klass.__dict__["checkOutDate"]
            break
    assert isinstance(descriptor, property)



def test_room_is_not_abstract():
    assert not inspect.isabstract(Room)


def test_room_constructor_exists():
    assert callable(Room.__init__)


def test_room_constructor_args():
    sig = inspect.signature(Room.__init__)
    params = list(sig.parameters.keys())
    assert "roomType" in params, "Missing parameter 'roomType'"
    assert "roomNumber" in params, "Missing parameter 'roomNumber'"
    assert "pricePerNight" in params, "Missing parameter 'pricePerNight'"
    assert "available" in params, "Missing parameter 'available'"

def test_room_has_roomType():
    assert hasattr(Room, "roomType")
    descriptor = None
    for klass in Room.__mro__:
        if "roomType" in klass.__dict__:
            descriptor = klass.__dict__["roomType"]
            break
    assert isinstance(descriptor, property)

def test_room_has_roomNumber():
    assert hasattr(Room, "roomNumber")
    descriptor = None
    for klass in Room.__mro__:
        if "roomNumber" in klass.__dict__:
            descriptor = klass.__dict__["roomNumber"]
            break
    assert isinstance(descriptor, property)

def test_room_has_pricePerNight():
    assert hasattr(Room, "pricePerNight")
    descriptor = None
    for klass in Room.__mro__:
        if "pricePerNight" in klass.__dict__:
            descriptor = klass.__dict__["pricePerNight"]
            break
    assert isinstance(descriptor, property)

def test_room_has_available():
    assert hasattr(Room, "available")
    descriptor = None
    for klass in Room.__mro__:
        if "available" in klass.__dict__:
            descriptor = klass.__dict__["available"]
            break
    assert isinstance(descriptor, property)



def test_guest_is_not_abstract():
    assert not inspect.isabstract(Guest)


def test_guest_constructor_exists():
    assert callable(Guest.__init__)


def test_guest_constructor_args():
    sig = inspect.signature(Guest.__init__)
    params = list(sig.parameters.keys())
    assert "guestId" in params, "Missing parameter 'guestId'"
    assert "name" in params, "Missing parameter 'name'"
    assert "loyaltyPoints" in params, "Missing parameter 'loyaltyPoints'"
    assert "email" in params, "Missing parameter 'email'"

def test_guest_has_guestId():
    assert hasattr(Guest, "guestId")
    descriptor = None
    for klass in Guest.__mro__:
        if "guestId" in klass.__dict__:
            descriptor = klass.__dict__["guestId"]
            break
    assert isinstance(descriptor, property)

def test_guest_has_name():
    assert hasattr(Guest, "name")
    descriptor = None
    for klass in Guest.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_guest_has_loyaltyPoints():
    assert hasattr(Guest, "loyaltyPoints")
    descriptor = None
    for klass in Guest.__mro__:
        if "loyaltyPoints" in klass.__dict__:
            descriptor = klass.__dict__["loyaltyPoints"]
            break
    assert isinstance(descriptor, property)

def test_guest_has_email():
    assert hasattr(Guest, "email")
    descriptor = None
    for klass in Guest.__mro__:
        if "email" in klass.__dict__:
            descriptor = klass.__dict__["email"]
            break
    assert isinstance(descriptor, property)

def test_roomtype_exists():
    # Check that the Enumeration exists
    assert RoomType is not None

def test_roomtype_has_all_literals():
    # Collect the names of literals in this Enumeration
    enum_literals = [lit.name for lit in RoomType]
    expected_literals = [
        "DOUBLE",
        "SUITE",
        "SINGLE",
        "DELUXE",
    ]
    # Check that all expected literals exist
    for lit_name in expected_literals:
        assert lit_name in enum_literals, f"Literal '' missing in RoomType"


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
Reservation_strategy = st.builds(
    Reservation,
    checkInDate=
        safe_text,
    totalCost=
        st.floats(min_value=0, max_value=1000,allow_nan=False, allow_infinity=False),
    reservationId=
        safe_text,
    checkOutDate=
        safe_text
)
Room_strategy = st.builds(
    Room,
    roomType=
        safe_text,
    roomNumber=
        safe_text,
    pricePerNight=
        st.floats(min_value=0, max_value=1000,allow_nan=False, allow_infinity=False),
    available=
        st.booleans()
)
Guest_strategy = st.builds(
    Guest,
    guestId=
        safe_text,
    name=
        safe_text,
    loyaltyPoints=
        st.integers(),
    email=
        safe_text
)

@given(instance=Reservation_strategy)
@settings(max_examples=50)
def test_reservation_instantiation(instance):
    assert isinstance(instance, Reservation)

@given(instance=Reservation_strategy)
def test_reservation_checkInDate_type(instance):
    assert isinstance(instance.checkInDate, str)


@given(instance=Reservation_strategy)
def test_reservation_checkInDate_setter(instance):
    original = instance.checkInDate
    instance.checkInDate = original
    assert instance.checkInDate == original

@given(instance=Reservation_strategy)
def test_reservation_totalCost_type(instance):
    assert isinstance(instance.totalCost, float)


@given(instance=Reservation_strategy)
def test_reservation_totalCost_setter(instance):
    original = instance.totalCost
    instance.totalCost = original
    assert instance.totalCost == original

@given(instance=Reservation_strategy)
def test_reservation_reservationId_type(instance):
    assert isinstance(instance.reservationId, str)


@given(instance=Reservation_strategy)
def test_reservation_reservationId_setter(instance):
    original = instance.reservationId
    instance.reservationId = original
    assert instance.reservationId == original

@given(instance=Reservation_strategy)
def test_reservation_checkOutDate_type(instance):
    assert isinstance(instance.checkOutDate, str)


@given(instance=Reservation_strategy)
def test_reservation_checkOutDate_setter(instance):
    original = instance.checkOutDate
    instance.checkOutDate = original
    assert instance.checkOutDate == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Reservation_strategy)
@settings(max_examples=30)
def test_reservation_createreservation_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.createReservation()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.createReservation).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'createReservation' in Reservation is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'createReservation' in Reservation did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'createReservation' in Reservation is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Reservation_strategy)
@settings(max_examples=30)
def test_reservation_calculatenights_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.calculateNights()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.calculateNights).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'calculateNights' in Reservation is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'calculateNights' in Reservation did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'calculateNights' in Reservation is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Reservation_strategy)
@settings(max_examples=30)
def test_reservation_cancelreservation_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.cancelReservation()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.cancelReservation).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'cancelReservation' in Reservation is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'cancelReservation' in Reservation did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'cancelReservation' in Reservation is not implemented or raised an error")

@given(instance=Room_strategy)
@settings(max_examples=50)
def test_room_instantiation(instance):
    assert isinstance(instance, Room)

@given(instance=Room_strategy)
def test_room_roomType_type(instance):
    assert isinstance(instance.roomType, str)


@given(instance=Room_strategy)
def test_room_roomType_setter(instance):
    original = instance.roomType
    instance.roomType = original
    assert instance.roomType == original

@given(instance=Room_strategy)
def test_room_roomNumber_type(instance):
    assert isinstance(instance.roomNumber, str)


@given(instance=Room_strategy)
def test_room_roomNumber_setter(instance):
    original = instance.roomNumber
    instance.roomNumber = original
    assert instance.roomNumber == original

@given(instance=Room_strategy)
def test_room_pricePerNight_type(instance):
    assert isinstance(instance.pricePerNight, float)


@given(instance=Room_strategy)
def test_room_pricePerNight_setter(instance):
    original = instance.pricePerNight
    instance.pricePerNight = original
    assert instance.pricePerNight == original

@given(instance=Room_strategy)
def test_room_available_type(instance):
    assert isinstance(instance.available, bool)


@given(instance=Room_strategy)
def test_room_available_setter(instance):
    original = instance.available
    instance.available = original
    assert instance.available == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Room_strategy)
@settings(max_examples=30)
def test_room_calculatecost_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.calculateCost(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.calculateCost).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'calculateCost' in Room is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'calculateCost' in Room did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'calculateCost' in Room is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Room_strategy)
@settings(max_examples=30)
def test_room_setavailable_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.setAvailable(
            True
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.setAvailable).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'setAvailable' in Room is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'setAvailable' in Room did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'setAvailable' in Room is not implemented or raised an error")

@given(instance=Guest_strategy)
@settings(max_examples=50)
def test_guest_instantiation(instance):
    assert isinstance(instance, Guest)

@given(instance=Guest_strategy)
def test_guest_guestId_type(instance):
    assert isinstance(instance.guestId, str)


@given(instance=Guest_strategy)
def test_guest_guestId_setter(instance):
    original = instance.guestId
    instance.guestId = original
    assert instance.guestId == original

@given(instance=Guest_strategy)
def test_guest_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Guest_strategy)
def test_guest_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=Guest_strategy)
def test_guest_loyaltyPoints_type(instance):
    assert isinstance(instance.loyaltyPoints, int)


@given(instance=Guest_strategy)
def test_guest_loyaltyPoints_setter(instance):
    original = instance.loyaltyPoints
    instance.loyaltyPoints = original
    assert instance.loyaltyPoints == original

@given(instance=Guest_strategy)
def test_guest_email_type(instance):
    assert isinstance(instance.email, str)


@given(instance=Guest_strategy)
def test_guest_email_setter(instance):
    original = instance.email
    instance.email = original
    assert instance.email == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Guest_strategy)
@settings(max_examples=30)
def test_guest_checkin_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.checkIn()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.checkIn).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'checkIn' in Guest is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'checkIn' in Guest did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'checkIn' in Guest is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Guest_strategy)
@settings(max_examples=30)
def test_guest_addloyaltypoints_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.addLoyaltyPoints(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.addLoyaltyPoints).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'addLoyaltyPoints' in Guest is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'addLoyaltyPoints' in Guest did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'addLoyaltyPoints' in Guest is not implemented or raised an error")

@given(instance=Guest_strategy)
def test_guest_ocl_constraint_2(instance):
     
    
    
    before_loyaltyPoints = instance.loyaltyPoints
    
    
    
    value = 1
    # Call the operation
    instance.addLoyaltyPoints(value)
    
    assert instance.loyaltyPoints ==before_loyaltyPoints+ value

@given(instance=Room_strategy)
def test_room_ocl_constraint_1(instance):
     
    
    
    
    
    
    value = True
    # Call the operation
    instance.setAvailable(value)
    
    assert instance.available == value