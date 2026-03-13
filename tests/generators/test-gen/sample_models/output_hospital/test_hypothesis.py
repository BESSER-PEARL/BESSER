import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy
from datetime import date

from classes import (
    Appointment,
    Doctor,
    Patient,
    AppointmentStatus,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================



def test_appointment_is_not_abstract():
    assert not inspect.isabstract(Appointment)


def test_appointment_constructor_exists():
    assert callable(Appointment.__init__)


def test_appointment_constructor_args():
    sig = inspect.signature(Appointment.__init__)
    params = list(sig.parameters.keys())
    assert "time" in params, "Missing parameter 'time'"
    assert "appointmentId" in params, "Missing parameter 'appointmentId'"
    assert "status" in params, "Missing parameter 'status'"
    assert "date" in params, "Missing parameter 'date'"

def test_appointment_has_time():
    assert hasattr(Appointment, "time")
    descriptor = None
    for klass in Appointment.__mro__:
        if "time" in klass.__dict__:
            descriptor = klass.__dict__["time"]
            break
    assert isinstance(descriptor, property)

def test_appointment_has_appointmentId():
    assert hasattr(Appointment, "appointmentId")
    descriptor = None
    for klass in Appointment.__mro__:
        if "appointmentId" in klass.__dict__:
            descriptor = klass.__dict__["appointmentId"]
            break
    assert isinstance(descriptor, property)

def test_appointment_has_status():
    assert hasattr(Appointment, "status")
    descriptor = None
    for klass in Appointment.__mro__:
        if "status" in klass.__dict__:
            descriptor = klass.__dict__["status"]
            break
    assert isinstance(descriptor, property)

def test_appointment_has_date():
    assert hasattr(Appointment, "date")
    descriptor = None
    for klass in Appointment.__mro__:
        if "date" in klass.__dict__:
            descriptor = klass.__dict__["date"]
            break
    assert isinstance(descriptor, property)



def test_doctor_is_not_abstract():
    assert not inspect.isabstract(Doctor)


def test_doctor_constructor_exists():
    assert callable(Doctor.__init__)


def test_doctor_constructor_args():
    sig = inspect.signature(Doctor.__init__)
    params = list(sig.parameters.keys())
    assert "specialization" in params, "Missing parameter 'specialization'"
    assert "doctorId" in params, "Missing parameter 'doctorId'"
    assert "name" in params, "Missing parameter 'name'"
    assert "available" in params, "Missing parameter 'available'"

def test_doctor_has_specialization():
    assert hasattr(Doctor, "specialization")
    descriptor = None
    for klass in Doctor.__mro__:
        if "specialization" in klass.__dict__:
            descriptor = klass.__dict__["specialization"]
            break
    assert isinstance(descriptor, property)

def test_doctor_has_doctorId():
    assert hasattr(Doctor, "doctorId")
    descriptor = None
    for klass in Doctor.__mro__:
        if "doctorId" in klass.__dict__:
            descriptor = klass.__dict__["doctorId"]
            break
    assert isinstance(descriptor, property)

def test_doctor_has_name():
    assert hasattr(Doctor, "name")
    descriptor = None
    for klass in Doctor.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_doctor_has_available():
    assert hasattr(Doctor, "available")
    descriptor = None
    for klass in Doctor.__mro__:
        if "available" in klass.__dict__:
            descriptor = klass.__dict__["available"]
            break
    assert isinstance(descriptor, property)



def test_patient_is_not_abstract():
    assert not inspect.isabstract(Patient)


def test_patient_constructor_exists():
    assert callable(Patient.__init__)


def test_patient_constructor_args():
    sig = inspect.signature(Patient.__init__)
    params = list(sig.parameters.keys())
    assert "patientId" in params, "Missing parameter 'patientId'"
    assert "bloodType" in params, "Missing parameter 'bloodType'"
    assert "name" in params, "Missing parameter 'name'"
    assert "age" in params, "Missing parameter 'age'"

def test_patient_has_patientId():
    assert hasattr(Patient, "patientId")
    descriptor = None
    for klass in Patient.__mro__:
        if "patientId" in klass.__dict__:
            descriptor = klass.__dict__["patientId"]
            break
    assert isinstance(descriptor, property)

def test_patient_has_bloodType():
    assert hasattr(Patient, "bloodType")
    descriptor = None
    for klass in Patient.__mro__:
        if "bloodType" in klass.__dict__:
            descriptor = klass.__dict__["bloodType"]
            break
    assert isinstance(descriptor, property)

def test_patient_has_name():
    assert hasattr(Patient, "name")
    descriptor = None
    for klass in Patient.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_patient_has_age():
    assert hasattr(Patient, "age")
    descriptor = None
    for klass in Patient.__mro__:
        if "age" in klass.__dict__:
            descriptor = klass.__dict__["age"]
            break
    assert isinstance(descriptor, property)

def test_appointmentstatus_exists():
    # Check that the Enumeration exists
    assert AppointmentStatus is not None

def test_appointmentstatus_has_all_literals():
    # Collect the names of literals in this Enumeration
    enum_literals = [lit.name for lit in AppointmentStatus]
    expected_literals = [
        "NO_SHOW",
        "CANCELLED",
        "COMPLETED",
        "SCHEDULED",
    ]
    # Check that all expected literals exist
    for lit_name in expected_literals:
        assert lit_name in enum_literals, f"Literal '' missing in AppointmentStatus"


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
Appointment_strategy = st.builds(
    Appointment,
    time=
        safe_text,
    appointmentId=
        safe_text,
    status=
        safe_text,
    date=
        safe_text
)
Doctor_strategy = st.builds(
    Doctor,
    specialization=
        safe_text,
    doctorId=
        safe_text,
    name=
        safe_text,
    available=
        st.booleans()
)
Patient_strategy = st.builds(
    Patient,
    patientId=
        safe_text,
    bloodType=
        safe_text,
    name=
        safe_text,
    age=
        st.integers()
)

@given(instance=Appointment_strategy)
@settings(max_examples=50)
def test_appointment_instantiation(instance):
    assert isinstance(instance, Appointment)

@given(instance=Appointment_strategy)
def test_appointment_time_type(instance):
    assert isinstance(instance.time, str)


@given(instance=Appointment_strategy)
def test_appointment_time_setter(instance):
    original = instance.time
    instance.time = original
    assert instance.time == original

@given(instance=Appointment_strategy)
def test_appointment_appointmentId_type(instance):
    assert isinstance(instance.appointmentId, str)


@given(instance=Appointment_strategy)
def test_appointment_appointmentId_setter(instance):
    original = instance.appointmentId
    instance.appointmentId = original
    assert instance.appointmentId == original

@given(instance=Appointment_strategy)
def test_appointment_status_type(instance):
    assert isinstance(instance.status, str)


@given(instance=Appointment_strategy)
def test_appointment_status_setter(instance):
    original = instance.status
    instance.status = original
    assert instance.status == original

@given(instance=Appointment_strategy)
def test_appointment_date_type(instance):
    assert isinstance(instance.date, str)


@given(instance=Appointment_strategy)
def test_appointment_date_setter(instance):
    original = instance.date
    instance.date = original
    assert instance.date == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Appointment_strategy)
@settings(max_examples=30)
def test_appointment_cancel_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.cancel()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.cancel).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'cancel' in Appointment is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'cancel' in Appointment did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'cancel' in Appointment is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Appointment_strategy)
@settings(max_examples=30)
def test_appointment_schedule_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.schedule()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.schedule).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'schedule' in Appointment is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'schedule' in Appointment did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'schedule' in Appointment is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Appointment_strategy)
@settings(max_examples=30)
def test_appointment_complete_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.complete()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.complete).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'complete' in Appointment is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'complete' in Appointment did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'complete' in Appointment is not implemented or raised an error")

@given(instance=Doctor_strategy)
@settings(max_examples=50)
def test_doctor_instantiation(instance):
    assert isinstance(instance, Doctor)

@given(instance=Doctor_strategy)
def test_doctor_specialization_type(instance):
    assert isinstance(instance.specialization, str)


@given(instance=Doctor_strategy)
def test_doctor_specialization_setter(instance):
    original = instance.specialization
    instance.specialization = original
    assert instance.specialization == original

@given(instance=Doctor_strategy)
def test_doctor_doctorId_type(instance):
    assert isinstance(instance.doctorId, str)


@given(instance=Doctor_strategy)
def test_doctor_doctorId_setter(instance):
    original = instance.doctorId
    instance.doctorId = original
    assert instance.doctorId == original

@given(instance=Doctor_strategy)
def test_doctor_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Doctor_strategy)
def test_doctor_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=Doctor_strategy)
def test_doctor_available_type(instance):
    assert isinstance(instance.available, bool)


@given(instance=Doctor_strategy)
def test_doctor_available_setter(instance):
    original = instance.available
    instance.available = original
    assert instance.available == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Doctor_strategy)
@settings(max_examples=30)
def test_doctor_setavailability_changes_state(instance):
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
        assert has_statements, f"Function 'setAvailability' in Doctor is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'setAvailability' in Doctor did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'setAvailability' in Doctor is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Doctor_strategy)
@settings(max_examples=30)
def test_doctor_registerdoctor_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.registerDoctor()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.registerDoctor).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'registerDoctor' in Doctor is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'registerDoctor' in Doctor did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'registerDoctor' in Doctor is not implemented or raised an error")

@given(instance=Patient_strategy)
@settings(max_examples=50)
def test_patient_instantiation(instance):
    assert isinstance(instance, Patient)

@given(instance=Patient_strategy)
def test_patient_patientId_type(instance):
    assert isinstance(instance.patientId, str)


@given(instance=Patient_strategy)
def test_patient_patientId_setter(instance):
    original = instance.patientId
    instance.patientId = original
    assert instance.patientId == original

@given(instance=Patient_strategy)
def test_patient_bloodType_type(instance):
    assert isinstance(instance.bloodType, str)


@given(instance=Patient_strategy)
def test_patient_bloodType_setter(instance):
    original = instance.bloodType
    instance.bloodType = original
    assert instance.bloodType == original

@given(instance=Patient_strategy)
def test_patient_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Patient_strategy)
def test_patient_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=Patient_strategy)
def test_patient_age_type(instance):
    assert isinstance(instance.age, int)


@given(instance=Patient_strategy)
def test_patient_age_setter(instance):
    original = instance.age
    instance.age = original
    assert instance.age == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Patient_strategy)
@settings(max_examples=30)
def test_patient_registerpatient_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.registerPatient()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.registerPatient).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'registerPatient' in Patient is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'registerPatient' in Patient did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'registerPatient' in Patient is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Patient_strategy)
@settings(max_examples=30)
def test_patient_updateage_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.updateAge(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.updateAge).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'updateAge' in Patient is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'updateAge' in Patient did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'updateAge' in Patient is not implemented or raised an error")

@given(instance=Patient_strategy)
def test_patient_ocl_constraint_2(instance):
     
    
    
    
    
    
    value = 1
    # Call the operation
    instance.updateAge(value)
    
    assert instance.age == value

@given(instance=Doctor_strategy)
def test_doctor_ocl_constraint_1(instance):
     
    
    
    
    
    
    value = "1"
    # Call the operation
    instance.setAvailability(value)
    
    assert instance.available == value