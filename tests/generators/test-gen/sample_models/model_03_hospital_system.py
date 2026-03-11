"""
BUML Model Example 3: Hospital Management System
A simple hospital system with patients, doctors, and appointments
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
scheduled_lit = EnumerationLiteral(name="SCHEDULED")
completed_lit = EnumerationLiteral(name="COMPLETED")
cancelled_lit = EnumerationLiteral(name="CANCELLED")
no_show_lit = EnumerationLiteral(name="NO_SHOW")

appointment_status_enum = Enumeration(
    name="AppointmentStatus",
    literals={scheduled_lit, completed_lit, cancelled_lit, no_show_lit}
)

# =============================================================================
# 2. Define Patient Attributes
# =============================================================================
patient_id_prop = Property(name="patientId", type=StringType, multiplicity=Multiplicity(1, 1))
patient_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
age_prop = Property(name="age", type=IntegerType, multiplicity=Multiplicity(1, 1))
blood_type_prop = Property(name="bloodType", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Doctor Attributes
# =============================================================================
doctor_id_prop = Property(name="doctorId", type=StringType, multiplicity=Multiplicity(1, 1))
doctor_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
specialization_prop = Property(name="specialization", type=StringType, multiplicity=Multiplicity(1, 1))
available_prop = Property(name="available", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Appointment Attributes
# =============================================================================
appointment_id_prop = Property(name="appointmentId", type=StringType, multiplicity=Multiplicity(1, 1))
date_prop = Property(name="date", type=StringType, multiplicity=Multiplicity(1, 1))
time_prop = Property(name="time", type=StringType, multiplicity=Multiplicity(1, 1))
status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Patient Methods
# =============================================================================
register_patient_method = Method(
    name="registerPatient",
    parameters=[],
    code="""
def registerPatient(self):
    print(f"Patient registered: {self.name}")
    print(f"ID: {self.patientId}, Age: {self.age}, Blood Type: {self.bloodType}")
"""
)

get_medical_info_method = Method(
    name="getMedicalInfo",
    parameters=[],
    code="""
def getMedicalInfo(self):
    info = f"Patient: {self.name}\\n"
    info += f"ID: {self.patientId}\\n"
    info += f"Age: {self.age}\\n"
    info += f"Blood Type: {self.bloodType}"
    return info
"""
)

update_age_method = Method(
    name="updateAge",
    parameters=[Parameter(name="newAge", type=IntegerType)],
    code="""
def updateAge(self, newAge):
    self.age = newAge
    print(f"Patient {self.name}'s age updated to {newAge}")
"""
)

# =============================================================================
# 6. Define Doctor Methods
# =============================================================================
register_doctor_method = Method(
    name="registerDoctor",
    parameters=[],
    code="""
def registerDoctor(self):
    self.available = True
    print(f"Doctor registered: Dr. {self.name}")
    print(f"ID: {self.doctorId}, Specialization: {self.specialization}")
"""
)

set_availability_method = Method(
    name="setAvailability",
    parameters=[Parameter(name="status", type=BooleanType)],
    code="""
def setAvailability(self, status):
    self.available = status
    status_text = "available" if status else "unavailable"
    print(f"Dr. {self.name} is now {status_text}")
"""
)

get_doctor_info_method = Method(
    name="getDoctorInfo",
    parameters=[],
    code="""
def getDoctorInfo(self):
    availability = "Available" if self.available else "Not Available"
    return f"Dr. {self.name} - {self.specialization} ({availability})"
"""
)

# =============================================================================
# 7. Define Appointment Methods
# =============================================================================
schedule_method = Method(
    name="schedule",
    parameters=[],
    code="""
def schedule(self):
    self.status = "Scheduled"
    print(f"Appointment {self.appointmentId} scheduled")
    print(f"Date: {self.date}, Time: {self.time}")
"""
)

cancel_method = Method(
    name="cancel",
    parameters=[],
    code="""
def cancel(self):
    self.status = "Cancelled"
    print(f"Appointment {self.appointmentId} has been cancelled")
"""
)

complete_method = Method(
    name="complete",
    parameters=[],
    code="""
def complete(self):
    self.status = "Completed"
    print(f"Appointment {self.appointmentId} marked as completed")
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
patient_class = Class(
    name="Patient",
    attributes={patient_id_prop, patient_name_prop, age_prop, blood_type_prop},
    methods={register_patient_method, get_medical_info_method, update_age_method}
)

doctor_class = Class(
    name="Doctor",
    attributes={doctor_id_prop, doctor_name_prop, specialization_prop, available_prop},
    methods={register_doctor_method, set_availability_method, get_doctor_info_method}
)

appointment_class = Class(
    name="Appointment",
    attributes={appointment_id_prop, date_prop, time_prop, status_prop},
    methods={schedule_method, cancel_method, complete_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# Patient --< Appointment (one patient can have many appointments)
patient_end = Property(
    name="patient",
    type=patient_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
patient_appointments_end = Property(
    name="appointments",
    type=appointment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
patient_appointment_assoc = BinaryAssociation(
    name="Has",
    ends={patient_end, patient_appointments_end}
)

# Doctor --< Appointment (one doctor can have many appointments)
doctor_end = Property(
    name="doctor",
    type=doctor_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
doctor_appointments_end = Property(
    name="schedule",
    type=appointment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
doctor_appointment_assoc = BinaryAssociation(
    name="Attends",
    ends={doctor_end, doctor_appointments_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
hospital_model = DomainModel(
    name="HospitalManagementSystem",
    types={patient_class, doctor_class, appointment_class, appointment_status_enum},
    associations={patient_appointment_assoc, doctor_appointment_assoc}
)

print("✓ Hospital Management System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in hospital_model.get_classes()]}")
print(f"  Associations: {[a.name for a in hospital_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=hospital_model, output_dir="output_hospital")
python_gen.generate()