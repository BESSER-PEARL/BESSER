"""Hospital Management System."""

####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, FloatType, DateType, Constraint,
)

# Enumerations
AppointmentStatus: Enumeration = Enumeration(
    name="AppointmentStatus",
    literals={
        EnumerationLiteral(name="CANCELLED"),
        EnumerationLiteral(name="COMPLETED"),
        EnumerationLiteral(name="NO_SHOW"),
        EnumerationLiteral(name="SCHEDULED")
    }
)

BloodType: Enumeration = Enumeration(
    name="BloodType",
    literals={
        EnumerationLiteral(name="AB_NEGATIVE"),
        EnumerationLiteral(name="AB_POSITIVE"),
        EnumerationLiteral(name="A_NEGATIVE"),
        EnumerationLiteral(name="A_POSITIVE"),
        EnumerationLiteral(name="B_NEGATIVE"),
        EnumerationLiteral(name="B_POSITIVE"),
        EnumerationLiteral(name="O_NEGATIVE"),
        EnumerationLiteral(name="O_POSITIVE")
    }
)

# Classes
Doctor = Class(name="Doctor")
MedicalAppointment = Class(name="MedicalAppointment")
MedicalRecord = Class(name="MedicalRecord")
Patient = Class(name="Patient")
Person = Class(name="Person", is_abstract=True)

# Doctor class attributes and methods
Doctor_consultationFee: Property = Property(name="consultationFee", type=FloatType)
Doctor_medicalLicense: Property = Property(name="medicalLicense", type=StringType)
Doctor_specialty: Property = Property(name="specialty", type=StringType)
Doctor.attributes = {Doctor_consultationFee, Doctor_medicalLicense, Doctor_specialty}

# MedicalAppointment class attributes and methods
MedicalAppointment_appointmentDate: Property = Property(name="appointmentDate", type=DateType)
MedicalAppointment_appointmentId: Property = Property(name="appointmentId", type=StringType)
MedicalAppointment_reason: Property = Property(name="reason", type=StringType)
MedicalAppointment_status: Property = Property(name="status", type=AppointmentStatus)
MedicalAppointment.attributes = {
    MedicalAppointment_appointmentDate, MedicalAppointment_appointmentId, MedicalAppointment_reason,
    MedicalAppointment_status,
}

# MedicalRecord class attributes and methods
MedicalRecord_createdDate: Property = Property(name="createdDate", type=DateType)
MedicalRecord_diagnosis: Property = Property(name="diagnosis", type=StringType)
MedicalRecord_prescription: Property = Property(name="prescription", type=StringType)
MedicalRecord_recordId: Property = Property(name="recordId", type=StringType)
MedicalRecord.attributes = {
    MedicalRecord_createdDate, MedicalRecord_diagnosis, MedicalRecord_prescription, MedicalRecord_recordId,
}

# Patient class attributes and methods
Patient_birthDate: Property = Property(name="birthDate", type=DateType)
Patient_bloodGroup: Property = Property(name="bloodGroup", type=BloodType)
Patient_patientId: Property = Property(name="patientId", type=StringType)
Patient.attributes = {Patient_birthDate, Patient_bloodGroup, Patient_patientId}

# Person class attributes and methods
Person_email: Property = Property(name="email", type=StringType)
Person_fullName: Property = Property(name="fullName", type=StringType)
Person_id: Property = Property(name="id", type=StringType)
Person_phone: Property = Property(name="phone", type=StringType)
Person.attributes = {Person_email, Person_fullName, Person_id, Person_phone}

# Relationships
AppointmentRecord: BinaryAssociation = BinaryAssociation(
    name="AppointmentRecord",
    ends={
        Property(name="appointment", type=MedicalAppointment, multiplicity=Multiplicity(1, 1)),
        Property(name="record", type=MedicalRecord, multiplicity=Multiplicity(0, 1))
    }
)
DoctorAppointments: BinaryAssociation = BinaryAssociation(
    name="DoctorAppointments",
    ends={
        Property(name="attendingDoctor", type=Doctor, multiplicity=Multiplicity(1, 1)),
        Property(name="scheduledAppointments", type=MedicalAppointment, multiplicity=Multiplicity(0, 9999))
    }
)
PatientAppointments: BinaryAssociation = BinaryAssociation(
    name="PatientAppointments",
    ends={
        Property(name="appointments", type=MedicalAppointment, multiplicity=Multiplicity(0, 9999)),
        Property(name="patient", type=Patient, multiplicity=Multiplicity(1, 1))
    }
)
PrimaryDoctor: BinaryAssociation = BinaryAssociation(
    name="PrimaryDoctor",
    ends={
        Property(name="patients", type=Patient, multiplicity=Multiplicity(0, 9999)),
        Property(name="primaryDoctor", type=Doctor, multiplicity=Multiplicity(0, 1))
    }
)

# Generalizations
gen_Doctor_Person = Generalization(general=Person, specific=Doctor)
gen_Patient_Person = Generalization(general=Person, specific=Patient)


# OCL Constraints
doctor_has_more_than_two_appointments: Constraint = Constraint(
    name="doctor_has_more_than_two_appointments",
    context=Doctor,
    expression="context Doctor inv doctor_has_more_than_two_appointments: self.scheduledAppointments->size() > 2",
    language="OCL"
)
# Date ordering comparisons are not supported by the Alloy translation: this invariant is disabled
# record_created_after_appointment: Constraint = Constraint(
#     name="record_created_after_appointment",
#     context=MedicalRecord,
#     expression="context MedicalRecord inv record_created_after_appointment: "
#                "self.createdDate > self.appointment.appointmentDate",
#     language="OCL"
# )
patient_is_not_own_primary_doctor: Constraint = Constraint(
    name="patient_is_not_own_primary_doctor",
    context=Patient,
    expression="context Patient inv patient_is_not_own_primary_doctor: "
               "(self.primaryDoctor <> null) implies (self.id <> self.primaryDoctor.id)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Hospital",
    types={Doctor, MedicalAppointment, MedicalRecord, Patient, Person, AppointmentStatus, BloodType},
    associations={AppointmentRecord, DoctorAppointments, PatientAppointments, PrimaryDoctor},
    constraints={doctor_has_more_than_two_appointments, patient_is_not_own_primary_doctor},
    generalizations={gen_Doctor_Person, gen_Patient_Person},
)
