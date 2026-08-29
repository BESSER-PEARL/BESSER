####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
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
Doctor.attributes={Doctor_consultationFee, Doctor_medicalLicense, Doctor_specialty}

# MedicalAppointment class attributes and methods
MedicalAppointment_appointmentDate: Property = Property(name="appointmentDate", type=DateType)
MedicalAppointment_appointmentId: Property = Property(name="appointmentId", type=StringType)
MedicalAppointment_reason: Property = Property(name="reason", type=StringType)
MedicalAppointment_status: Property = Property(name="status", type=AppointmentStatus)
MedicalAppointment.attributes={MedicalAppointment_appointmentDate, MedicalAppointment_appointmentId, MedicalAppointment_reason, MedicalAppointment_status}

# MedicalRecord class attributes and methods
MedicalRecord_createdDate: Property = Property(name="createdDate", type=DateType)
MedicalRecord_diagnosis: Property = Property(name="diagnosis", type=StringType)
MedicalRecord_prescription: Property = Property(name="prescription", type=StringType)
MedicalRecord_recordId: Property = Property(name="recordId", type=StringType)
MedicalRecord.attributes={MedicalRecord_createdDate, MedicalRecord_diagnosis, MedicalRecord_prescription, MedicalRecord_recordId}

# Patient class attributes and methods
Patient_birthDate: Property = Property(name="birthDate", type=DateType)
Patient_bloodGroup: Property = Property(name="bloodGroup", type=BloodType)
Patient_patientId: Property = Property(name="patientId", type=StringType)
Patient.attributes={Patient_birthDate, Patient_bloodGroup, Patient_patientId}

# Person class attributes and methods
Person_email: Property = Property(name="email", type=StringType)
Person_fullName: Property = Property(name="fullName", type=StringType)
Person_id: Property = Property(name="id", type=StringType)
Person_phone: Property = Property(name="phone", type=StringType)
Person.attributes={Person_email, Person_fullName, Person_id, Person_phone}

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
Patient_Doctor: BinaryAssociation = BinaryAssociation(
    name="Patient_Doctor",
    ends={
        Property(name="patients", type=Patient, multiplicity=Multiplicity(0, 9999)),
        Property(name="primaryDoctor", type=Doctor, multiplicity=Multiplicity(0, 1))
    }
)

# Generalizations
gen_Patient_Person = Generalization(general=Person, specific=Patient)
gen_Doctor_Person = Generalization(general=Person, specific=Doctor)


# OCL Constraints
MedicalRecord_inv_1_1: Constraint = Constraint(
    name="MedicalRecord_inv_1_1",
    context=MedicalRecord,
    expression="context MedicalRecord inv MedicalRecord_inv_1_1 : self.createdDate > self.appointment.appointmentDate",
    language="OCL"
)
Doctor_inv_2_1: Constraint = Constraint(
    name="Doctor_inv_2_1",
    context=Doctor,
    expression="context Doctor inv Doctor_inv_2_1 : self.scheduledAppointments ->size()>2",
    language="OCL"
)
Patient_inv_3_1: Constraint = Constraint(
    name="Patient_inv_3_1",
    context=Patient,
    expression="context Patient inv Patient_inv_3_1 : self.bloodGroup=BloodType::AB_NEGATIVE",
    language="OCL"
)
Patient_inv_4_1: Constraint = Constraint(
    name="Patient_inv_4_1",
    context=Patient,
    expression="context Patient inv Patient_inv_4_1 : Patient.allInstances()->forAll(p| p.bloodGroup=BloodType::AB_NEGATIVE)",
    language="OCL"
)
Patient_inv_5_1: Constraint = Constraint(
    name="Patient_inv_5_1",
    context=Patient,
    expression="context Patient inv Patient_inv_5_1 : (self.primaryDoctor<>null) implies (self.id<>self.primaryDoctor.id)",
    language="OCL"
)
Person_inv_6_1: Constraint = Constraint(
    name="Person_inv_6_1",
    context=Person,
    expression="context Person inv Person_inv_6_1 : Person.allInstances()->forAll(c,p:Person | c.id=p.id   implies c=p)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Doctor, MedicalAppointment, MedicalRecord, Patient, Person, AppointmentStatus, BloodType},
    associations={AppointmentRecord, DoctorAppointments, PatientAppointments, Patient_Doctor},
    constraints={MedicalRecord_inv_1_1, Doctor_inv_2_1, Patient_inv_3_1, Patient_inv_4_1, Patient_inv_5_1, Person_inv_6_1},
    generalizations={gen_Patient_Person, gen_Doctor_Person},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="Hospital Management System BUML Model")
project = Project(
    name="HospitalManagementSystem",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
