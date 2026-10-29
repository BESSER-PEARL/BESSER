import pytest
from tests.generators.alloy.BUML_hospital_model import hospital_model  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def spec(hospital_model):  # noqa: F811
    return translate(hospital_model, scope=5)


def test_library_modules(spec):
    # Date and String attributes require the date and strings libraries
    assert "date" in spec.opens
    assert "strings" in spec.opens
    assert len(spec.opens) == 2


def test_signatures(spec):
    assert "abstract sig Person" in spec.sigs
    assert "sig Doctor extends Person" in spec.sigs
    assert "sig Patient extends Person" in spec.sigs
    assert "sig MedicalAppointment" in spec.sigs
    assert "sig MedicalRecord" in spec.sigs
    # 5 strings filling the scope, as there are no OCL string literals
    assert len(spec.subsigs_of("Str")) == 5
    assert "one sig Str_str0 extends Str" in spec.sigs
    assert "one sig Str_str1 extends Str" in spec.sigs
    assert "one sig Str_str2 extends Str" in spec.sigs
    assert "one sig Str_str3 extends Str" in spec.sigs
    assert "one sig Str_str4 extends Str" in spec.sigs
    # 5 dates, all random (seed 0) as there are no OCL date literals, sorted chronologically
    assert len(spec.subsigs_of("Date")) == 5
    assert "one sig Date_2001_10_25 extends Date" in spec.sigs
    assert "one sig Date_2017_04_12 extends Date" in spec.sigs
    assert "one sig Date_2018_11_12 extends Date" in spec.sigs
    assert "one sig Date_2033_12_31 extends Date" in spec.sigs
    assert "one sig Date_2037_11_17 extends Date" in spec.sigs


def test_signature_fields(spec):
    assert len(spec.fields["Person"]) == 4
    assert "Person_id: Str" in spec.fields["Person"]
    assert "Person_fullName: Str" in spec.fields["Person"]
    assert "Person_email: Str" in spec.fields["Person"]
    assert "Person_phone: Str" in spec.fields["Person"]

    assert len(spec.fields["Doctor"]) == 5
    assert "Doctor_consultationFee: Int" in spec.fields["Doctor"]
    assert "Doctor_medicalLicense: Str" in spec.fields["Doctor"]
    assert "Doctor_specialty: Str" in spec.fields["Doctor"]
    assert "Doctor_patients: set Patient" in spec.fields["Doctor"]
    assert "Doctor_scheduledAppointments: set MedicalAppointment" in spec.fields["Doctor"]

    assert len(spec.fields["Patient"]) == 5
    assert "Patient_birthDate: Date" in spec.fields["Patient"]
    assert "Patient_bloodGroup: BloodType" in spec.fields["Patient"]
    assert "Patient_patientId: Str" in spec.fields["Patient"]
    assert "Patient_primaryDoctor: lone Doctor" in spec.fields["Patient"]
    assert "Patient_appointments: set MedicalAppointment" in spec.fields["Patient"]

    assert len(spec.fields["MedicalAppointment"]) == 7
    assert "MedicalAppointment_appointmentDate: Date" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_appointmentId: Str" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_reason: Str" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_status: AppointmentStatus" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_attendingDoctor: one Doctor" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_patient: one Patient" in spec.fields["MedicalAppointment"]
    assert "MedicalAppointment_record: lone MedicalRecord" in spec.fields["MedicalAppointment"]

    assert len(spec.fields["MedicalRecord"]) == 5
    assert "MedicalRecord_createdDate: Date" in spec.fields["MedicalRecord"]
    assert "MedicalRecord_diagnosis: Str" in spec.fields["MedicalRecord"]
    assert "MedicalRecord_prescription: Str" in spec.fields["MedicalRecord"]
    assert "MedicalRecord_recordId: Str" in spec.fields["MedicalRecord"]
    assert "MedicalRecord_appointment: one MedicalAppointment" in spec.fields["MedicalRecord"]


def test_enumerations(spec):
    assert len(spec.subsigs_of("AppointmentStatus")) == 4
    assert "abstract sig AppointmentStatus" in spec.sigs
    assert "one sig ENUM_AppointmentStatus_CANCELLED extends AppointmentStatus" in spec.sigs
    assert "one sig ENUM_AppointmentStatus_COMPLETED extends AppointmentStatus" in spec.sigs
    assert "one sig ENUM_AppointmentStatus_NO_SHOW extends AppointmentStatus" in spec.sigs
    assert "one sig ENUM_AppointmentStatus_SCHEDULED extends AppointmentStatus" in spec.sigs

    assert len(spec.subsigs_of("BloodType")) == 8
    assert "abstract sig BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_A_POSITIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_A_NEGATIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_B_POSITIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_B_NEGATIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_AB_POSITIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_AB_NEGATIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_O_POSITIVE extends BloodType" in spec.sigs
    assert "one sig ENUM_BloodType_O_NEGATIVE extends BloodType" in spec.sigs


def test_facts(spec):
    assert len(spec.facts) == 9

    # The dates are ordered chronologically
    assert "dateLt[Date_2001_10_25, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2018_11_12] " \
        "and dateLt[Date_2018_11_12, Date_2033_12_31] and dateLt[Date_2033_12_31, Date_2037_11_17]" in spec.facts

    # Associations navigable both ways: each field is the transpose of the opposite one
    assert "Doctor_patients = ~Patient_primaryDoctor" in spec.facts
    assert "MedicalAppointment_attendingDoctor = ~Doctor_scheduledAppointments" in spec.facts
    assert "Patient_appointments = ~MedicalAppointment_patient" in spec.facts
    assert "MedicalRecord_appointment = ~MedicalAppointment_record" in spec.facts
    # Patient -> Doctor, primary doctor (0..1), and MedicalAppointment -> MedicalRecord (0..1)
    assert "all a: Patient | #(a.Patient_primaryDoctor) <= 1" in spec.facts
    assert "all a: MedicalAppointment | #(a.MedicalAppointment_record) <= 1" in spec.facts

    # OCL: a doctor has more than two appointments (self.scheduledAppointments->size() > 2)
    assert "all self:this/Doctor|(#(self.Doctor_scheduledAppointments) > 2)" in spec.facts
    # OCL: a patient is not their own primary doctor
    # ((self.primaryDoctor <> null) implies (self.id <> self.primaryDoctor.id))
    assert "all self:this/Patient|((some self.Patient_primaryDoctor) => " \
        "(strNe[self.Person_id,self.Patient_primaryDoctor.Person_id]))" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Person" in spec.run_scopes
    assert "5 Doctor" in spec.run_scopes
    assert "5 Patient" in spec.run_scopes
    assert "5 MedicalAppointment" in spec.run_scopes
    assert "5 MedicalRecord" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
    assert "5 Date" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
