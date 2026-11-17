import pytest
from tests.generators.alloy.BUML_school_model import school_model  # noqa: F401
from tests.generators.alloy.utils_alloy_translation import translate


@pytest.fixture
def spec(school_model):  # noqa: F811
    return translate(school_model, scope=5)


def test_library_modules(spec):
    # Date and String attributes require the date and strings libraries
    assert "date" in spec.opens
    assert "strings" in spec.opens
    assert len(spec.opens) == 2


def test_signatures(spec):
    assert "sig Student" in spec.sigs
    assert "sig Subject" in spec.sigs
    assert "sig Professor" in spec.sigs
    assert "sig Exam" in spec.sigs
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
    assert len(spec.fields["Student"]) == 6
    assert "Student_birthDate: Date" in spec.fields["Student"]
    assert "Student_email: Str" in spec.fields["Student"]
    assert "Student_fullName: Str" in spec.fields["Student"]
    assert "Student_studentId: Str" in spec.fields["Student"]
    assert "Student_subjects: set Subject" in spec.fields["Student"]
    assert "Student_exams: set Exam" in spec.fields["Student"]

    assert len(spec.fields["Subject"]) == 6
    assert "Subject_credits: Int" in spec.fields["Subject"]
    assert "Subject_name: Str" in spec.fields["Subject"]
    assert "Subject_subjectId: Str" in spec.fields["Subject"]
    assert "Subject_students: set Student" in spec.fields["Subject"]
    assert "Subject_exams: set Exam" in spec.fields["Subject"]
    assert "Subject_teacher: one Professor" in spec.fields["Subject"]

    assert len(spec.fields["Professor"]) == 4
    assert "Professor_department: Str" in spec.fields["Professor"]
    assert "Professor_fullName: Str" in spec.fields["Professor"]
    assert "Professor_professorId: Str" in spec.fields["Professor"]
    assert "Professor_subjectsTaught: set Subject" in spec.fields["Professor"]

    assert len(spec.fields["Exam"]) == 5
    assert "Exam_examDate: Date" in spec.fields["Exam"]
    assert "Exam_examId: Str" in spec.fields["Exam"]
    assert "Exam_grade: Int" in spec.fields["Exam"]
    assert "Exam_student: one Student" in spec.fields["Exam"]
    assert "Exam_subject: one Subject" in spec.fields["Exam"]


def test_facts(spec):
    assert len(spec.facts) == 8

    # The dates are ordered chronologically
    assert "dateLt[Date_2001_10_25, Date_2017_04_12] and dateLt[Date_2017_04_12, Date_2018_11_12] " \
        "and dateLt[Date_2018_11_12, Date_2033_12_31] and dateLt[Date_2033_12_31, Date_2037_11_17]" in spec.facts

    # Associations navigable both ways: each field is the transpose of the opposite one
    assert "Subject_students = ~Student_subjects" in spec.facts
    assert "Student_exams = ~Exam_student" in spec.facts
    assert "Subject_exams = ~Exam_subject" in spec.facts
    assert "Professor_subjectsTaught = ~Subject_teacher" in spec.facts
    # Student -> Subject (1..*)
    assert "all a: Student | #(a.Student_subjects) >= 1" in spec.facts

    # OCL: self.grade >= 0 and self.grade <= 10
    assert "all self:this/Exam|((self.Exam_grade >= 0) and (self.Exam_grade <= 10))" in spec.facts
    # OCL: some exam is graded 10 (Exam::allInstances()->exists(n | n.grade = 10))
    assert "all self:this/Exam|some n : Exam | (n.Exam_grade = 10)" in spec.facts


def test_run_command(spec):
    assert spec.run_pred == "instance_model"
    assert "5 Student" in spec.run_scopes
    assert "5 Subject" in spec.run_scopes
    assert "5 Professor" in spec.run_scopes
    assert "5 Exam" in spec.run_scopes
    assert "5 Str" in spec.run_scopes
    assert "5 Date" in spec.run_scopes
    assert "5 Int" in spec.run_scopes
