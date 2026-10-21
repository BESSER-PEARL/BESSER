"""Academic Management System."""

####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, DomainModel, Multiplicity,
    StringType, IntegerType, DateType, Constraint,
)

# Classes
Exam = Class(name="Exam")
Professor = Class(name="Professor")
Student = Class(name="Student")
Subject = Class(name="Subject")

# Exam class attributes and methods
Exam_examDate: Property = Property(name="examDate", type=DateType)
Exam_examId: Property = Property(name="examId", type=StringType)
Exam_grade: Property = Property(name="grade", type=IntegerType)
Exam.attributes = {Exam_examDate, Exam_examId, Exam_grade}

# Professor class attributes and methods
Professor_department: Property = Property(name="department", type=StringType)
Professor_fullName: Property = Property(name="fullName", type=StringType)
Professor_professorId: Property = Property(name="professorId", type=StringType)
Professor.attributes = {Professor_department, Professor_fullName, Professor_professorId}

# Student class attributes and methods
Student_email: Property = Property(name="email", type=StringType)
Student_fullName: Property = Property(name="fullName", type=StringType)
Student_studentId: Property = Property(name="studentId", type=StringType)
Student_birthDate: Property = Property(name="birthDate", type=DateType)
Student.attributes = {Student_birthDate, Student_email, Student_fullName, Student_studentId}

# Subject class attributes and methods
Subject_credits: Property = Property(name="credits", type=IntegerType)
Subject_name: Property = Property(name="name", type=StringType)
Subject_subjectId: Property = Property(name="subjectId", type=StringType)
Subject.attributes = {Subject_credits, Subject_name, Subject_subjectId}

# Relationships
Enrollment: BinaryAssociation = BinaryAssociation(
    name="Enrollment",
    ends={
        Property(name="students", type=Student, multiplicity=Multiplicity(0, 9999)),
        Property(name="subjects", type=Subject, multiplicity=Multiplicity(1, 9999))
    }
)
StudentExams: BinaryAssociation = BinaryAssociation(
    name="StudentExams",
    ends={
        Property(name="exams", type=Exam, multiplicity=Multiplicity(0, 9999)),
        Property(name="student", type=Student, multiplicity=Multiplicity(1, 1))
    }
)
SubjectExams: BinaryAssociation = BinaryAssociation(
    name="SubjectExams",
    ends={
        Property(name="exams", type=Exam, multiplicity=Multiplicity(0, 9999)),
        Property(name="subject", type=Subject, multiplicity=Multiplicity(1, 1))
    }
)
Teaching: BinaryAssociation = BinaryAssociation(
    name="Teaching",
    ends={
        Property(name="subjectsTaught", type=Subject, multiplicity=Multiplicity(0, 9999)),
        Property(name="teacher", type=Professor, multiplicity=Multiplicity(1, 1))
    }
)


# OCL Constraints
# Date ordering comparisons are not supported by the Alloy translation: this invariant is disabled
# exam_after_birth: Constraint = Constraint(
#     name="exam_after_birth",
#     context=Exam,
#     expression="context Exam inv exam_after_birth: self.examDate > self.student.birthDate",
#     language="OCL"
# )
grade_between_0_and_10: Constraint = Constraint(
    name="grade_between_0_and_10",
    context=Exam,
    expression="context Exam inv grade_between_0_and_10: self.grade >= 0 and self.grade <= 10",
    language="OCL"
)
# Strings are uninterpreted in the Alloy translation, so they have no size: this invariant is disabled
# subject_name_longer_than_4: Constraint = Constraint(
#     name="subject_name_longer_than_4",
#     context=Subject,
#     expression="context Subject inv subject_name_longer_than_4: self.name.size() > 4",
#     language="OCL"
# )
some_exam_graded_10: Constraint = Constraint(
    name="some_exam_graded_10",
    context=Exam,
    expression="context Exam inv some_exam_graded_10: Exam::allInstances()->exists(n | n.grade = 10)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="School",
    types={Exam, Professor, Student, Subject},
    associations={Enrollment, StudentExams, SubjectExams, Teaching},
    constraints={grade_between_0_and_10, some_exam_graded_10},
)
