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

# Classes
InfoExam = Class(name="InfoExam")
Professor = Class(name="Professor")
Student = Class(name="Student")
Subject = Class(name="Subject")

# InfoExam class attributes and methods
InfoExam_examDate: Property = Property(name="examDate", type=DateType)
InfoExam_examId: Property = Property(name="examId", type=StringType)
InfoExam_grade: Property = Property(name="grade", type=IntegerType)
InfoExam.attributes={InfoExam_examDate, InfoExam_examId, InfoExam_grade}

# Professor class attributes and methods
Professor_department: Property = Property(name="department", type=StringType)
Professor_fullName: Property = Property(name="fullName", type=StringType)
Professor_professorId: Property = Property(name="professorId", type=StringType)
Professor.attributes={Professor_department, Professor_fullName, Professor_professorId}

# Student class attributes and methods
Student_email: Property = Property(name="email", type=StringType)
Student_fullName: Property = Property(name="fullName", type=StringType)
Student_studentId: Property = Property(name="studentId", type=StringType)
Student_birthDate: Property = Property(name="birthDate", type=DateType)
Student.attributes={Student_birthDate, Student_email, Student_fullName, Student_studentId}

# Subject class attributes and methods
Subject_credits: Property = Property(name="credits", type=IntegerType)
Subject_name: Property = Property(name="name", type=StringType)
Subject_subjectId: Property = Property(name="subjectId", type=StringType)
Subject.attributes={Subject_credits, Subject_name, Subject_subjectId}

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
        Property(name="exams", type=InfoExam, multiplicity=Multiplicity(0, 9999)),
        Property(name="student", type=Student, multiplicity=Multiplicity(1, 1))
    }
)
SubjectExams: BinaryAssociation = BinaryAssociation(
    name="SubjectExams",
    ends={
        Property(name="exams", type=InfoExam, multiplicity=Multiplicity(0, 9999)),
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
InfoExam_inv_2_1: Constraint = Constraint(
    name="InfoExam_inv_2_1",
    context=InfoExam,
    expression="context InfoExam inv InfoExam_inv_2_1 : self.examDate>self.student.birthDate",
    language="OCL"
)
Exam_inv_grade_range: Constraint = Constraint(
    name="Exam_inv_grade_range",
    context=InfoExam,
    expression="context InfoExam inv Exam_inv_grade_range : self.grade >= 0 and self.grade <= 10",
    language="OCL"
)
InfoExam_inv_3_1: Constraint = Constraint(
    name="InfoExam_inv_3_1",
    context=InfoExam,
    expression="context InfoExam inv InfoExam_inv_3_1 : self.examDate>self.student.birthDate",
    language="OCL"
)
Subject_inv_4_1: Constraint = Constraint(
    name="Subject_inv_4_1",
    context=Subject,
    expression="context Subject inv Subject_inv_4_1 : self.name.size()>4",
    language="OCL"
)
InfoExam_inv_5_1: Constraint = Constraint(
    name="InfoExam_inv_5_1",
    context=InfoExam,
    expression="context InfoExam inv InfoExam_inv_5_1 : InfoExam::allInstances()->exists(n | n.grade=10)",
    language="OCL"
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={InfoExam, Professor, Student, Subject},
    associations={Enrollment, StudentExams, SubjectExams, Teaching},
    constraints={InfoExam_inv_2_1, Exam_inv_grade_range, InfoExam_inv_3_1, Subject_inv_4_1, InfoExam_inv_5_1},
    generalizations={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="Academic Management System BUML Model")
project = Project(
    name="AcademicManagementSystem",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)
