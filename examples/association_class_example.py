from besser.BUML.metamodel.structural.structural import (
    Class, Property, BinaryAssociation, AssociationClass, Multiplicity,
    StringType, IntegerType, DomainModel
)

from besser.generators.sql_alchemy.sql_alchemy_generator import SQLAlchemyGenerator

# Create two regular classes
student = Class(name="Student")
course = Class(name="Course")
test = Class(name="Test")

# Create attributes for the Student class
student_id = Property(name="studentId", type=StringType, is_id=True)
student_name = Property(name="name", type=StringType)
student.attributes = {student_id, student_name}

# Create attributes for the Course class
course_id = Property(name="courseId", type=StringType, is_id=True)
course_title = Property(name="title", type=StringType)
course.attributes = {course_id, course_title}

# Create attributes for the test class
test_id = Property(name="testId", type=StringType, is_id=True)
test_name = Property(name="name", type=StringType)

# Create association ends
students_end = Property(
    name="students",
    type=student,
    multiplicity=Multiplicity(0, "*")  # Many students can enroll in a course
)
courses_end = Property(
    name="courses",
    type=course,
    multiplicity=Multiplicity(0, "*")  # A student can enroll in many courses
)

# Create the binary association between Student and Course
enrollment_association = BinaryAssociation(
    name="EnrollmentAsso",
    ends={students_end, courses_end}
)

#Create the ends fpr the test class
test_students_end = Property(
    name="students",
    type=student,
    multiplicity=Multiplicity(0, "*")  # Many students can enroll in a course
)
test__end = Property(
    name="test",
    type=test,
    multiplicity=Multiplicity(0, "*")  # A student can enroll in many courses
)

# Create the binary association between Student and test
enrollment_association_test = BinaryAssociation(
    name="EnrollmentAssoTest",
    ends={test_students_end, test__end}
)

# Create attributes for the association class
grade = Property(name="grade", type=StringType)
semester = Property(name="semester", type=StringType)

# Create the association class
enrollment_class = AssociationClass(
    name="Enrollment",
    attributes={grade, semester},
    association=enrollment_association
)

# Add all elements to a domain model
domain_model = DomainModel(
    name="UniversityModel",
    types={student, course, enrollment_class, test},  # Note: association class is a type
    associations={enrollment_association, enrollment_association_test},  # Add the association to the model
)

x = SQLAlchemyGenerator(domain_model, output_dir="examples")
x.generate()

# Print the model elements to verify
print(f"Model: {domain_model.name}")
print("\nClasses:")
for cls in domain_model.get_classes():
    print(f"- {cls.name}")
    if isinstance(cls, AssociationClass):
        print(f"  (Association Class connected to: {cls.association.name})")
    print(f"  Attributes: {', '.join([attr.name for attr in cls.attributes])}")

print("\nAssociations:")
for assoc in domain_model.associations:
    print(f"- {assoc.name}")
    for end in assoc.ends:
        print(f"  {end.name}: {end.type.name} [{end.multiplicity.min}..{end.multiplicity.max}]")
