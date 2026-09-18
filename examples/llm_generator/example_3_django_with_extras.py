"""
Example 3: Generate a Django project with custom extras.

The LLM will:
1. Call generate_django() to get the base Django project
2. Read the generated code
3. Add Django REST Framework for API endpoints
4. Add custom model managers and querysets
5. Add admin customization (list display, filters, search)
6. Add unit tests
7. Verify everything works

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python example_3_django_with_extras.py
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
    BinaryAssociation, Multiplicity, UNLIMITED_MAX_MULTIPLICITY,
    Enumeration, EnumerationLiteral,
)
from besser.generators.llm import LLMGenerator

# ── University management model ──────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")
FloatType = PrimitiveDataType("float")
BooleanType = PrimitiveDataType("bool")
DateType = PrimitiveDataType("date")

semester = Enumeration(name="Semester", literals={
    EnumerationLiteral(name="FALL"),
    EnumerationLiteral(name="SPRING"),
    EnumerationLiteral(name="SUMMER"),
})

grade = Enumeration(name="Grade", literals={
    EnumerationLiteral(name="A"),
    EnumerationLiteral(name="B"),
    EnumerationLiteral(name="C"),
    EnumerationLiteral(name="D"),
    EnumerationLiteral(name="F"),
})

student = Class(name="Student")
student.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="student_id", type=StringType),
    Property(name="first_name", type=StringType),
    Property(name="last_name", type=StringType),
    Property(name="email", type=StringType),
    Property(name="enrollment_date", type=DateType),
    Property(name="gpa", type=FloatType, is_optional=True),
}

professor = Class(name="Professor")
professor.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="first_name", type=StringType),
    Property(name="last_name", type=StringType),
    Property(name="email", type=StringType),
    Property(name="department", type=StringType),
}

course = Class(name="Course")
course.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="code", type=StringType),
    Property(name="title", type=StringType),
    Property(name="credits", type=IntegerType),
    Property(name="max_students", type=IntegerType),
}

enrollment = Class(name="Enrollment")
enrollment.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="year", type=IntegerType),
    Property(name="score", type=FloatType, is_optional=True),
}

# Associations
professor_course = BinaryAssociation(name="Professor_Course", ends={
    Property(name="professor", type=professor, multiplicity=Multiplicity(1, 1)),
    Property(name="courses", type=course, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

student_enrollment = BinaryAssociation(name="Student_Enrollment", ends={
    Property(name="student", type=student, multiplicity=Multiplicity(1, 1)),
    Property(name="enrollments", type=enrollment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

course_enrollment = BinaryAssociation(name="Course_Enrollment", ends={
    Property(name="course", type=course, multiplicity=Multiplicity(1, 1)),
    Property(name="enrollments", type=enrollment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
})

university_model = DomainModel(
    name="UniversityManagement",
    types={student, professor, course, enrollment, semester, grade},
    associations={professor_course, student_enrollment, course_enrollment},
)

# ── Generate Django project with extras ───────────────────────────────

from run_helper import run_generator

run_generator(
    model=university_model,
    output_dir="./output/university_django",
    instructions="""
    Build a Django project for this university management system:

    1. Use the Django generator for the base project
    2. Add Django REST Framework with:
       - ViewSets and Routers for all models
       - Serializers with nested relationships
       - Pagination (20 items per page)
       - Filtering by department, semester, grade
       - Search by student name, course title
    3. Customize Django Admin:
       - list_display with key fields
       - list_filter for enums and foreign keys
       - search_fields for text fields
    4. Add a custom manager on Student with:
       - active_students() — students enrolled this year
       - honor_roll() — students with GPA > 3.5
    5. Add unit tests for the API endpoints
    6. Include requirements.txt
    """,
)
