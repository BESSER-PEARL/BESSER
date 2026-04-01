"""
Comprehensive B-UML model exercising ALL metamodel features.
Generates: standalone SQLAlchemy, standalone Pydantic, full layered Backend.

Features covered:
  - All primitive types: str, int, float, bool, date, datetime, time
  - Enumerations (3 enums)
  - is_id (custom primary keys)
  - is_optional (nullable fields)
  - default_value (default column values)
  - is_abstract (abstract base class for concrete-table inheritance)
  - Deep inheritance chain: Person <- Student <- GraduateStudent
  - Sibling inheritance: Person <- Professor
  - Concrete-table inheritance: Vehicle (abstract, no parents, no assoc ends)
  - Association 1:1 (optional): Department <-> Building
  - Association N:1 (mandatory): Professor -> Department
  - Association 1:N: Professor -> Course (teaches)
  - Association N:M: Student <-> Course
  - Association N:M self-referencing: Student mentors Student
  - AssociationClass: Enrollment on Student<->Course with own attributes
  - Composition: Building owns Rooms (is_composite=True)
  - Non-navigable end: Room -> Building (one-way)
  - Methods with BAL implementation: Professor.promote()
  - OCL constraints: gpa range, credits > 0, capacity > 0
  - Nested creations mode (for Backend)
"""

import os
import sys

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    BinaryAssociation, AssociationClass,
    Generalization,
    Enumeration, EnumerationLiteral, Multiplicity,
    Constraint, MethodImplementationType,
    StringType, IntegerType, FloatType, BooleanType,
    DateType, DateTimeType, TimeType,
)

# ====================================================================
# ENUMERATIONS
# ====================================================================

academic_rank = Enumeration(name="AcademicRank", literals={
    EnumerationLiteral(name="ASSISTANT"),
    EnumerationLiteral(name="ASSOCIATE"),
    EnumerationLiteral(name="FULL"),
})

course_level = Enumeration(name="CourseLevel", literals={
    EnumerationLiteral(name="UNDERGRADUATE"),
    EnumerationLiteral(name="GRADUATE"),
    EnumerationLiteral(name="DOCTORAL"),
})

semester_enum = Enumeration(name="Semester", literals={
    EnumerationLiteral(name="FALL"),
    EnumerationLiteral(name="SPRING"),
    EnumerationLiteral(name="SUMMER"),
})

# ====================================================================
# CLASSES — exercising all attribute options
# ====================================================================

# Regular class — will be parent for joined-table inheritance
person = Class(name="Person", attributes={
    Property(name="email", type=StringType, is_id=True),       # Custom PK
    Property(name="name", type=StringType),
    Property(name="birth_date", type=DateType),                 # date type
    Property(name="is_active", type=BooleanType, default_value=True),  # default value
})

# Inherits from Person — adds float + int attributes
student = Class(name="Student", attributes={
    Property(name="gpa", type=FloatType),                       # OCL constrained
    Property(name="enrollment_year", type=IntegerType),
})

# Deep inheritance: Person <- Student <- GraduateStudent
grad_student = Class(name="GraduateStudent", attributes={
    Property(name="thesis_title", type=StringType, is_optional=True),  # nullable
    Property(name="defense_time", type=TimeType, is_optional=True),    # time type + nullable
})

# Sibling inheritance: Person <- Professor (with enum + datetime)
professor = Class(name="Professor", attributes={
    Property(name="rank", type=academic_rank),                  # enum attribute
    Property(name="salary", type=FloatType),
    Property(name="hired_at", type=DateTimeType),               # datetime type
})

# Standalone class with custom PK
department = Class(name="Department", attributes={
    Property(name="code", type=StringType, is_id=True),         # Custom PK
    Property(name="title", type=StringType),
    Property(name="budget", type=FloatType, is_optional=True),  # nullable
})

# Class with enum attribute + custom PK
course = Class(name="Course", attributes={
    Property(name="code", type=StringType, is_id=True),         # Custom PK
    Property(name="title", type=StringType),
    Property(name="credits", type=IntegerType),                 # OCL constrained
    Property(name="level", type=course_level),                  # enum attribute
})

# Class with default value
building = Class(name="Building", attributes={
    Property(name="name", type=StringType, is_id=True),         # Custom PK
    Property(name="floors", type=IntegerType),
    Property(name="has_lab", type=BooleanType, default_value=False),  # default
})

# Simple class (auto-generated PK)
room = Class(name="Room", attributes={
    Property(name="number", type=StringType),
    Property(name="capacity", type=IntegerType),                # OCL constrained
})

# Abstract class for concrete-table inheritance
# (abstract=True, no parents, no association ends)
vehicle = Class(name="Vehicle", is_abstract=True, attributes={
    Property(name="plate", type=StringType, is_id=True),
    Property(name="year", type=IntegerType),
})

car = Class(name="Car", attributes={
    Property(name="doors", type=IntegerType, default_value=4),
})

bicycle = Class(name="Bicycle", attributes={
    Property(name="gear_count", type=IntegerType),
})

# ====================================================================
# GENERALIZATIONS — inheritance
# ====================================================================

gen_person_student = Generalization(general=person, specific=student)
gen_student_grad = Generalization(general=student, specific=grad_student)
gen_person_prof = Generalization(general=person, specific=professor)

# Concrete-table inheritance
gen_vehicle_car = Generalization(general=vehicle, specific=car)
gen_vehicle_bicycle = Generalization(general=vehicle, specific=bicycle)

# ====================================================================
# ASSOCIATIONS — all relationship types
# ====================================================================

# N:1 mandatory — Professor must belong to a Department
prof_dept = BinaryAssociation(name="ProfDept", ends={
    Property(name="professors", type=professor, multiplicity=Multiplicity(0, 9999)),
    Property(name="department", type=department, multiplicity=Multiplicity(1, 1)),
})

# 1:N — Professor teaches many Courses
prof_course = BinaryAssociation(name="ProfCourse", ends={
    Property(name="taught_by", type=professor, multiplicity=Multiplicity(1, 1)),
    Property(name="courses", type=course, multiplicity=Multiplicity(0, 9999)),
})

# N:M — Student enrolled in many Courses (will have AssociationClass)
student_course = BinaryAssociation(name="StudentCourse", ends={
    Property(name="students", type=student, multiplicity=Multiplicity(0, 9999)),
    Property(name="courses", type=course, multiplicity=Multiplicity(0, 9999)),
})

# 1:1 optional — Department optionally housed in a Building
dept_building = BinaryAssociation(name="DeptBuilding", ends={
    Property(name="department", type=department, multiplicity=Multiplicity(0, 1)),
    Property(name="housed_in", type=building, multiplicity=Multiplicity(0, 1)),
})

# 1:N composition — Building owns Rooms
building_room = BinaryAssociation(name="BuildingRoom", ends={
    Property(name="building", type=building, multiplicity=Multiplicity(1, 1)),
    Property(name="rooms", type=room, multiplicity=Multiplicity(0, 9999)),
})

# N:M self-association — Student mentors/mentees
student_mentoring = BinaryAssociation(name="Mentoring", ends={
    Property(name="mentors", type=student, multiplicity=Multiplicity(0, 9999)),
    Property(name="mentees", type=student, multiplicity=Multiplicity(0, 9999)),
})

# ====================================================================
# ASSOCIATION CLASS — on the Student<->Course N:M
# ====================================================================

enrollment = AssociationClass(
    name="Enrollment",
    attributes={
        Property(name="grade", type=FloatType, is_optional=True),   # nullable
        Property(name="semester", type=semester_enum),               # enum
        Property(name="year", type=IntegerType),
    },
    association=student_course,
)

# ====================================================================
# METHODS — BAL implementation
# ====================================================================

promote = Method(
    name="promote",
    parameters=[],
    type=None,
    implementation_type=MethodImplementationType.BAL,
)
promote.code = """def promote() -> nothing {
    this.salary = 1.1 * this.salary
}"""
professor.add_method(promote)

# ====================================================================
# OCL CONSTRAINTS
# ====================================================================

gpa_min = Constraint(name="gpa_min", context=student,
                     expression="self.gpa >= 0.0", language="OCL")
gpa_max = Constraint(name="gpa_max", context=student,
                     expression="self.gpa <= 4.0", language="OCL")
credits_positive = Constraint(name="credits_positive", context=course,
                              expression="self.credits > 0", language="OCL")
capacity_positive = Constraint(name="capacity_positive", context=room,
                               expression="self.capacity > 0", language="OCL")

# ====================================================================
# DOMAIN MODEL
# ====================================================================

model = DomainModel(
    name="University",
    types={
        person, student, grad_student, professor,
        department, course, building, room,
        vehicle, car, bicycle,
        academic_rank, course_level, semester_enum,
        enrollment,
    },
    associations={
        prof_dept, prof_course, student_course,
        dept_building, building_room, student_mentoring,
    },
    generalizations={
        gen_person_student, gen_student_grad, gen_person_prof,
        gen_vehicle_car, gen_vehicle_bicycle,
    },
    constraints={gpa_min, gpa_max, credits_positive, capacity_positive},
)

# ====================================================================
# GENERATE ALL OUTPUTS
# ====================================================================

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_university")

# 1) Standalone SQLAlchemy (flat file)
from besser.generators.sql_alchemy import SQLAlchemyGenerator
sa_dir = os.path.join(BASE, "standalone_sqlalchemy")
os.makedirs(sa_dir, exist_ok=True)
sa_gen = SQLAlchemyGenerator(model=model, output_dir=sa_dir)
sa_gen.generate()

# 2) Standalone Pydantic (flat file)
from besser.generators.pydantic_classes import PydanticGenerator
py_dir = os.path.join(BASE, "standalone_pydantic")
os.makedirs(py_dir, exist_ok=True)
py_gen = PydanticGenerator(model=model, backend=True, nested_creations=True, output_dir=py_dir)
py_gen.generate()

# 3) Full layered Backend
from besser.generators.backend import BackendGenerator
be_dir = os.path.join(BASE, "backend")
be_gen = BackendGenerator(
    model=model,
    output_dir=be_dir,
    nested_creations=True,
    api_title="University Management API",
    api_description="Comprehensive university management system",
    api_version="2.0.0",
)
be_gen.generate()

# ====================================================================
# PRINT STRUCTURE
# ====================================================================

print("\n" + "=" * 60)
print("GENERATED FILES")
print("=" * 60)
for root, dirs, files in os.walk(BASE):
    level = root.replace(BASE, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        print(f"{indent}  {f}  ({size} bytes)")
