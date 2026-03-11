"""
BUML Model Example 4: University System
A simple university system with students, courses, and enrollments
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
active_lit = EnumerationLiteral(name="ACTIVE")
completed_lit = EnumerationLiteral(name="COMPLETED")
dropped_lit = EnumerationLiteral(name="DROPPED")
withdrawn_lit = EnumerationLiteral(name="WITHDRAWN")

enrollment_status_enum = Enumeration(
    name="EnrollmentStatus",
    literals={active_lit, completed_lit, dropped_lit, withdrawn_lit}
)

# =============================================================================
# 2. Define Student Attributes
# =============================================================================
student_id_prop = Property(name="studentId", type=StringType, multiplicity=Multiplicity(1, 1))
student_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
major_prop = Property(name="major", type=StringType, multiplicity=Multiplicity(1, 1))
gpa_prop = Property(name="gpa", type=FloatType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Course Attributes
# =============================================================================
course_id_prop = Property(name="courseId", type=StringType, multiplicity=Multiplicity(1, 1))
course_title_prop = Property(name="title", type=StringType, multiplicity=Multiplicity(1, 1))
credits_prop = Property(name="credits", type=IntegerType, multiplicity=Multiplicity(1, 1))
max_students_prop = Property(name="maxStudents", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Enrollment Attributes
# =============================================================================
enrollment_id_prop = Property(name="enrollmentId", type=StringType, multiplicity=Multiplicity(1, 1))
enrollment_date_prop = Property(name="enrollmentDate", type=StringType, multiplicity=Multiplicity(1, 1))
grade_prop = Property(name="grade", type=StringType, multiplicity=Multiplicity(1, 1))
enrollment_status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Student Methods
# =============================================================================
enroll_student_method = Method(
    name="enrollStudent",
    parameters=[],
    code="""
def enrollStudent(self):
    print(f"Student enrolled: {self.name}")
    print(f"ID: {self.studentId}, Major: {self.major}, GPA: {self.gpa}")
"""
)

update_gpa_method = Method(
    name="updateGpa",
    parameters=[Parameter(name="newGpa", type=FloatType)],
    code="""
def updateGpa(self, newGpa):
    old_gpa = self.gpa
    self.gpa = newGpa
    print(f"GPA updated for {self.name}: {old_gpa:.2f} -> {newGpa:.2f}")
"""
)

is_honor_student_method = Method(
    name="isHonorStudent",
    parameters=[],
    code="""
def isHonorStudent(self):
    if self.gpa >= 3.5:
        print(f"{self.name} is an honor student with GPA {self.gpa}")
        return True
    else:
        return False
"""
)

# =============================================================================
# 6. Define Course Methods
# =============================================================================
create_course_method = Method(
    name="createCourse",
    parameters=[],
    code="""
def createCourse(self):
    print(f"Course created: {self.title}")
    print(f"ID: {self.courseId}, Credits: {self.credits}, Max Students: {self.maxStudents}")
"""
)

has_space_method = Method(
    name="hasSpace",
    parameters=[Parameter(name="currentEnrollment", type=IntegerType)],
    code="""
def hasSpace(self, currentEnrollment):
    if currentEnrollment < self.maxStudents:
        print(f"Course '{self.title}' has space available")
        return True
    else:
        print(f"Course '{self.title}' is full")
        return False
"""
)

get_course_info_method = Method(
    name="getCourseInfo",
    parameters=[],
    code="""
def getCourseInfo(self):
    return f"{self.courseId}: {self.title} ({self.credits} credits)"
"""
)

# =============================================================================
# 7. Define Enrollment Methods
# =============================================================================
register_method = Method(
    name="register",
    parameters=[],
    code="""
def register(self):
    self.status = "Active"
    self.grade = "N/A"
    print(f"Enrollment {self.enrollmentId} registered on {self.enrollmentDate}")
"""
)

assign_grade_method = Method(
    name="assignGrade",
    parameters=[Parameter(name="grade", type=StringType)],
    code="""
def assignGrade(self, grade):
    self.grade = grade
    self.status = "Completed"
    print(f"Grade '{grade}' assigned to enrollment {self.enrollmentId}")
"""
)

drop_course_method = Method(
    name="dropCourse",
    parameters=[],
    code="""
def dropCourse(self):
    self.status = "Dropped"
    print(f"Enrollment {self.enrollmentId} has been dropped")
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
student_class = Class(
    name="Student",
    attributes={student_id_prop, student_name_prop, major_prop, gpa_prop},
    methods={enroll_student_method, update_gpa_method, is_honor_student_method}
)

course_class = Class(
    name="Course",
    attributes={course_id_prop, course_title_prop, credits_prop, max_students_prop},
    methods={create_course_method, has_space_method, get_course_info_method}
)

enrollment_class = Class(
    name="Enrollment",
    attributes={enrollment_id_prop, enrollment_date_prop, grade_prop, enrollment_status_prop},
    methods={register_method, assign_grade_method, drop_course_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# Student --< Enrollment (one student can have many enrollments)
student_end = Property(
    name="student",
    type=student_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
student_enrollments_end = Property(
    name="enrollments",
    type=enrollment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
student_enrollment_assoc = BinaryAssociation(
    name="Registers",
    ends={student_end, student_enrollments_end}
)

# Course --< Enrollment (one course can have many enrollments)
course_end = Property(
    name="course",
    type=course_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
course_enrollments_end = Property(
    name="enrollments",
    type=enrollment_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
course_enrollment_assoc = BinaryAssociation(
    name="HasEnrollments",
    ends={course_end, course_enrollments_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
university_model = DomainModel(
    name="UniversitySystem",
    types={student_class, course_class, enrollment_class, enrollment_status_enum},
    associations={student_enrollment_assoc, course_enrollment_assoc}
)

print("✓ University System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in university_model.get_classes()]}")
print(f"  Associations: {[a.name for a in university_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=university_model, output_dir="output_university")
python_gen.generate()