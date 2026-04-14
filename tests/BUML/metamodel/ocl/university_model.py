"""
University domain model for comprehensive OCL constraint testing.

Domain:
    - University: name, ranking, isPublic
    - Department: name, budget, isActive
    - Professor: name, age, salary, tenured
    - Student: name, gpa, age, isGraduate
    - Course: title, credits, maxStudents

Associations:
    - University hasDept Department (1..*)
    - Department employs Professor (1..*)
    - Department offers Course (0..*)
    - Professor teaches Course (0..*)
    - Student enrolledIn Course (0..*)

Object instances are designed so that each constraint has a known
True or False outcome.
"""

from besser.BUML.metamodel.structural import (
    Class, Property, BinaryAssociation, Generalization,
    DomainModel, Multiplicity, Constraint,
    StringType, IntegerType, FloatType, BooleanType,
)
from besser.BUML.metamodel.object import ObjectModel

# =====================================================================
# Classes
# =====================================================================
University = Class(name="University")
Department = Class(name="Department")
Professor = Class(name="Professor")
Student = Class(name="Student")
Course = Class(name="Course")

# University attributes
Uni_name: Property = Property(name="name", type=StringType)
Uni_ranking: Property = Property(name="ranking", type=IntegerType)
Uni_isPublic: Property = Property(name="isPublic", type=BooleanType)
University.attributes = {Uni_name, Uni_ranking, Uni_isPublic}

# Department attributes
Dept_name: Property = Property(name="name", type=StringType)
Dept_budget: Property = Property(name="budget", type=IntegerType)
Dept_isActive: Property = Property(name="isActive", type=BooleanType)
Department.attributes = {Dept_name, Dept_budget, Dept_isActive}

# Professor attributes
Prof_name: Property = Property(name="name", type=StringType)
Prof_age: Property = Property(name="age", type=IntegerType)
Prof_salary: Property = Property(name="salary", type=IntegerType)
Prof_tenured: Property = Property(name="tenured", type=BooleanType)
Professor.attributes = {Prof_name, Prof_age, Prof_salary, Prof_tenured}

# Student attributes
Stu_name: Property = Property(name="name", type=StringType)
Stu_gpa: Property = Property(name="gpa", type=IntegerType)
Stu_age: Property = Property(name="age", type=IntegerType)
Stu_isGraduate: Property = Property(name="isGraduate", type=BooleanType)
Student.attributes = {Stu_name, Stu_gpa, Stu_age, Stu_isGraduate}

# Course attributes
Crs_title: Property = Property(name="title", type=StringType)
Crs_credits: Property = Property(name="credits", type=IntegerType)
Crs_maxStudents: Property = Property(name="maxStudents", type=IntegerType)
Course.attributes = {Crs_title, Crs_credits, Crs_maxStudents}

# =====================================================================
# Associations
# =====================================================================
Uni_Dept = BinaryAssociation(
    name="Uni_Dept",
    ends={
        Property(name="university", type=University, multiplicity=Multiplicity(1, 1)),
        Property(name="hasDept", type=Department, multiplicity=Multiplicity(1, 9999)),
    }
)

Dept_Prof = BinaryAssociation(
    name="Dept_Prof",
    ends={
        Property(name="department", type=Department, multiplicity=Multiplicity(1, 1)),
        Property(name="employs", type=Professor, multiplicity=Multiplicity(1, 9999)),
    }
)

Dept_Course = BinaryAssociation(
    name="Dept_Course",
    ends={
        Property(name="offeredBy", type=Department, multiplicity=Multiplicity(1, 1)),
        Property(name="offers", type=Course, multiplicity=Multiplicity(0, 9999)),
    }
)

Prof_Course = BinaryAssociation(
    name="Prof_Course",
    ends={
        Property(name="taughtBy", type=Professor, multiplicity=Multiplicity(1, 1)),
        Property(name="teaches", type=Course, multiplicity=Multiplicity(0, 9999)),
    }
)

Stu_Course = BinaryAssociation(
    name="Stu_Course",
    ends={
        Property(name="students", type=Student, multiplicity=Multiplicity(0, 9999)),
        Property(name="enrolledIn", type=Course, multiplicity=Multiplicity(0, 9999)),
    }
)

# =====================================================================
# Constraints — expected TRUE
# =====================================================================

# --- Simple comparisons ---
c_true_01 = Constraint(
    name="T01_prof_age_above_25",
    context=Professor,
    expression="context Professor inv: self.age > 25",
    language="OCL"
)

c_true_02 = Constraint(
    name="T02_prof_salary_positive",
    context=Professor,
    expression="context Professor inv: self.salary > 0",
    language="OCL"
)

c_true_03 = Constraint(
    name="T03_student_gpa_nonneg",
    context=Student,
    expression="context Student inv: self.gpa >= 0",
    language="OCL"
)

c_true_04 = Constraint(
    name="T04_dept_budget_positive",
    context=Department,
    expression="context Department inv: self.budget > 0",
    language="OCL"
)

c_true_05 = Constraint(
    name="T05_course_credits_range",
    context=Course,
    expression="context Course inv: self.credits >= 1 and self.credits <= 6",
    language="OCL"
)

# --- String comparisons ---
c_true_06 = Constraint(
    name="T06_prof_name_not_empty",
    context=Professor,
    expression="context Professor inv: self.name <> ''",
    language="OCL"
)

c_true_07 = Constraint(
    name="T07_uni_name_match",
    context=University,
    expression="context University inv: self.name <> 'Unknown'",
    language="OCL"
)

# --- Boolean attributes ---
c_true_08 = Constraint(
    name="T08_dept_all_active",
    context=Department,
    expression="context Department inv: self.isActive = True",
    language="OCL"
)

c_true_09 = Constraint(
    name="T09_uni_is_public",
    context=University,
    expression="context University inv: self.isPublic = True",
    language="OCL"
)

# --- forAll ---
c_true_10 = Constraint(
    name="T10_dept_all_profs_above_25",
    context=Department,
    expression="context Department inv: self.employs->forAll(p:Professor | p.age > 25)",
    language="OCL"
)

c_true_11 = Constraint(
    name="T11_uni_all_depts_active",
    context=University,
    expression="context University inv: self.hasDept->forAll(d:Department | d.isActive = True)",
    language="OCL"
)

# --- exists ---
c_true_12 = Constraint(
    name="T12_dept_has_tenured",
    context=Department,
    expression="context Department inv: self.employs->exists(p:Professor | p.tenured = True)",
    language="OCL"
)

c_true_13 = Constraint(
    name="T13_uni_has_big_dept",
    context=University,
    expression="context University inv: self.hasDept->exists(d:Department | d.budget > 400000)",
    language="OCL"
)

# --- size ---
c_true_14 = Constraint(
    name="T14_uni_has_depts",
    context=University,
    expression="context University inv: self.hasDept->size() > 0",
    language="OCL"
)

c_true_15 = Constraint(
    name="T15_dept_has_profs",
    context=Department,
    expression="context Department inv: self.employs->size() > 0",
    language="OCL"
)

# --- select + size ---
c_true_16 = Constraint(
    name="T16_dept_senior_profs_exist",
    context=Department,
    expression="context Department inv: self.employs->select(p:Professor | p.age > 40)->size() >= 0",
    language="OCL"
)

# --- reject + size ---
c_true_17 = Constraint(
    name="T17_dept_reject_low_salary",
    context=Department,
    expression="context Department inv: self.employs->reject(p:Professor | p.salary < 10000)->size() > 0",
    language="OCL"
)

# --- oclIsTypeOf ---
c_true_18 = Constraint(
    name="T18_prof_name_is_string",
    context=Professor,
    expression="context Professor inv: self.name.oclIsTypeOf(String)",
    language="OCL"
)

c_true_19 = Constraint(
    name="T19_student_age_is_int",
    context=Student,
    expression="context Student inv: self.age.oclIsTypeOf(Integer)",
    language="OCL"
)

# --- if/then/else ---
c_true_20 = Constraint(
    name="T20_uni_if_public_has_depts",
    context=University,
    expression="context University inv: if self.isPublic = True then self.hasDept->size() > 0 else self.ranking > 0 endif",
    language="OCL"
)

c_true_21 = Constraint(
    name="T21_prof_if_tenured_salary",
    context=Professor,
    expression="context Professor inv: if self.tenured = True then self.salary > 50000 else self.salary > 0 endif",
    language="OCL"
)

# --- implies ---
c_true_22 = Constraint(
    name="T22_prof_tenured_implies_senior",
    context=Professor,
    expression="context Professor inv: self.tenured = True implies self.age > 30",
    language="OCL"
)

# --- or ---
c_true_23 = Constraint(
    name="T23_student_grad_or_young",
    context=Student,
    expression="context Student inv: self.isGraduate = True or self.age < 30",
    language="OCL"
)

# --- not ---
c_true_24 = Constraint(
    name="T24_dept_not_zero_budget",
    context=Department,
    expression="context Department inv: not (self.budget = 0)",
    language="OCL"
)

# --- arithmetic ---
c_true_25 = Constraint(
    name="T25_prof_salary_above_age_times_1000",
    context=Professor,
    expression="context Professor inv: self.salary > self.age * 1000",
    language="OCL"
)

# =====================================================================
# Constraints — expected FALSE
# =====================================================================

c_false_01 = Constraint(
    name="F01_prof_age_above_50",
    context=Professor,
    expression="context Professor inv: self.age > 50",
    language="OCL"
)

c_false_02 = Constraint(
    name="F02_student_all_graduate",
    context=Student,
    expression="context Student inv: self.isGraduate = True",
    language="OCL"
)

c_false_03 = Constraint(
    name="F03_dept_budget_above_million",
    context=Department,
    expression="context Department inv: self.budget > 1000000",
    language="OCL"
)

c_false_04 = Constraint(
    name="F04_course_credits_above_5",
    context=Course,
    expression="context Course inv: self.credits > 5",
    language="OCL"
)

c_false_05 = Constraint(
    name="F05_dept_all_profs_tenured",
    context=Department,
    expression="context Department inv: self.employs->forAll(p:Professor | p.tenured = True)",
    language="OCL"
)

c_false_06 = Constraint(
    name="F06_uni_has_5_depts",
    context=University,
    expression="context University inv: self.hasDept->size() > 5",
    language="OCL"
)

c_false_07 = Constraint(
    name="F07_prof_name_is_int",
    context=Professor,
    expression="context Professor inv: self.name.oclIsTypeOf(Integer)",
    language="OCL"
)

c_false_08 = Constraint(
    name="F08_prof_salary_above_200k",
    context=Professor,
    expression="context Professor inv: self.salary > 200000",
    language="OCL"
)

c_false_09 = Constraint(
    name="F09_student_gpa_above_5",
    context=Student,
    expression="context Student inv: self.gpa > 5",
    language="OCL"
)

c_false_10 = Constraint(
    name="F10_dept_no_profs",
    context=Department,
    expression="context Department inv: self.employs->size() = 0",
    language="OCL"
)

c_false_11 = Constraint(
    name="F11_uni_not_public",
    context=University,
    expression="context University inv: self.isPublic = False",
    language="OCL"
)

c_false_12 = Constraint(
    name="F12_dept_exists_salary_above_200k",
    context=Department,
    expression="context Department inv: self.employs->exists(p:Professor | p.salary > 200000)",
    language="OCL"
)

c_false_13 = Constraint(
    name="F13_if_else_false",
    context=University,
    expression="context University inv: if self.name = 'Unknown' then self.hasDept->size() > 0 else self.hasDept->size() = 0 endif",
    language="OCL"
)

c_false_14 = Constraint(
    name="F14_student_name_equals_nobody",
    context=Student,
    expression="context Student inv: self.name = 'Nobody'",
    language="OCL"
)

c_false_15 = Constraint(
    name="F15_prof_young_and_tenured",
    context=Professor,
    expression="context Professor inv: self.age < 30 and self.tenured = True",
    language="OCL"
)

# =====================================================================
# Collect all constraints
# =====================================================================
true_constraints = {
    c_true_01, c_true_02, c_true_03, c_true_04, c_true_05,
    c_true_06, c_true_07, c_true_08, c_true_09, c_true_10,
    c_true_11, c_true_12, c_true_13, c_true_14, c_true_15,
    c_true_16, c_true_17, c_true_18, c_true_19, c_true_20,
    c_true_21, c_true_22, c_true_23, c_true_24, c_true_25,
}

false_constraints = {
    c_false_01, c_false_02, c_false_03, c_false_04, c_false_05,
    c_false_06, c_false_07, c_false_08, c_false_09, c_false_10,
    c_false_11, c_false_12, c_false_13, c_false_14, c_false_15,
}

all_constraints = true_constraints | false_constraints

# =====================================================================
# Domain Model
# =====================================================================
domain_model = DomainModel(
    name="University_Model",
    types={University, Department, Professor, Student, Course},
    associations={Uni_Dept, Dept_Prof, Dept_Course, Prof_Course, Stu_Course},
    constraints=all_constraints,
)

# =====================================================================
# Object Model
# =====================================================================

# --- University ---
uni1 = University("uni1").attributes(
    name="TechState", ranking=15, isPublic=True
).build()

# --- Departments ---
dept_cs = Department("dept_cs").attributes(
    name="ComputerScience", budget=500000, isActive=True
).build()
dept_math = Department("dept_math").attributes(
    name="Mathematics", budget=300000, isActive=True
).build()

# --- Professors ---
prof_smith = Professor("prof_smith").attributes(
    name="Smith", age=52, salary=120000, tenured=True
).build()
prof_jones = Professor("prof_jones").attributes(
    name="Jones", age=38, salary=85000, tenured=True
).build()
prof_lee = Professor("prof_lee").attributes(
    name="Lee", age=33, salary=70000, tenured=False
).build()
prof_clark = Professor("prof_clark").attributes(
    name="Clark", age=45, salary=95000, tenured=True
).build()

# --- Students ---
stu_alice = Student("stu_alice").attributes(
    name="Alice", gpa=4, age=22, isGraduate=True
).build()
stu_bob = Student("stu_bob").attributes(
    name="Bob", gpa=3, age=20, isGraduate=False
).build()
stu_carol = Student("stu_carol").attributes(
    name="Carol", gpa=4, age=25, isGraduate=True
).build()

# --- Courses ---
crs_algo = Course("crs_algo").attributes(
    title="Algorithms", credits=4, maxStudents=30
).build()
crs_db = Course("crs_db").attributes(
    title="Databases", credits=3, maxStudents=25
).build()
crs_calc = Course("crs_calc").attributes(
    title="Calculus", credits=4, maxStudents=40
).build()
crs_intro = Course("crs_intro").attributes(
    title="IntroToCS", credits=3, maxStudents=100
).build()

# --- Links (use single Object or set, not list) ---
# University -> Departments
uni1.hasDept = {dept_cs, dept_math}

# Departments -> Professors
dept_cs.employs = {prof_smith, prof_jones, prof_lee}
dept_math.employs = {prof_clark}

# Departments -> Courses
dept_cs.offers = {crs_algo, crs_db, crs_intro}
dept_math.offers = {crs_calc}

# Professors -> Courses
prof_smith.teaches = crs_algo
prof_jones.teaches = crs_db
prof_lee.teaches = crs_intro
prof_clark.teaches = crs_calc

# Students -> Courses
stu_alice.enrolledIn = {crs_algo, crs_db}
stu_bob.enrolledIn = {crs_intro, crs_calc}
stu_carol.enrolledIn = {crs_algo, crs_calc}

# Object model
object_model = ObjectModel(
    name="University_Objects",
    objects={
        uni1, dept_cs, dept_math,
        prof_smith, prof_jones, prof_lee, prof_clark,
        stu_alice, stu_bob, stu_carol,
        crs_algo, crs_db, crs_calc, crs_intro,
    }
)
