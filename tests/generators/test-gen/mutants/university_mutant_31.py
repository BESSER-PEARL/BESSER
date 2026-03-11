# Mutant Operator: Incorrect Return Value
from datetime import datetime, date, time
from enum import Enum

class EnrollmentStatus(Enum):
    ACTIVE = 'ACTIVE'
    COMPLETED = 'COMPLETED'
    DROPPED = 'DROPPED'
    WITHDRAWN = 'WITHDRAWN'

class Enrollment:

    def __init__(self, enrollmentId: str, enrollmentDate: str, grade: str, status: str, student: 'Student'=None, course: 'Course'=None):
        self.enrollmentId = enrollmentId
        self.enrollmentDate = enrollmentDate
        self.grade = grade
        self.status = status
        self.student = student
        self.course = course

    @property
    def enrollmentDate(self) -> str:
        return self.__enrollmentDate

    @enrollmentDate.setter
    def enrollmentDate(self, enrollmentDate: str):
        self.__enrollmentDate = enrollmentDate

    @property
    def enrollmentId(self) -> str:
        return self.__enrollmentId

    @enrollmentId.setter
    def enrollmentId(self, enrollmentId: str):
        self.__enrollmentId = enrollmentId

    @property
    def status(self) -> str:
        return self.__status

    @status.setter
    def status(self, status: str):
        self.__status = status

    @property
    def grade(self) -> str:
        return self.__grade

    @grade.setter
    def grade(self, grade: str):
        self.__grade = grade

    @property
    def student(self):
        return None

    @student.setter
    def student(self, value):
        old_value = getattr(self, f'_Enrollment__student', None)
        self.__student = value
        if old_value is not None:
            if hasattr(old_value, 'enrollments'):
                opp_val = getattr(old_value, 'enrollments', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'enrollments'):
                opp_val = getattr(value, 'enrollments', None)
                if opp_val is None:
                    setattr(value, 'enrollments', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def course(self):
        return self.__course

    @course.setter
    def course(self, value):
        old_value = getattr(self, f'_Enrollment__course', None)
        self.__course = value
        if old_value is not None:
            if hasattr(old_value, 'enrollments'):
                opp_val = getattr(old_value, 'enrollments', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'enrollments'):
                opp_val = getattr(value, 'enrollments', None)
                if opp_val is None:
                    setattr(value, 'enrollments', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def assignGrade(self, grade):
        self.grade = grade
        self.status = 'Completed'
        print(f"Grade '{grade}' assigned to enrollment {self.enrollmentId}")

    def register(self):
        self.status = 'Active'
        self.grade = 'N/A'
        print(f'Enrollment {self.enrollmentId} registered on {self.enrollmentDate}')

    def dropCourse(self):
        self.status = 'Dropped'
        print(f'Enrollment {self.enrollmentId} has been dropped')

class Course:

    def __init__(self, courseId: str, title: str, credits: int, maxStudents: int, enrollments: set['Enrollment']=None):
        self.courseId = courseId
        self.title = title
        self.credits = credits
        self.maxStudents = maxStudents
        self.enrollments = enrollments if enrollments is not None else set()

    @property
    def courseId(self) -> str:
        return self.__courseId

    @courseId.setter
    def courseId(self, courseId: str):
        self.__courseId = courseId

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str):
        self.__title = title

    @property
    def credits(self) -> int:
        return self.__credits

    @credits.setter
    def credits(self, credits: int):
        self.__credits = credits

    @property
    def maxStudents(self) -> int:
        return self.__maxStudents

    @maxStudents.setter
    def maxStudents(self, maxStudents: int):
        self.__maxStudents = maxStudents

    @property
    def enrollments(self):
        return self.__enrollments

    @enrollments.setter
    def enrollments(self, value):
        old_value = getattr(self, f'_Course__enrollments', None)
        self.__enrollments = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'course'):
                    opp_val = getattr(item, 'course', None)
                    if opp_val == self:
                        setattr(item, 'course', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'course'):
                    opp_val = getattr(item, 'course', None)
                    setattr(item, 'course', self)

    def createCourse(self):
        print(f'Course created: {self.title}')
        print(f'ID: {self.courseId}, Credits: {self.credits}, Max Students: {self.maxStudents}')

    def hasSpace(self, currentEnrollment):
        if currentEnrollment < self.maxStudents:
            print(f"Course '{self.title}' has space available")
            return True
        else:
            print(f"Course '{self.title}' is full")
            return False

    def getCourseInfo(self):
        return f'{self.courseId}: {self.title} ({self.credits} credits)'

class Student:

    def __init__(self, studentId: str, name: str, major: str, gpa: float, enrollments: set['Enrollment']=None):
        self.studentId = studentId
        self.name = name
        self.major = major
        self.gpa = gpa
        self.enrollments = enrollments if enrollments is not None else set()

    @property
    def major(self) -> str:
        return self.__major

    @major.setter
    def major(self, major: str):
        self.__major = major

    @property
    def gpa(self) -> float:
        return self.__gpa

    @gpa.setter
    def gpa(self, gpa: float):
        self.__gpa = gpa

    @property
    def studentId(self) -> str:
        return self.__studentId

    @studentId.setter
    def studentId(self, studentId: str):
        self.__studentId = studentId

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def enrollments(self):
        return self.__enrollments

    @enrollments.setter
    def enrollments(self, value):
        old_value = getattr(self, f'_Student__enrollments', None)
        self.__enrollments = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'student'):
                    opp_val = getattr(item, 'student', None)
                    if opp_val == self:
                        setattr(item, 'student', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'student'):
                    opp_val = getattr(item, 'student', None)
                    setattr(item, 'student', self)

    def updateGpa(self, newGpa):
        old_gpa = self.gpa
        self.gpa = newGpa
        print(f'GPA updated for {self.name}: {old_gpa:.2f} -> {newGpa:.2f}')

    def enrollStudent(self):
        print(f'Student enrolled: {self.name}')
        print(f'ID: {self.studentId}, Major: {self.major}, GPA: {self.gpa}')

    def isHonorStudent(self):
        if self.gpa >= 3.5:
            print(f'{self.name} is an honor student with GPA {self.gpa}')
            return True
        else:
            return False