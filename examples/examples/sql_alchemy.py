import enum
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, ForeignKey, Table, Text, Boolean, String, Date, 
    Time, DateTime, Float, Integer, Enum
)
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, mapped_column, relationship
)
from datetime import datetime, time, date

class Base(DeclarativeBase):
    pass



# Tables definition for many-to-many relationships
enrollmentassotest = Table(
    "enrollmentassotest",
    Base.metadata,
    Column("students", ForeignKey("student.studentId"), primary_key=True),
    Column("test", ForeignKey("test.id"), primary_key=True),
)

# Tables definition
class Test(Base):
    __tablename__ = "test"
    id: Mapped[int] = mapped_column(primary_key=True)

class Course(Base):
    __tablename__ = "course"
    courseId: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(100))

class Student(Base):
    __tablename__ = "student"
    studentId: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

class Enrollment(Base):
    __tablename__ = "enrollment"
    courses_id: Mapped[str] = mapped_column(ForeignKey("course.courseId"), primary_key=True)
    students_id: Mapped[str] = mapped_column(ForeignKey("student.studentId"), primary_key=True)
    grade: Mapped[str] = mapped_column(String(100))
    semester: Mapped[str] = mapped_column(String(100))


#--- Foreign keys and relationships of the test table
Test.students: Mapped[List["Student"]] = relationship("Student", secondary=enrollmentassotest, back_populates="test")

#--- Foreign keys and relationships of the course table

#--- Foreign keys and relationships of the student table
Student.test: Mapped[List["Test"]] = relationship("Test", secondary=enrollmentassotest, back_populates="students")
#--- Relationships for association class Enrollment
Enrollment.courses: Mapped["Course"] = relationship("Course", back_populates="enrollments")
Course.enrollments: Mapped[List["Enrollment"]] = relationship("Enrollment", back_populates="courses")
Enrollment.students: Mapped["Student"] = relationship("Student", back_populates="enrollments")
Student.enrollments: Mapped[List["Enrollment"]] = relationship("Enrollment", back_populates="students")

# Database connection

DATABASE_URL = "sqlite:///UniversityModel.db"  # SQLite connection

engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine, checkfirst=True)