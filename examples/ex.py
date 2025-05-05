
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

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Course(Base):
    __tablename__ = 'courses'
    id = Column(Integer, primary_key=True)
    title = Column(String)

class Enrollment(Base):  # Esto es la Association Class
    __tablename__ = 'enrollments'
    student_id = Column(Integer, ForeignKey('students.id'), primary_key=True)
    course_id = Column(Integer, ForeignKey('courses.id'), primary_key=True)
    grade = Column(String)


Student.enrollments = relationship("Enrollment", back_populates="course")

Course.enrollments = relationship("Enrollment", back_populates="student")

Enrollment.student = relationship("Student", back_populates="enrollments")
Enrollment.course = relationship("Course", back_populates="enrollments")
 