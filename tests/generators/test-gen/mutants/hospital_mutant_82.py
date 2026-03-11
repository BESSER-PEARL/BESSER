# Mutant Operator: Updated Attribute Name: old_value -> old_value_mutant
from datetime import datetime, date, time
from enum import Enum

class AppointmentStatus(Enum):
    SCHEDULED = 'SCHEDULED'
    COMPLETED = 'COMPLETED'
    CANCELLED = 'CANCELLED'
    NO_SHOW = 'NO_SHOW'

class Appointment:

    def __init__(self, appointmentId: str, date: str, time: str, status: str, patient: 'Patient'=None, doctor: 'Doctor'=None):
        self.appointmentId = appointmentId
        self.date = date
        self.time = time
        self.status = status
        self.patient = patient
        self.doctor = doctor

    @property
    def appointmentId(self) -> str:
        return self.__appointmentId

    @appointmentId.setter
    def appointmentId(self, appointmentId: str):
        self.__appointmentId = appointmentId

    @property
    def time(self) -> str:
        return self.__time

    @time.setter
    def time(self, time: str):
        self.__time = time

    @property
    def date(self) -> str:
        return self.__date

    @date.setter
    def date(self, date: str):
        self.__date = date

    @property
    def status(self) -> str:
        return self.__status

    @status.setter
    def status(self, status: str):
        self.__status = status

    @property
    def patient(self):
        return self.__patient

    @patient.setter
    def patient(self, value):
        old_value = getattr(self, f'_Appointment__patient', None)
        self.__patient = value
        if old_value is not None:
            if hasattr(old_value, 'appointments'):
                opp_val = getattr(old_value, 'appointments', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'appointments'):
                opp_val = getattr(value, 'appointments', None)
                if opp_val is None:
                    setattr(value, 'appointments', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    @property
    def doctor(self):
        return self.__doctor

    @doctor.setter
    def doctor(self, value):
        old_value = getattr(self, f'_Appointment__doctor', None)
        self.__doctor = value
        if old_value is not None:
            if hasattr(old_value, 'schedule'):
                opp_val = getattr(old_value, 'schedule', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'schedule'):
                opp_val = getattr(value, 'schedule', None)
                if opp_val is None:
                    setattr(value, 'schedule', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def cancel(self):
        self.status = 'Cancelled'
        print(f'Appointment {self.appointmentId} has been cancelled')

    def schedule(self):
        self.status = 'Scheduled'
        print(f'Appointment {self.appointmentId} scheduled')
        print(f'Date: {self.date}, Time: {self.time}')

    def complete(self):
        self.status = 'Completed'
        print(f'Appointment {self.appointmentId} marked as completed')

class Doctor:

    def __init__(self, doctorId: str, name: str, specialization: str, available: bool, schedule: set['Appointment']=None):
        self.doctorId = doctorId
        self.name = name
        self.specialization = specialization
        self.available = available
        self.schedule = schedule if schedule is not None else set()

    @property
    def specialization(self) -> str:
        return self.__specialization

    @specialization.setter
    def specialization(self, specialization: str):
        self.__specialization = specialization

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def doctorId(self) -> str:
        return self.__doctorId

    @doctorId.setter
    def doctorId(self, doctorId: str):
        self.__doctorId = doctorId

    @property
    def available(self) -> bool:
        return self.__available

    @available.setter
    def available(self, available: bool):
        self.__available = available

    @property
    def schedule(self):
        return self.__schedule

    @schedule.setter
    def schedule(self, value):
        old_value_mutant = getattr(self, f'_Doctor__schedule', None)
        self.__schedule = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'doctor'):
                    opp_val = getattr(item, 'doctor', None)
                    if opp_val == self:
                        setattr(item, 'doctor', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'doctor'):
                    opp_val = getattr(item, 'doctor', None)
                    setattr(item, 'doctor', self)

    def registerDoctor(self):
        self.available = True
        print(f'Doctor registered: Dr. {self.name}')
        print(f'ID: {self.doctorId}, Specialization: {self.specialization}')

    def setAvailability(self, status):
        self.available = status
        status_text = 'available' if status else 'unavailable'
        print(f'Dr. {self.name} is now {status_text}')

    def getDoctorInfo(self):
        availability = 'Available' if self.available else 'Not Available'
        return f'Dr. {self.name} - {self.specialization} ({availability})'

class Patient:

    def __init__(self, patientId: str, name: str, age: int, bloodType: str, appointments: set['Appointment']=None):
        self.patientId = patientId
        self.name = name
        self.age = age
        self.bloodType = bloodType
        self.appointments = appointments if appointments is not None else set()

    @property
    def patientId(self) -> str:
        return self.__patientId

    @patientId.setter
    def patientId(self, patientId: str):
        self.__patientId = patientId

    @property
    def age(self) -> int:
        return self.__age

    @age.setter
    def age(self, age: int):
        self.__age = age

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def bloodType(self) -> str:
        return self.__bloodType

    @bloodType.setter
    def bloodType(self, bloodType: str):
        self.__bloodType = bloodType

    @property
    def appointments(self):
        return self.__appointments

    @appointments.setter
    def appointments(self, value):
        old_value = getattr(self, f'_Patient__appointments', None)
        self.__appointments = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'patient'):
                    opp_val = getattr(item, 'patient', None)
                    if opp_val == self:
                        setattr(item, 'patient', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'patient'):
                    opp_val = getattr(item, 'patient', None)
                    setattr(item, 'patient', self)

    def registerPatient(self):
        print(f'Patient registered: {self.name}')
        print(f'ID: {self.patientId}, Age: {self.age}, Blood Type: {self.bloodType}')

    def updateAge(self, newAge):
        self.age = newAge
        print(f"Patient {self.name}'s age updated to {newAge}")

    def getMedicalInfo(self):
        info = f'Patient: {self.name}\n'
        info += f'ID: {self.patientId}\n'
        info += f'Age: {self.age}\n'
        info += f'Blood Type: {self.bloodType}'
        return info