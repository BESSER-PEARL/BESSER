# Mutant Operator: Removed Assignment
from datetime import datetime, date, time
from enum import Enum

class MembershipType(Enum):
    BASIC = 'BASIC'
    PREMIUM = 'PREMIUM'
    VIP = 'VIP'

class TrainingSession:

    def __init__(self, sessionId: str, date: str, duration: int, status: str, member: 'FitnessMember'=None, trainer: 'Trainer'=None):
        self.sessionId = sessionId
        self.date = date
        self.duration = duration
        self.status = status
        self.member = member
        self.trainer = trainer

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
    def sessionId(self) -> str:
        return self.__sessionId

    @sessionId.setter
    def sessionId(self, sessionId: str):
        self.__sessionId = sessionId

    @property
    def duration(self) -> int:
        return self.__duration

    @duration.setter
    def duration(self, duration: int):
        self.__duration = duration

    @property
    def trainer(self):
        return self.__trainer

    @trainer.setter
    def trainer(self, value):
        old_value = getattr(self, f'_TrainingSession__trainer', None)
        self.__trainer = value
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

    @property
    def member(self):
        return self.__member

    @member.setter
    def member(self, value):
        old_value = getattr(self, f'_TrainingSession__member', None)
        pass
        if old_value is not None:
            if hasattr(old_value, 'sessions'):
                opp_val = getattr(old_value, 'sessions', None)
                if isinstance(opp_val, set):
                    opp_val.discard(self)
        if value is not None:
            if hasattr(value, 'sessions'):
                opp_val = getattr(value, 'sessions', None)
                if opp_val is None:
                    setattr(value, 'sessions', set([self]))
                elif isinstance(opp_val, set):
                    opp_val.add(self)

    def calculateCost(self, trainerRate):
        hours = self.duration / 60.0
        cost = hours * trainerRate
        print(f'Session cost: ${cost:.2f} ({hours:.1f} hours x ${trainerRate:.2f}/hour)')
        return cost

    def completeSession(self):
        self.status = 'Completed'
        print(f'Session {self.sessionId} marked as completed')

    def scheduleSession(self):
        self.status = 'Scheduled'
        print(f'Session {self.sessionId} scheduled')
        print(f'Date: {self.date}, Duration: {self.duration} minutes')

class Trainer:

    def __init__(self, trainerId: str, name: str, specialization: str, hourlyRate: float, schedule: set['TrainingSession']=None):
        self.trainerId = trainerId
        self.name = name
        self.specialization = specialization
        self.hourlyRate = hourlyRate
        self.schedule = schedule if schedule is not None else set()

    @property
    def hourlyRate(self) -> float:
        return self.__hourlyRate

    @hourlyRate.setter
    def hourlyRate(self, hourlyRate: float):
        self.__hourlyRate = hourlyRate

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def specialization(self) -> str:
        return self.__specialization

    @specialization.setter
    def specialization(self, specialization: str):
        self.__specialization = specialization

    @property
    def trainerId(self) -> str:
        return self.__trainerId

    @trainerId.setter
    def trainerId(self, trainerId: str):
        self.__trainerId = trainerId

    @property
    def schedule(self):
        return self.__schedule

    @schedule.setter
    def schedule(self, value):
        old_value = getattr(self, f'_Trainer__schedule', None)
        self.__schedule = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'trainer'):
                    opp_val = getattr(item, 'trainer', None)
                    if opp_val == self:
                        setattr(item, 'trainer', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'trainer'):
                    opp_val = getattr(item, 'trainer', None)
                    setattr(item, 'trainer', self)

    def registerTrainer(self):
        print(f'Trainer registered: {self.name}')
        print(f'ID: {self.trainerId}, Specialization: {self.specialization}')
        print(f'Rate: ${self.hourlyRate:.2f}/hour')

    def getTrainerInfo(self):
        return f'Trainer: {self.name}\nSpecialization: {self.specialization}\nRate: ${self.hourlyRate:.2f}/hour'

    def updateRate(self, newRate):
        old_rate = self.hourlyRate
        self.hourlyRate = newRate
        print(f'Hourly rate updated for {self.name}: ${old_rate:.2f} -> ${newRate:.2f}')

class FitnessMember:

    def __init__(self, memberId: str, name: str, membershipType: str, active: bool, sessions: set['TrainingSession']=None):
        self.memberId = memberId
        self.name = name
        self.membershipType = membershipType
        self.active = active
        self.sessions = sessions if sessions is not None else set()

    @property
    def active(self) -> bool:
        return self.__active

    @active.setter
    def active(self, active: bool):
        self.__active = active

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def membershipType(self) -> str:
        return self.__membershipType

    @membershipType.setter
    def membershipType(self, membershipType: str):
        self.__membershipType = membershipType

    @property
    def memberId(self) -> str:
        return self.__memberId

    @memberId.setter
    def memberId(self, memberId: str):
        self.__memberId = memberId

    @property
    def sessions(self):
        return self.__sessions

    @sessions.setter
    def sessions(self, value):
        old_value = getattr(self, f'_FitnessMember__sessions', None)
        self.__sessions = value if value is not None else set()
        if old_value is not None:
            for item in old_value:
                if hasattr(item, 'member'):
                    opp_val = getattr(item, 'member', None)
                    if opp_val == self:
                        setattr(item, 'member', None)
        if value is not None:
            for item in value:
                if hasattr(item, 'member'):
                    opp_val = getattr(item, 'member', None)
                    setattr(item, 'member', self)

    def registerMember(self):
        self.active = True
        print(f'Member registered: {self.name}')
        print(f'ID: {self.memberId}, Membership: {self.membershipType}')

    def suspendMembership(self):
        self.active = False
        print(f'Membership suspended for {self.name}')

    def upgradeMembership(self, newType):
        old_type = self.membershipType
        self.membershipType = newType
        print(f'Membership upgraded for {self.name}: {old_type} -> {newType}')