"""
BUML Model Example 10: Fitness Center Management System
A simple fitness center system with members, trainers, and sessions
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
basic_lit = EnumerationLiteral(name="BASIC")
premium_lit = EnumerationLiteral(name="PREMIUM")
vip_lit = EnumerationLiteral(name="VIP")

membership_type_enum = Enumeration(
    name="MembershipType",
    literals={basic_lit, premium_lit, vip_lit}
)

# =============================================================================
# 2. Define FitnessMember Attributes
# =============================================================================
fitness_member_id_prop = Property(name="memberId", type=StringType, multiplicity=Multiplicity(1, 1))
fitness_member_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
membership_type_prop = Property(name="membershipType", type=StringType, multiplicity=Multiplicity(1, 1))
member_active_prop = Property(name="active", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Trainer Attributes
# =============================================================================
trainer_id_prop = Property(name="trainerId", type=StringType, multiplicity=Multiplicity(1, 1))
trainer_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
trainer_specialization_prop = Property(name="specialization", type=StringType, multiplicity=Multiplicity(1, 1))
hourly_rate_prop = Property(name="hourlyRate", type=FloatType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define TrainingSession Attributes
# =============================================================================
session_id_prop = Property(name="sessionId", type=StringType, multiplicity=Multiplicity(1, 1))
session_date_prop = Property(name="date", type=StringType, multiplicity=Multiplicity(1, 1))
duration_prop = Property(name="duration", type=IntegerType, multiplicity=Multiplicity(1, 1))
session_status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define FitnessMember Methods
# =============================================================================
register_fitness_member_method = Method(
    name="registerMember",
    parameters=[],
    code="""
def registerMember(self):
    self.active = True
    print(f"Member registered: {self.name}")
    print(f"ID: {self.memberId}, Membership: {self.membershipType}")
"""
)

upgrade_membership_method = Method(
    name="upgradeMembership",
    parameters=[Parameter(name="newType", type=StringType)],
    code="""
def upgradeMembership(self, newType):
    old_type = self.membershipType
    self.membershipType = newType
    print(f"Membership upgraded for {self.name}: {old_type} -> {newType}")
"""
)

suspend_membership_method = Method(
    name="suspendMembership",
    parameters=[],
    code="""
def suspendMembership(self):
    self.active = False
    print(f"Membership suspended for {self.name}")
"""
)

# =============================================================================
# 6. Define Trainer Methods
# =============================================================================
register_trainer_method = Method(
    name="registerTrainer",
    parameters=[],
    code="""
def registerTrainer(self):
    print(f"Trainer registered: {self.name}")
    print(f"ID: {self.trainerId}, Specialization: {self.specialization}")
    print(f"Rate: ${self.hourlyRate:.2f}/hour")
"""
)

update_rate_method = Method(
    name="updateRate",
    parameters=[Parameter(name="newRate", type=FloatType)],
    code="""
def updateRate(self, newRate):
    old_rate = self.hourlyRate
    self.hourlyRate = newRate
    print(f"Hourly rate updated for {self.name}: ${old_rate:.2f} -> ${newRate:.2f}")
"""
)

get_trainer_info_method = Method(
    name="getTrainerInfo",
    parameters=[],
    code="""
def getTrainerInfo(self):
    return f"Trainer: {self.name}\\nSpecialization: {self.specialization}\\nRate: ${self.hourlyRate:.2f}/hour"
"""
)

# =============================================================================
# 7. Define TrainingSession Methods
# =============================================================================
schedule_session_method = Method(
    name="scheduleSession",
    parameters=[],
    code="""
def scheduleSession(self):
    self.status = "Scheduled"
    print(f"Session {self.sessionId} scheduled")
    print(f"Date: {self.date}, Duration: {self.duration} minutes")
"""
)

complete_session_method = Method(
    name="completeSession",
    parameters=[],
    code="""
def completeSession(self):
    self.status = "Completed"
    print(f"Session {self.sessionId} marked as completed")
"""
)

calculate_session_cost_method = Method(
    name="calculateCost",
    parameters=[Parameter(name="trainerRate", type=FloatType)],
    code="""
def calculateCost(self, trainerRate):
    hours = self.duration / 60.0
    cost = hours * trainerRate
    print(f"Session cost: ${cost:.2f} ({hours:.1f} hours x ${trainerRate:.2f}/hour)")
    return cost
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
fitness_member_class = Class(
    name="FitnessMember",
    attributes={fitness_member_id_prop, fitness_member_name_prop, membership_type_prop, member_active_prop},
    methods={register_fitness_member_method, upgrade_membership_method, suspend_membership_method}
)

trainer_class = Class(
    name="Trainer",
    attributes={trainer_id_prop, trainer_name_prop, trainer_specialization_prop, hourly_rate_prop},
    methods={register_trainer_method, update_rate_method, get_trainer_info_method}
)

training_session_class = Class(
    name="TrainingSession",
    attributes={session_id_prop, session_date_prop, duration_prop, session_status_prop},
    methods={schedule_session_method, complete_session_method, calculate_session_cost_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# FitnessMember --< TrainingSession (one member can attend many sessions)
member_end = Property(
    name="member",
    type=fitness_member_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
member_sessions_end = Property(
    name="sessions",
    type=training_session_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
member_session_assoc = BinaryAssociation(
    name="Attends",
    ends={member_end, member_sessions_end}
)

# Trainer --< TrainingSession (one trainer can conduct many sessions)
trainer_end = Property(
    name="trainer",
    type=trainer_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
trainer_sessions_end = Property(
    name="schedule",
    type=training_session_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
trainer_session_assoc = BinaryAssociation(
    name="Conducts",
    ends={trainer_end, trainer_sessions_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
fitness_model = DomainModel(
    name="FitnessCenterSystem",
    types={fitness_member_class, trainer_class, training_session_class, membership_type_enum},
    associations={member_session_assoc, trainer_session_assoc}
)

print("✓ Fitness Center Management System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in fitness_model.get_classes()]}")
print(f"  Associations: {[a.name for a in fitness_model.associations]}")


from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=fitness_model, output_dir="output_fitness")
python_gen.generate()