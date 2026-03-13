"""
BUML Model Example 8: Car Rental System
A simple car rental system with vehicles, customers, and rentals
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation,Constraint
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
sedan_lit = EnumerationLiteral(name="SEDAN")
suv_lit = EnumerationLiteral(name="SUV")
truck_lit = EnumerationLiteral(name="TRUCK")
sports_lit = EnumerationLiteral(name="SPORTS")

vehicle_category_enum = Enumeration(
    name="VehicleCategory",
    literals={sedan_lit, suv_lit, truck_lit, sports_lit}
)

# =============================================================================
# 2. Define Vehicle Attributes
# =============================================================================
vehicle_id_prop = Property(name="vehicleId", type=StringType, multiplicity=Multiplicity(1, 1))
make_prop = Property(name="make", type=StringType, multiplicity=Multiplicity(1, 1))
model_prop = Property(name="model", type=StringType, multiplicity=Multiplicity(1, 1))
daily_rate_prop = Property(name="dailyRate", type=FloatType, multiplicity=Multiplicity(1, 1))
vehicle_available_prop = Property(name="available", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define RentalCustomer Attributes
# =============================================================================
rental_customer_id_prop = Property(name="customerId", type=StringType, multiplicity=Multiplicity(1, 1))
rental_customer_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
license_number_prop = Property(name="licenseNumber", type=StringType, multiplicity=Multiplicity(1, 1))
customer_verified_prop = Property(name="verified", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Rental Attributes
# =============================================================================
rental_id_prop = Property(name="rentalId", type=StringType, multiplicity=Multiplicity(1, 1))
start_date_prop = Property(name="startDate", type=StringType, multiplicity=Multiplicity(1, 1))
end_date_prop = Property(name="endDate", type=StringType, multiplicity=Multiplicity(1, 1))
rental_total_cost_prop = Property(name="totalCost", type=FloatType, multiplicity=Multiplicity(1, 1))
rental_status_prop = Property(name="status", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Vehicle Methods
# =============================================================================
register_vehicle_method = Method(
    name="registerVehicle",
    parameters=[],
    code="""
def registerVehicle(self):
    self.available = True
    print(f"Vehicle registered: {self.make} {self.model}")
    print(f"ID: {self.vehicleId}, Daily Rate: ${self.dailyRate:.2f}")
"""
)

set_vehicle_availability_method = Method(
    name="setAvailability",
    parameters=[Parameter(name="status", type=BooleanType)],
    code="""
def setAvailability(self, status):
    self.available = status
    status_text = "available" if status else "rented"
    print(f"{self.make} {self.model} is now {status_text}")
"""
)

calculate_rental_cost_method = Method(
    name="calculateRentalCost",
    parameters=[Parameter(name="days", type=IntegerType)],
    code="""
def calculateRentalCost(self, days):
    total = self.dailyRate * days
    print(f"Rental cost for {days} day(s): ${total:.2f}")
    return total
"""
)

# =============================================================================
# 6. Define RentalCustomer Methods
# =============================================================================
register_rental_customer_method = Method(
    name="registerCustomer",
    parameters=[],
    code="""
def registerCustomer(self):
    self.verified = False
    print(f"Customer registered: {self.name}")
    print(f"ID: {self.customerId}, License: {self.licenseNumber}")
"""
)

verify_license_method = Method(
    name="verifyLicense",
    parameters=[],
    code="""
def verifyLicense(self):
    self.verified = True
    print(f"License verified for customer {self.name}")
"""
)

get_rental_customer_info_method = Method(
    name="getCustomerInfo",
    parameters=[],
    code="""
def getCustomerInfo(self):
    status = "Verified" if self.verified else "Not Verified"
    return f"Customer: {self.name} ({status})\\nLicense: {self.licenseNumber}"
"""
)

# =============================================================================
# 7. Define Rental Methods
# =============================================================================
start_rental_method = Method(
    name="startRental",
    parameters=[],
    code="""
def startRental(self):
    self.status = "Active"
    print(f"Rental {self.rentalId} started on {self.startDate}")
    print(f"End date: {self.endDate}, Total cost: ${self.totalCost:.2f}")
"""
)

complete_rental_method = Method(
    name="completeRental",
    parameters=[Parameter(name="value", type=IntegerType)],
    code="""
def completeRental(self,value):
    self.status = value
    print(f"Rental {self.rentalId} has been completed")
"""
)

extend_rental_method = Method(
    name="extendRental",
    parameters=[
        Parameter(name="newEndDate", type=StringType),
        Parameter(name="additionalCost", type=FloatType)
    ],
    code="""
def extendRental(self, newEndDate, additionalCost):
    self.endDate = newEndDate
    self.totalCost += additionalCost
    print(f"Rental {self.rentalId} extended to {newEndDate}")
    print(f"New total cost: ${self.totalCost:.2f}")
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
vehicle_class = Class(
    name="Vehicle",
    attributes={vehicle_id_prop, make_prop, model_prop, daily_rate_prop, vehicle_available_prop},
    methods={register_vehicle_method, set_vehicle_availability_method, calculate_rental_cost_method}
)

rental_customer_class = Class(
    name="RentalCustomer",
    attributes={rental_customer_id_prop, rental_customer_name_prop, license_number_prop, customer_verified_prop},
    methods={register_rental_customer_method, verify_license_method, get_rental_customer_info_method}
)

rental_class = Class(
    name="Rental",
    attributes={rental_id_prop, start_date_prop, end_date_prop, rental_total_cost_prop, rental_status_prop},
    methods={start_rental_method, complete_rental_method, extend_rental_method}
)
constraint_1: Constraint = Constraint(
    name="constraint_1",
    context=vehicle_class,
    expression="context Vehicle::setAvailability(value:bool) post set: self.available = value",
    language="OCL"
)
constraint_2: Constraint = Constraint(
    name="constraint_2",
    context=rental_class,
    expression="context Rental::completeRental(value:bool) post set: self.status =value",
    language="OCL"
)
# =============================================================================
# 9. Define Associations
# =============================================================================
# RentalCustomer --< Rental (one customer can have many rentals)
rental_customer_end = Property(
    name="customer",
    type=rental_customer_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
customer_rentals_end = Property(
    name="rentals",
    type=rental_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
customer_rental_assoc = BinaryAssociation(
    name="Initiates",
    ends={rental_customer_end, customer_rentals_end}
)

# Vehicle --< Rental (one vehicle can have many rentals over time)
vehicle_end = Property(
    name="vehicle",
    type=vehicle_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
vehicle_rentals_end = Property(
    name="rentalHistory",
    type=rental_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
vehicle_rental_assoc = BinaryAssociation(
    name="IsRentedIn",
    ends={vehicle_end, vehicle_rentals_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
car_rental_model = DomainModel(
    name="CarRentalSystem",
    types={vehicle_class, rental_customer_class, rental_class, vehicle_category_enum},
    associations={customer_rental_assoc, vehicle_rental_assoc}
    , constraints={constraint_1, constraint_2}

)

print("✓ Car Rental System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in car_rental_model.get_classes()]}")
print(f"  Associations: {[a.name for a in car_rental_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=car_rental_model, output_dir="output_car_rental")
python_gen.generate()

from besser.generators.testgen.test_generator import TestGenerator
generator = TestGenerator(model=car_rental_model, output_dir="output_car_rental")
generator.generate()