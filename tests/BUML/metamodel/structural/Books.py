####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# -----------------------------------------------------------------------------
# 1. ENUMERATIONS
# -----------------------------------------------------------------------------
VehicleStatus: Enumeration = Enumeration(
    name="VehicleStatus",
    literals={
        EnumerationLiteral(name="AVAILABLE"),
        EnumerationLiteral(name="RESERVED"),
        EnumerationLiteral(name="SOLD"),
        EnumerationLiteral(name="UNDER_MAINTENANCE")
    }
)

FuelType: Enumeration = Enumeration(
    name="FuelType",
    literals={
        EnumerationLiteral(name="GASOLINE"),
        EnumerationLiteral(name="DIESEL"),
        EnumerationLiteral(name="ELECTRIC"),
        EnumerationLiteral(name="HYBRID")
    }
)

# -----------------------------------------------------------------------------
# 2. CLASSES AND ATTRIBUTES
# -----------------------------------------------------------------------------
# Abstract Superclass: Person
Person = Class(name="Person", is_abstract=True)
Person_id: Property = Property(name="id", type=StringType)
Person_fullName: Property = Property(name="fullName", type=StringType)
Person_email: Property = Property(name="email", type=StringType)
Person_phone: Property = Property(name="phone", type=StringType)
Person.attributes = {Person_id, Person_fullName, Person_email, Person_phone}

# Subclass: Customer
Customer = Class(name="Customer")
Customer_driverLicenseNumber: Property = Property(name="driverLicenseNumber", type=StringType)
Customer_creditScore: Property = Property(name="creditScore", type=IntegerType)
Customer.attributes = {Customer_driverLicenseNumber, Customer_creditScore}

# Subclass: Salesperson
Salesperson = Class(name="Salesperson")
Salesperson_employeeId: Property = Property(name="employeeId", type=StringType)
Salesperson_commissionRate: Property = Property(name="commissionRate", type=FloatType)
Salesperson.attributes = {Salesperson_employeeId, Salesperson_commissionRate}

# Abstract Superclass: Vehicle
Vehicle = Class(name="Vehicle", is_abstract=True)
Vehicle_vin: Property = Property(name="vin", type=StringType)
Vehicle_brand: Property = Property(name="brand", type=StringType)
Vehicle_model: Property = Property(name="model", type=StringType)
Vehicle_year: Property = Property(name="year", type=IntegerType)
Vehicle_price: Property = Property(name="price", type=FloatType)
Vehicle_fuel: Property = Property(name="fuel", type=FuelType)
Vehicle_status: Property = Property(name="status", type=VehicleStatus)
Vehicle.attributes = {
    Vehicle_vin, Vehicle_brand, Vehicle_model, 
    Vehicle_year, Vehicle_price, Vehicle_fuel, Vehicle_status
}

# Subclass: Car
Car = Class(name="Car")
Car_numDoors: Property = Property(name="numDoors", type=IntegerType)
Car_bodyStyle: Property = Property(name="bodyStyle", type=StringType)
Car.attributes = {Car_numDoors, Car_bodyStyle}

# Concrete Class: SaleTransaction
SaleTransaction = Class(name="SaleTransaction")
SaleTransaction_transactionId: Property = Property(name="transactionId", type=StringType)
SaleTransaction_saleDate: Property = Property(name="saleDate", type=DateType)
SaleTransaction_finalPrice: Property = Property(name="finalPrice", type=FloatType)
SaleTransaction_paymentMethod: Property = Property(name="paymentMethod", type=StringType)
SaleTransaction.attributes = {
    SaleTransaction_transactionId, SaleTransaction_saleDate, 
    SaleTransaction_finalPrice, SaleTransaction_paymentMethod
}

# -----------------------------------------------------------------------------
# 3. GENERALIZATIONS (INHERITANCE)
# -----------------------------------------------------------------------------
gen_customer_person: Generalization = Generalization(general=Person, specific=Customer)
gen_salesperson_person: Generalization = Generalization(general=Person, specific=Salesperson)
gen_car_vehicle: Generalization = Generalization(general=Vehicle, specific=Car)

# -----------------------------------------------------------------------------
# 4. RELATIONSHIPS / ASSOCIATIONS
# -----------------------------------------------------------------------------
# Customer (1) <---> (0..*) SaleTransaction
CustomerPurchases: BinaryAssociation = BinaryAssociation(
    name="CustomerPurchases",
    ends={
        Property(name="buyer", type=Customer, multiplicity=Multiplicity(1, 1)),
        Property(name="purchases", type=SaleTransaction, multiplicity=Multiplicity(0, 9999))
    }
)

# Salesperson (1) <---> (0..*) SaleTransaction
SalespersonTransactions: BinaryAssociation = BinaryAssociation(
    name="SalespersonTransactions",
    ends={
        Property(name="seller", type=Salesperson, multiplicity=Multiplicity(1, 1)),
        Property(name="sales", type=SaleTransaction, multiplicity=Multiplicity(0, 9999))
    }
)

# Vehicle (1) <---> (0..1) SaleTransaction
VehicleSale: BinaryAssociation = BinaryAssociation(
    name="VehicleSale",
    ends={
        Property(name="soldVehicle", type=Vehicle, multiplicity=Multiplicity(1, 1)),
        Property(name="transaction", type=SaleTransaction, multiplicity=Multiplicity(0, 1))
    }
)

# -----------------------------------------------------------------------------
# 5. OCL CONSTRAINTS (Comentadas)
# -----------------------------------------------------------------------------
# Constraint 1: El precio final de la venta debe ser mayor a 0
# context SaleTransaction inv SaleTransaction_inv_PositivePrice: self.finalPrice > 0.0

# Constraint 2: El puntaje crediticio del cliente debe estar entre 300 y 850
# context Customer inv Customer_inv_ValidCreditScore: self.creditScore >= 300 and self.creditScore <= 850

# Constraint 3: El auto debe tener entre 2 y 5 puertas
# context Car inv Car_inv_ValidDoors: self.numDoors >= 2 and self.numDoors <= 5

# -----------------------------------------------------------------------------
# 6. DOMAIN MODEL
# -----------------------------------------------------------------------------
domain_model: DomainModel = DomainModel(
    name="CarDealershipModel",
    types={
        Person, Customer, Salesperson, Vehicle, Car, SaleTransaction,
        VehicleStatus, FuelType
    },
    associations={CustomerPurchases, SalespersonTransactions, VehicleSale},
    generalizations={gen_customer_person, gen_salesperson_person, gen_car_vehicle},
    constraints={},
    metadata=None
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project

metadata = Metadata(description="Car Dealership Management System BUML Model")
project = Project(
    name="CarDealershipManagementSystem",
    models=[domain_model],
    owner="BESSER User",
    metadata=metadata
)