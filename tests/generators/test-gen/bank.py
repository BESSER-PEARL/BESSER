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
from besser.generators.python_classes.python_classes_generator import PythonGenerator
from besser.generators.testgen.test_generator import TestGenerator  # [5]

# Classes
Transaction = Class(name="Transaction")
Account = Class(name="Account")
Customer = Class(name="Customer")
Branch = Class(name="Branch")

# Transaction class attributes and methods
Transaction_transactionId: Property = Property(name="transactionId", type=StringType)
Transaction_transactionType: Property = Property(name="transactionType", type=StringType)
Transaction_amount: Property = Property(name="amount", type=FloatType)
Transaction_transactionDate: Property = Property(name="transactionDate", type=DateType)
Transaction.attributes={Transaction_transactionId, Transaction_transactionDate, Transaction_amount, Transaction_transactionType}

# Account class attributes and methods
Account_accountType: Property = Property(name="accountType", type=StringType)
Account_balance: Property = Property(name="balance", type=FloatType)
Account_accountNumber: Property = Property(name="accountNumber", type=StringType)
Account_m_deposit: Method = Method(name="deposit", parameters={Parameter(name='amount', type=FloatType)}, implementation_type=MethodImplementationType.NONE)
Account_m_withdraw: Method = Method(name="withdraw", parameters={Parameter(name='amount', type=FloatType)}, implementation_type=MethodImplementationType.NONE)
Account.attributes={Account_accountType, Account_balance, Account_accountNumber}
Account.methods={Account_m_withdraw, Account_m_deposit}

# Customer class attributes and methods
Customer_email: Property = Property(name="email", type=StringType)
Customer_customerId: Property = Property(name="customerId", type=StringType)
Customer_phoneNumber: Property = Property(name="phoneNumber", type=StringType)
Customer_name: Property = Property(name="name", type=StringType)
Customer.attributes={Customer_customerId, Customer_name, Customer_email, Customer_phoneNumber}

# Branch class attributes and methods
Branch_branchId: Property = Property(name="branchId", type=StringType)
Branch_location: Property = Property(name="location", type=StringType)
Branch_branchName: Property = Property(name="branchName", type=StringType)
Branch.attributes={Branch_branchName, Branch_location, Branch_branchId}

# Relationships
maintains: BinaryAssociation = BinaryAssociation(
    name="maintains",
    ends={
        Property(name="branch", type=Branch, multiplicity=Multiplicity(1, 1)),
        Property(name="maintains", type=Account, multiplicity=Multiplicity(0, 9999))
    }
)
owns: BinaryAssociation = BinaryAssociation(
    name="owns",
    ends={
        Property(name="owns", type=Account, multiplicity=Multiplicity(0, 9999)),
        Property(name="customer", type=Customer, multiplicity=Multiplicity(1, 1))
    }
)
records: BinaryAssociation = BinaryAssociation(
    name="records",
    ends={
        Property(name="records", type=Transaction, multiplicity=Multiplicity(0, 9999)),
        Property(name="account", type=Account, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Transaction, Account, Customer, Branch},
    associations={maintains, owns, records},
    generalizations={},
    metadata=None
)


from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, QuantumRegister, ClassicalRegister,
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SGate, TGate, SwapGate, RXGate, RYGate, RZGate, PhaseGate,
    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,
    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,
    PrimitiveGate, ParametricGate, OrderGate, ScalarGate, FunctionGate,
    GateDefinition
)





######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="New project")
project = Project(
    name="aa",
    models=[domain_model],
    owner="a",
    metadata=metadata
)
python_gen = PythonGenerator(model=domain_model, output_dir="output_bank")
python_gen.generate()
# Produces: output/classes.py

# =============================================================================
# 9. Step 2 — Generate Hypothesis + pytest test suite [5]
# =============================================================================
test_gen = TestGenerator(model=domain_model, output_dir="output_bank")
test_gen.generate()