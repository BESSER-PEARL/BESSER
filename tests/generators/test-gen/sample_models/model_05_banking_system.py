"""
BUML Model Example 5: Banking System
A simple banking system with accounts, customers, and transactions
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
deposit_lit = EnumerationLiteral(name="DEPOSIT")
withdrawal_lit = EnumerationLiteral(name="WITHDRAWAL")
transfer_lit = EnumerationLiteral(name="TRANSFER")
payment_lit = EnumerationLiteral(name="PAYMENT")

transaction_type_enum = Enumeration(
    name="TransactionType",
    literals={deposit_lit, withdrawal_lit, transfer_lit, payment_lit}
)

# =============================================================================
# 2. Define BankAccount Attributes
# =============================================================================
account_number_prop = Property(name="accountNumber", type=StringType, multiplicity=Multiplicity(1, 1))
balance_prop = Property(name="balance", type=FloatType, multiplicity=Multiplicity(1, 1))
account_type_prop = Property(name="accountType", type=StringType, multiplicity=Multiplicity(1, 1))
active_prop = Property(name="active", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Customer Attributes
# =============================================================================
customer_id_prop = Property(name="customerId", type=StringType, multiplicity=Multiplicity(1, 1))
customer_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
phone_prop = Property(name="phone", type=StringType, multiplicity=Multiplicity(1, 1))
verified_prop = Property(name="verified", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Transaction Attributes
# =============================================================================
transaction_id_prop = Property(name="transactionId", type=StringType, multiplicity=Multiplicity(1, 1))
transaction_type_prop = Property(name="transactionType", type=StringType, multiplicity=Multiplicity(1, 1))
amount_prop = Property(name="amount", type=FloatType, multiplicity=Multiplicity(1, 1))
timestamp_prop = Property(name="timestamp", type=StringType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define BankAccount Methods
# =============================================================================
open_account_method = Method(
    name="openAccount",
    parameters=[Parameter(name="initialDeposit", type=FloatType)],
    code="""
def openAccount(self, initialDeposit):
    self.balance = initialDeposit
    self.active = True
    print(f"Account {self.accountNumber} opened with balance: ${self.balance:.2f}")
"""
)

deposit_method = Method(
    name="deposit",
    parameters=[Parameter(name="amount", type=FloatType)],
    code="""
def deposit(self, amount):
    if amount > 0:
        self.balance += amount
        print(f"Deposited ${amount:.2f}. New balance: ${self.balance:.2f}")
    else:
        print("Invalid deposit amount")
"""
)

withdraw_method = Method(
    name="withdraw",
    parameters=[Parameter(name="amount", type=FloatType)],
    code="""
def withdraw(self, amount):
    if amount > 0 and self.balance >= amount:
        self.balance -= amount
        print(f"Withdrew ${amount:.2f}. New balance: ${self.balance:.2f}")
        return True
    else:
        print(f"Insufficient funds or invalid amount")
        return False
"""
)

# =============================================================================
# 6. Define Customer Methods
# =============================================================================
register_customer_method = Method(
    name="registerCustomer",
    parameters=[],
    code="""
def registerCustomer(self):
    self.verified = False
    print(f"Customer registered: {self.name}")
    print(f"ID: {self.customerId}, Phone: {self.phone}")
"""
)

verify_identity_method = Method(
    name="verifyIdentity",
    parameters=[],
    code="""
def verifyIdentity(self):
    self.verified = True
    print(f"Customer {self.name} has been verified")
"""
)

update_phone_method = Method(
    name="updatePhone",
    parameters=[Parameter(name="newPhone", type=StringType)],
    code="""
def updatePhone(self, newPhone):
    old_phone = self.phone
    self.phone = newPhone
    print(f"Phone updated from {old_phone} to {newPhone}")
"""
)

# =============================================================================
# 7. Define Transaction Methods
# =============================================================================
process_method = Method(
    name="process",
    parameters=[],
    code="""
def process(self):
    print(f"Processing {self.transactionType} transaction")
    print(f"ID: {self.transactionId}, Amount: ${self.amount:.2f}")
    print(f"Timestamp: {self.timestamp}")
"""
)

validate_method = Method(
    name="validate",
    parameters=[],
    code="""
def validate(self):
    if self.amount > 0:
        print(f"Transaction {self.transactionId} is valid")
        return True
    else:
        print(f"Transaction {self.transactionId} is invalid")
        return False
"""
)

get_details_method = Method(
    name="getDetails",
    parameters=[],
    code="""
def getDetails(self):
    return f"Transaction {self.transactionId}: {self.transactionType} - ${self.amount:.2f} at {self.timestamp}"
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
account_class = Class(
    name="BankAccount",
    attributes={account_number_prop, balance_prop, account_type_prop, active_prop},
    methods={open_account_method, deposit_method, withdraw_method}
)

customer_class = Class(
    name="Customer",
    attributes={customer_id_prop, customer_name_prop, phone_prop, verified_prop},
    methods={register_customer_method, verify_identity_method, update_phone_method}
)

transaction_class = Class(
    name="Transaction",
    attributes={transaction_id_prop, transaction_type_prop, amount_prop, timestamp_prop},
    methods={process_method, validate_method, get_details_method}
)

# =============================================================================
# 9. Define Associations
# =============================================================================
# Customer --< BankAccount (one customer can have many accounts)
customer_end = Property(
    name="owner",
    type=customer_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
customer_accounts_end = Property(
    name="accounts",
    type=account_class,
    multiplicity=Multiplicity(1, "*"),
    is_navigable=True
)
customer_account_assoc = BinaryAssociation(
    name="Owns",
    ends={customer_end, customer_accounts_end}
)

# BankAccount --< Transaction (one account can have many transactions)
account_end = Property(
    name="account",
    type=account_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
account_transactions_end = Property(
    name="transactions",
    type=transaction_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
account_transaction_assoc = BinaryAssociation(
    name="Performs",
    ends={account_end, account_transactions_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
banking_model = DomainModel(
    name="BankingSystem",
    types={account_class, customer_class, transaction_class, transaction_type_enum},
    associations={customer_account_assoc, account_transaction_assoc}
)

print("✓ Banking System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in banking_model.get_classes()]}")
print(f"  Associations: {[a.name for a in banking_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=banking_model, output_dir="output_banking")
python_gen.generate()