"""
BUML Model Example 1: Library Management System
A simple library system with books, members, and loans
"""
from besser.generators.python_classes.python_classes_generator import PythonGenerator
from besser.generators.testgen.test_generator import TestGenerator  # [5]

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    StringType, IntegerType, FloatType, BooleanType,
    Multiplicity, Enumeration, EnumerationLiteral,
    BinaryAssociation,Constraint
)

# =============================================================================
# 1. Define Enumerations
# =============================================================================
available_lit = EnumerationLiteral(name="AVAILABLE")
checked_out_lit = EnumerationLiteral(name="CHECKED_OUT")
reserved_lit = EnumerationLiteral(name="RESERVED")

book_status_enum = Enumeration(
    name="BookStatus",
    literals={available_lit, checked_out_lit, reserved_lit}
)

# =============================================================================
# 2. Define Book Attributes
# =============================================================================
isbn_prop = Property(name="isbn", type=StringType, multiplicity=Multiplicity(1, 1))
book_title_prop = Property(name="title", type=StringType, multiplicity=Multiplicity(1, 1))
author_name_prop = Property(name="author", type=StringType, multiplicity=Multiplicity(1, 1))
book_available_prop = Property(name="available", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 3. Define Member Attributes
# =============================================================================
member_id_prop = Property(name="memberId", type=StringType, multiplicity=Multiplicity(1, 1))
member_name_prop = Property(name="name", type=StringType, multiplicity=Multiplicity(1, 1))
member_email_prop = Property(name="email", type=StringType, multiplicity=Multiplicity(1, 1))
max_books_prop = Property(name="maxBooks", type=IntegerType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 4. Define Loan Attributes
# =============================================================================
loan_id_prop = Property(name="loanId", type=StringType, multiplicity=Multiplicity(1, 1))
loan_date_prop = Property(name="loanDate", type=StringType, multiplicity=Multiplicity(1, 1))
due_date_prop = Property(name="dueDate", type=StringType, multiplicity=Multiplicity(1, 1))
returned_prop = Property(name="returned", type=BooleanType, multiplicity=Multiplicity(1, 1))

# =============================================================================
# 5. Define Book Methods
# =============================================================================
get_info_method = Method(
    name="getInfo",
    parameters=[],
    code="""
def getInfo(self):
    return f"{self.title} by {self.author} (ISBN: {self.isbn})"
"""
)

check_availability_method = Method(
    name="getAvailability",
    parameters=[],
    code="""
def checkAvailability(self):
    return self.available
"""
)

set_availability_method = Method(
    name="setAvailability",
    parameters=[Parameter(name="status", type=BooleanType)],
    code="""
def setAvailability(self, status):
    self.available = status
    print(f"Book '{self.title}' availability set to {status}")
"""
)

# =============================================================================
# 6. Define Member Methods
# =============================================================================
register_member_method = Method(
    name="register",
    parameters=[],
    code="""
def getregister(self):
    print(f"Member {self.name} registered with ID: {self.memberId}")
    print(f"Email: {self.email}")
"""
)

can_borrow_method = Method(
    name="canBorrow",
    parameters=[Parameter(name="currentLoans", type=IntegerType)],
    code="""
def canBorrow(self, currentLoans):
    if currentLoans < self.maxBooks:
        return True
    else:
        print(f"Member {self.name} has reached maximum loan limit")
        return False
"""
)

update_email_method = Method(
    name="updateEmail",
    parameters=[Parameter(name="newEmail", type=StringType)],
    code="""
def updateEmail(self, newEmail):
    self.email = newEmail
    print(f"Email updated to {newEmail} for member {self.name}")
"""
)

# =============================================================================
# 7. Define Loan Methods
# =============================================================================
create_loan_method = Method(
    name="createLoan",
    parameters=[Parameter(name="val", type=BooleanType)],
    code="""
def createLoan(self,val):
    self.returned =val
    print(f"Loan {self.loanId} created on {self.loanDate}")
    print(f"Due date: {self.dueDate}")
"""
)

return_book_method = Method(
    name="returnBook",
    parameters=[],
    code="""
def returnBook(self):
    self.returned = True
    print(f"Loan {self.loanId} marked as returned")
"""
)

is_overdue_method = Method(
    name="isOverdue",
    parameters=[Parameter(name="currentDate", type=StringType)],
    code="""
def isOverdue(self, currentDate):
    if not self.returned and currentDate > self.dueDate:
        print(f"Loan {self.loanId} is overdue!")
        return True
    return False
"""
)

# =============================================================================
# 8. Define Classes
# =============================================================================
book_class = Class(
    name="Book",
    attributes={isbn_prop, book_title_prop, author_name_prop, book_available_prop},
    methods={get_info_method, check_availability_method, set_availability_method}
)

member_class = Class(
    name="Member",
    attributes={member_id_prop, member_name_prop, member_email_prop, max_books_prop},
    methods={register_member_method, can_borrow_method, update_email_method}
)

constraint_updateEmail: Constraint = Constraint(
    name="constraint_updateEmail",
    context=member_class,
    expression="context Member::updateEmail(value:bool) post changeLoan: self.email = value",
    language="OCL"
)

loan_class = Class(
    name="Loan",
    attributes={loan_id_prop, loan_date_prop, due_date_prop, returned_prop},
    methods={create_loan_method, return_book_method, is_overdue_method}
)


constraint_create_loan: Constraint = Constraint(
    name="constraint_create_loan",
    context=loan_class,
    expression="context Loan::createLoan(value:bool) post changeLoan: self.returned = value",
    language="OCL"
)
# =============================================================================
# 9. Define Associations
# =============================================================================
# Member --< Loan (one member can have many loans)
member_end = Property(
    name="member",
    type=member_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
member_loans_end = Property(
    name="loans",
    type=loan_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
member_loan_assoc = BinaryAssociation(
    name="Borrows",
    ends={member_end, member_loans_end}
)

# Book --< Loan (one book can be in many loans over time)
book_end = Property(
    name="book",
    type=book_class,
    multiplicity=Multiplicity(1, 1),
    is_navigable=True
)
book_loans_end = Property(
    name="loanHistory",
    type=loan_class,
    multiplicity=Multiplicity(0, "*"),
    is_navigable=True
)
book_loan_assoc = BinaryAssociation(
    name="IsLoanedIn",
    ends={book_end, book_loans_end}
)

# =============================================================================
# 10. Build the DomainModel
# =============================================================================
library_model = DomainModel(
    name="LibraryManagementSystem",
    types={book_class, member_class, loan_class, book_status_enum},
    associations={member_loan_assoc, book_loan_assoc},
    constraints={constraint_updateEmail,constraint_create_loan}
)

print("✓ Library Management System BUML Model created successfully!")
print(f"  Classes: {[c.name for c in library_model.get_classes()]}")
print(f"  Associations: {[a.name for a in library_model.associations]}")
from besser.generators.python_classes.python_classes_generator import PythonGenerator
python_gen = PythonGenerator(model=library_model, output_dir="output_library")
# python_gen.generate()

from besser.generators.testgen.test_generator import TestGenerator
generator = TestGenerator(model=library_model, output_dir="output_library")
generator.generate()

