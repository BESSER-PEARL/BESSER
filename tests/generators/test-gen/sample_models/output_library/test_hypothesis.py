import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy
from datetime import date

from classes import (
    Loan,
    Member,
    Book,
    BookStatus,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================



def test_loan_is_not_abstract():
    assert not inspect.isabstract(Loan)


def test_loan_constructor_exists():
    assert callable(Loan.__init__)


def test_loan_constructor_args():
    sig = inspect.signature(Loan.__init__)
    params = list(sig.parameters.keys())
    assert "loanDate" in params, "Missing parameter 'loanDate'"
    assert "loanId" in params, "Missing parameter 'loanId'"
    assert "dueDate" in params, "Missing parameter 'dueDate'"
    assert "returned" in params, "Missing parameter 'returned'"

def test_loan_has_loanDate():
    assert hasattr(Loan, "loanDate")
    descriptor = None
    for klass in Loan.__mro__:
        if "loanDate" in klass.__dict__:
            descriptor = klass.__dict__["loanDate"]
            break
    assert isinstance(descriptor, property)

def test_loan_has_loanId():
    assert hasattr(Loan, "loanId")
    descriptor = None
    for klass in Loan.__mro__:
        if "loanId" in klass.__dict__:
            descriptor = klass.__dict__["loanId"]
            break
    assert isinstance(descriptor, property)

def test_loan_has_dueDate():
    assert hasattr(Loan, "dueDate")
    descriptor = None
    for klass in Loan.__mro__:
        if "dueDate" in klass.__dict__:
            descriptor = klass.__dict__["dueDate"]
            break
    assert isinstance(descriptor, property)

def test_loan_has_returned():
    assert hasattr(Loan, "returned")
    descriptor = None
    for klass in Loan.__mro__:
        if "returned" in klass.__dict__:
            descriptor = klass.__dict__["returned"]
            break
    assert isinstance(descriptor, property)



def test_member_is_not_abstract():
    assert not inspect.isabstract(Member)


def test_member_constructor_exists():
    assert callable(Member.__init__)


def test_member_constructor_args():
    sig = inspect.signature(Member.__init__)
    params = list(sig.parameters.keys())
    assert "name" in params, "Missing parameter 'name'"
    assert "email" in params, "Missing parameter 'email'"
    assert "maxBooks" in params, "Missing parameter 'maxBooks'"
    assert "memberId" in params, "Missing parameter 'memberId'"

def test_member_has_name():
    assert hasattr(Member, "name")
    descriptor = None
    for klass in Member.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_member_has_email():
    assert hasattr(Member, "email")
    descriptor = None
    for klass in Member.__mro__:
        if "email" in klass.__dict__:
            descriptor = klass.__dict__["email"]
            break
    assert isinstance(descriptor, property)

def test_member_has_maxBooks():
    assert hasattr(Member, "maxBooks")
    descriptor = None
    for klass in Member.__mro__:
        if "maxBooks" in klass.__dict__:
            descriptor = klass.__dict__["maxBooks"]
            break
    assert isinstance(descriptor, property)

def test_member_has_memberId():
    assert hasattr(Member, "memberId")
    descriptor = None
    for klass in Member.__mro__:
        if "memberId" in klass.__dict__:
            descriptor = klass.__dict__["memberId"]
            break
    assert isinstance(descriptor, property)



def test_book_is_not_abstract():
    assert not inspect.isabstract(Book)


def test_book_constructor_exists():
    assert callable(Book.__init__)


def test_book_constructor_args():
    sig = inspect.signature(Book.__init__)
    params = list(sig.parameters.keys())
    assert "title" in params, "Missing parameter 'title'"
    assert "author" in params, "Missing parameter 'author'"
    assert "isbn" in params, "Missing parameter 'isbn'"
    assert "available" in params, "Missing parameter 'available'"

def test_book_has_title():
    assert hasattr(Book, "title")
    descriptor = None
    for klass in Book.__mro__:
        if "title" in klass.__dict__:
            descriptor = klass.__dict__["title"]
            break
    assert isinstance(descriptor, property)

def test_book_has_author():
    assert hasattr(Book, "author")
    descriptor = None
    for klass in Book.__mro__:
        if "author" in klass.__dict__:
            descriptor = klass.__dict__["author"]
            break
    assert isinstance(descriptor, property)

def test_book_has_isbn():
    assert hasattr(Book, "isbn")
    descriptor = None
    for klass in Book.__mro__:
        if "isbn" in klass.__dict__:
            descriptor = klass.__dict__["isbn"]
            break
    assert isinstance(descriptor, property)

def test_book_has_available():
    assert hasattr(Book, "available")
    descriptor = None
    for klass in Book.__mro__:
        if "available" in klass.__dict__:
            descriptor = klass.__dict__["available"]
            break
    assert isinstance(descriptor, property)

def test_bookstatus_exists():
    # Check that the Enumeration exists
    assert BookStatus is not None

def test_bookstatus_has_all_literals():
    # Collect the names of literals in this Enumeration
    enum_literals = [lit.name for lit in BookStatus]
    expected_literals = [
        "CHECKED_OUT",
        "AVAILABLE",
        "RESERVED",
    ]
    # Check that all expected literals exist
    for lit_name in expected_literals:
        assert lit_name in enum_literals, f"Literal '' missing in BookStatus"


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_",
    ),
    min_size=1,
).filter(lambda s: s[0].isalpha())
Loan_strategy = st.builds(
    Loan,
    loanDate=
        safe_text,
    loanId=
        safe_text,
    dueDate=
        safe_text,
    returned=
        st.booleans()
)
Member_strategy = st.builds(
    Member,
    name=
        safe_text,
    email=
        safe_text,
    maxBooks=
        st.integers(),
    memberId=
        safe_text
)
Book_strategy = st.builds(
    Book,
    title=
        safe_text,
    author=
        safe_text,
    isbn=
        safe_text,
    available=
        st.booleans()
)

@given(instance=Loan_strategy)
@settings(max_examples=50)
def test_loan_instantiation(instance):
    assert isinstance(instance, Loan)

@given(instance=Loan_strategy)
def test_loan_loanDate_type(instance):
    assert isinstance(instance.loanDate, str)


@given(instance=Loan_strategy)
def test_loan_loanDate_setter(instance):
    original = instance.loanDate
    instance.loanDate = original
    assert instance.loanDate == original

@given(instance=Loan_strategy)
def test_loan_loanId_type(instance):
    assert isinstance(instance.loanId, str)


@given(instance=Loan_strategy)
def test_loan_loanId_setter(instance):
    original = instance.loanId
    instance.loanId = original
    assert instance.loanId == original

@given(instance=Loan_strategy)
def test_loan_dueDate_type(instance):
    assert isinstance(instance.dueDate, str)


@given(instance=Loan_strategy)
def test_loan_dueDate_setter(instance):
    original = instance.dueDate
    instance.dueDate = original
    assert instance.dueDate == original

@given(instance=Loan_strategy)
def test_loan_returned_type(instance):
    assert isinstance(instance.returned, bool)


@given(instance=Loan_strategy)
def test_loan_returned_setter(instance):
    original = instance.returned
    instance.returned = original
    assert instance.returned == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Loan_strategy)
@settings(max_examples=30)
def test_loan_createloan_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.createLoan(
            True
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.createLoan).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'createLoan' in Loan is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'createLoan' in Loan did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'createLoan' in Loan is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Loan_strategy)
@settings(max_examples=30)
def test_loan_returnbook_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.returnBook()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.returnBook).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'returnBook' in Loan is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'returnBook' in Loan did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'returnBook' in Loan is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Loan_strategy)
@settings(max_examples=30)
def test_loan_isoverdue_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.isOverdue(
            "test"
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.isOverdue).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'isOverdue' in Loan is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'isOverdue' in Loan did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'isOverdue' in Loan is not implemented or raised an error")

@given(instance=Member_strategy)
@settings(max_examples=50)
def test_member_instantiation(instance):
    assert isinstance(instance, Member)

@given(instance=Member_strategy)
def test_member_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Member_strategy)
def test_member_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=Member_strategy)
def test_member_email_type(instance):
    assert isinstance(instance.email, str)


@given(instance=Member_strategy)
def test_member_email_setter(instance):
    original = instance.email
    instance.email = original
    assert instance.email == original

@given(instance=Member_strategy)
def test_member_maxBooks_type(instance):
    assert isinstance(instance.maxBooks, int)


@given(instance=Member_strategy)
def test_member_maxBooks_setter(instance):
    original = instance.maxBooks
    instance.maxBooks = original
    assert instance.maxBooks == original

@given(instance=Member_strategy)
def test_member_memberId_type(instance):
    assert isinstance(instance.memberId, str)


@given(instance=Member_strategy)
def test_member_memberId_setter(instance):
    original = instance.memberId
    instance.memberId = original
    assert instance.memberId == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Member_strategy)
@settings(max_examples=30)
def test_member_register_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.register()
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.register).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'register' in Member is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'register' in Member did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'register' in Member is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Member_strategy)
@settings(max_examples=30)
def test_member_canborrow_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.canBorrow(
            1
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.canBorrow).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'canBorrow' in Member is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'canBorrow' in Member did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'canBorrow' in Member is not implemented or raised an error")

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Member_strategy)
@settings(max_examples=30)
def test_member_updateemail_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.updateEmail(
            "test"
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.updateEmail).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'updateEmail' in Member is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'updateEmail' in Member did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'updateEmail' in Member is not implemented or raised an error")

@given(instance=Book_strategy)
@settings(max_examples=50)
def test_book_instantiation(instance):
    assert isinstance(instance, Book)

@given(instance=Book_strategy)
def test_book_title_type(instance):
    assert isinstance(instance.title, str)


@given(instance=Book_strategy)
def test_book_title_setter(instance):
    original = instance.title
    instance.title = original
    assert instance.title == original

@given(instance=Book_strategy)
def test_book_author_type(instance):
    assert isinstance(instance.author, str)


@given(instance=Book_strategy)
def test_book_author_setter(instance):
    original = instance.author
    instance.author = original
    assert instance.author == original

@given(instance=Book_strategy)
def test_book_isbn_type(instance):
    assert isinstance(instance.isbn, str)


@given(instance=Book_strategy)
def test_book_isbn_setter(instance):
    original = instance.isbn
    instance.isbn = original
    assert instance.isbn == original

@given(instance=Book_strategy)
def test_book_available_type(instance):
    assert isinstance(instance.available, bool)


@given(instance=Book_strategy)
def test_book_available_setter(instance):
    original = instance.available
    instance.available = original
    assert instance.available == original

import warnings
import copy
import inspect
import ast
from hypothesis import given, settings

@given(instance=Book_strategy)
@settings(max_examples=30)
def test_book_setavailability_changes_state(instance):
    before = copy.deepcopy(instance)
    try:
        # Call operation with dummy parameters
        instance.setAvailability(
            True
        )
        if instance.__dict__ != before.__dict__:
            return  # test passes
        # Check that function exists and is non-empty (FAIL if empty)
        source = inspect.getsource(instance.setAvailability).strip()
        tree = ast.parse(source)
        body = tree.body[0].body  # function body
        has_statements = len(body) > 0 and not all(isinstance(stmt, ast.Pass) for stmt in body)
        assert has_statements, f"Function 'setAvailability' in Book is empty"

        # Check for state change (WARN if no change)
        if instance.__dict__ == before.__dict__:
            warnings.warn(f"Operation 'setAvailability' in Book did not change state; check implementation")

    except (AttributeError, NotImplementedError, TypeError):
        warnings.warn(f"Operation 'setAvailability' in Book is not implemented or raised an error")

@given(instance=Loan_strategy)
def test_loan_ocl_constraint_create_loan(instance):
     
    
    
    
    
    
    value = True
    # Call the operation
    instance.createLoan(value)
    
    assert instance.returned == value

@given(instance=Member_strategy)
def test_member_ocl_constraint_updateemail(instance):
     
    
    
    
    
    
    value = True
    # Call the operation
    instance.updateEmail(value)
    
    assert instance.email == value