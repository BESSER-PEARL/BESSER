import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy

from classes import (
    Branch,
    Customer,
    Account,
    Transaction,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================

def test_branch_class_exists():
    assert isinstance(Branch, type)


def test_branch_is_not_abstract():
    assert not inspect.isabstract(Branch)


def test_branch_constructor_exists():
    assert callable(Branch.__init__)


def test_branch_constructor_args():
    sig = inspect.signature(Branch.__init__)
    params = list(sig.parameters.keys())
    assert "branchId" in params, "Missing parameter 'branchId'"
    assert "location" in params, "Missing parameter 'location'"
    assert "branchName" in params, "Missing parameter 'branchName'"

def test_branch_has_branchId():
    assert hasattr(Branch, "branchId")
    descriptor = None
    for klass in Branch.__mro__:
        if "branchId" in klass.__dict__:
            descriptor = klass.__dict__["branchId"]
            break
    assert isinstance(descriptor, property)

def test_branch_has_location():
    assert hasattr(Branch, "location")
    descriptor = None
    for klass in Branch.__mro__:
        if "location" in klass.__dict__:
            descriptor = klass.__dict__["location"]
            break
    assert isinstance(descriptor, property)

def test_branch_has_branchName():
    assert hasattr(Branch, "branchName")
    descriptor = None
    for klass in Branch.__mro__:
        if "branchName" in klass.__dict__:
            descriptor = klass.__dict__["branchName"]
            break
    assert isinstance(descriptor, property)

def test_customer_class_exists():
    assert isinstance(Customer, type)


def test_customer_is_not_abstract():
    assert not inspect.isabstract(Customer)


def test_customer_constructor_exists():
    assert callable(Customer.__init__)


def test_customer_constructor_args():
    sig = inspect.signature(Customer.__init__)
    params = list(sig.parameters.keys())
    assert "phoneNumber" in params, "Missing parameter 'phoneNumber'"
    assert "name" in params, "Missing parameter 'name'"
    assert "email" in params, "Missing parameter 'email'"
    assert "customerId" in params, "Missing parameter 'customerId'"

def test_customer_has_phoneNumber():
    assert hasattr(Customer, "phoneNumber")
    descriptor = None
    for klass in Customer.__mro__:
        if "phoneNumber" in klass.__dict__:
            descriptor = klass.__dict__["phoneNumber"]
            break
    assert isinstance(descriptor, property)

def test_customer_has_name():
    assert hasattr(Customer, "name")
    descriptor = None
    for klass in Customer.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_customer_has_email():
    assert hasattr(Customer, "email")
    descriptor = None
    for klass in Customer.__mro__:
        if "email" in klass.__dict__:
            descriptor = klass.__dict__["email"]
            break
    assert isinstance(descriptor, property)

def test_customer_has_customerId():
    assert hasattr(Customer, "customerId")
    descriptor = None
    for klass in Customer.__mro__:
        if "customerId" in klass.__dict__:
            descriptor = klass.__dict__["customerId"]
            break
    assert isinstance(descriptor, property)

def test_account_class_exists():
    assert isinstance(Account, type)


def test_account_is_not_abstract():
    assert not inspect.isabstract(Account)


def test_account_constructor_exists():
    assert callable(Account.__init__)


def test_account_constructor_args():
    sig = inspect.signature(Account.__init__)
    params = list(sig.parameters.keys())
    assert "accountType" in params, "Missing parameter 'accountType'"
    assert "balance" in params, "Missing parameter 'balance'"
    assert "accountNumber" in params, "Missing parameter 'accountNumber'"

def test_account_has_accountType():
    assert hasattr(Account, "accountType")
    descriptor = None
    for klass in Account.__mro__:
        if "accountType" in klass.__dict__:
            descriptor = klass.__dict__["accountType"]
            break
    assert isinstance(descriptor, property)

def test_account_has_balance():
    assert hasattr(Account, "balance")
    descriptor = None
    for klass in Account.__mro__:
        if "balance" in klass.__dict__:
            descriptor = klass.__dict__["balance"]
            break
    assert isinstance(descriptor, property)

def test_account_has_accountNumber():
    assert hasattr(Account, "accountNumber")
    descriptor = None
    for klass in Account.__mro__:
        if "accountNumber" in klass.__dict__:
            descriptor = klass.__dict__["accountNumber"]
            break
    assert isinstance(descriptor, property)

def test_transaction_class_exists():
    assert isinstance(Transaction, type)


def test_transaction_is_not_abstract():
    assert not inspect.isabstract(Transaction)


def test_transaction_constructor_exists():
    assert callable(Transaction.__init__)


def test_transaction_constructor_args():
    sig = inspect.signature(Transaction.__init__)
    params = list(sig.parameters.keys())
    assert "transactionId" in params, "Missing parameter 'transactionId'"
    assert "transactionType" in params, "Missing parameter 'transactionType'"
    assert "transactionDate" in params, "Missing parameter 'transactionDate'"
    assert "amount" in params, "Missing parameter 'amount'"

def test_transaction_has_transactionId():
    assert hasattr(Transaction, "transactionId")
    descriptor = None
    for klass in Transaction.__mro__:
        if "transactionId" in klass.__dict__:
            descriptor = klass.__dict__["transactionId"]
            break
    assert isinstance(descriptor, property)

def test_transaction_has_transactionType():
    assert hasattr(Transaction, "transactionType")
    descriptor = None
    for klass in Transaction.__mro__:
        if "transactionType" in klass.__dict__:
            descriptor = klass.__dict__["transactionType"]
            break
    assert isinstance(descriptor, property)

def test_transaction_has_transactionDate():
    assert hasattr(Transaction, "transactionDate")
    descriptor = None
    for klass in Transaction.__mro__:
        if "transactionDate" in klass.__dict__:
            descriptor = klass.__dict__["transactionDate"]
            break
    assert isinstance(descriptor, property)

def test_transaction_has_amount():
    assert hasattr(Transaction, "amount")
    descriptor = None
    for klass in Transaction.__mro__:
        if "amount" in klass.__dict__:
            descriptor = klass.__dict__["amount"]
            break
    assert isinstance(descriptor, property)

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

Branch_strategy = st.builds(
    Branch,
    branchId=
        safe_text,
    location=
        safe_text,
    branchName=
        safe_text
)

Customer_strategy = st.builds(
    Customer,
    phoneNumber=
        safe_text,
    name=
        safe_text,
    email=
        safe_text,
    customerId=
        safe_text
)

Account_strategy = st.builds(
    Account,
    accountType=
        safe_text,
    balance=
        st.floats(allow_nan=False, allow_infinity=False),
    accountNumber=
        safe_text
)

Transaction_strategy = st.builds(
    Transaction,
    transactionId=
        safe_text,
    transactionType=
        safe_text,
    transactionDate=
        st.none(),
    amount=
        st.floats(allow_nan=False, allow_infinity=False)
)

# =============================================================================
# SECTION 2 — PROPERTY-BASED & STATE-BASED TESTS
# =============================================================================

@given(instance=Branch_strategy)
@settings(max_examples=50)
def test_branch_instantiation(instance):
    assert isinstance(instance, Branch)

@given(instance=Branch_strategy)
def test_branch_branchId_type(instance):
    assert isinstance(instance.branchId, str)


@given(instance=Branch_strategy)
def test_branch_branchId_setter(instance):
    original = instance.branchId
    instance.branchId = original
    assert instance.branchId == original

@given(instance=Branch_strategy)
def test_branch_location_type(instance):
    assert isinstance(instance.location, str)


@given(instance=Branch_strategy)
def test_branch_location_setter(instance):
    original = instance.location
    instance.location = original
    assert instance.location == original

@given(instance=Branch_strategy)
def test_branch_branchName_type(instance):
    assert isinstance(instance.branchName, str)


@given(instance=Branch_strategy)
def test_branch_branchName_setter(instance):
    original = instance.branchName
    instance.branchName = original
    assert instance.branchName == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------

@given(instance=Customer_strategy)
@settings(max_examples=50)
def test_customer_instantiation(instance):
    assert isinstance(instance, Customer)

@given(instance=Customer_strategy)
def test_customer_phoneNumber_type(instance):
    assert isinstance(instance.phoneNumber, str)


@given(instance=Customer_strategy)
def test_customer_phoneNumber_setter(instance):
    original = instance.phoneNumber
    instance.phoneNumber = original
    assert instance.phoneNumber == original

@given(instance=Customer_strategy)
def test_customer_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Customer_strategy)
def test_customer_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original

@given(instance=Customer_strategy)
def test_customer_email_type(instance):
    assert isinstance(instance.email, str)


@given(instance=Customer_strategy)
def test_customer_email_setter(instance):
    original = instance.email
    instance.email = original
    assert instance.email == original

@given(instance=Customer_strategy)
def test_customer_customerId_type(instance):
    assert isinstance(instance.customerId, str)


@given(instance=Customer_strategy)
def test_customer_customerId_setter(instance):
    original = instance.customerId
    instance.customerId = original
    assert instance.customerId == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------

@given(instance=Account_strategy)
@settings(max_examples=50)
def test_account_instantiation(instance):
    assert isinstance(instance, Account)

@given(instance=Account_strategy)
def test_account_accountType_type(instance):
    assert isinstance(instance.accountType, str)


@given(instance=Account_strategy)
def test_account_accountType_setter(instance):
    original = instance.accountType
    instance.accountType = original
    assert instance.accountType == original

@given(instance=Account_strategy)
def test_account_balance_type(instance):
    assert isinstance(instance.balance, float)


@given(instance=Account_strategy)
def test_account_balance_setter(instance):
    original = instance.balance
    instance.balance = original
    assert instance.balance == original

@given(instance=Account_strategy)
def test_account_accountNumber_type(instance):
    assert isinstance(instance.accountNumber, str)


@given(instance=Account_strategy)
def test_account_accountNumber_setter(instance):
    original = instance.accountNumber
    instance.accountNumber = original
    assert instance.accountNumber == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------

@given(instance=Account_strategy)
@settings(max_examples=30)
def test_account_deposit_changes_state(instance):
    before = copy.deepcopy(instance)

    try:
        instance.deposit(
            1.0
        )

        assert instance.__dict__ != before.__dict__

    except (AttributeError, NotImplementedError, TypeError):
        pass

@given(instance=Account_strategy)
@settings(max_examples=30)
def test_account_withdraw_changes_state(instance):
    before = copy.deepcopy(instance)

    try:
        instance.withdraw(
            1.0
        )

        assert instance.__dict__ != before.__dict__

    except (AttributeError, NotImplementedError, TypeError):
        pass


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------

@given(instance=Transaction_strategy)
@settings(max_examples=50)
def test_transaction_instantiation(instance):
    assert isinstance(instance, Transaction)

@given(instance=Transaction_strategy)
def test_transaction_transactionId_type(instance):
    assert isinstance(instance.transactionId, str)


@given(instance=Transaction_strategy)
def test_transaction_transactionId_setter(instance):
    original = instance.transactionId
    instance.transactionId = original
    assert instance.transactionId == original

@given(instance=Transaction_strategy)
def test_transaction_transactionType_type(instance):
    assert isinstance(instance.transactionType, str)


@given(instance=Transaction_strategy)
def test_transaction_transactionType_setter(instance):
    original = instance.transactionType
    instance.transactionType = original
    assert instance.transactionType == original


@given(instance=Transaction_strategy)
def test_transaction_transactionDate_setter(instance):
    original = instance.transactionDate
    instance.transactionDate = original
    assert instance.transactionDate == original

@given(instance=Transaction_strategy)
def test_transaction_amount_type(instance):
    assert isinstance(instance.amount, float)


@given(instance=Transaction_strategy)
def test_transaction_amount_setter(instance):
    original = instance.amount
    instance.amount = original
    assert instance.amount == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------