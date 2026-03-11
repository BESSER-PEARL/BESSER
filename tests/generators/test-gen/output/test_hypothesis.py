import inspect
import pytest
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import copy

from classes import (
    Author,
    Book,
    Genre,
)

# =============================================================================
# SECTION 1 — STRUCTURAL TESTS
# =============================================================================

def test_author_class_exists():
    assert isinstance(Author, type)


def test_author_is_not_abstract():
    assert not inspect.isabstract(Author)


def test_author_constructor_exists():
    assert callable(Author.__init__)


def test_author_constructor_args():
    sig = inspect.signature(Author.__init__)
    params = list(sig.parameters.keys())
    assert "email" in params, "Missing parameter 'email'"
    assert "name" in params, "Missing parameter 'name'"

def test_author_has_email():
    assert hasattr(Author, "email")
    descriptor = None
    for klass in Author.__mro__:
        if "email" in klass.__dict__:
            descriptor = klass.__dict__["email"]
            break
    assert isinstance(descriptor, property)

def test_author_has_name():
    assert hasattr(Author, "name")
    descriptor = None
    for klass in Author.__mro__:
        if "name" in klass.__dict__:
            descriptor = klass.__dict__["name"]
            break
    assert isinstance(descriptor, property)

def test_book_class_exists():
    assert isinstance(Book, type)


def test_book_is_not_abstract():
    assert not inspect.isabstract(Book)


def test_book_constructor_exists():
    assert callable(Book.__init__)


def test_book_constructor_args():
    sig = inspect.signature(Book.__init__)
    params = list(sig.parameters.keys())
    assert "rating" in params, "Missing parameter 'rating'"
    assert "inPrint" in params, "Missing parameter 'inPrint'"
    assert "title" in params, "Missing parameter 'title'"
    assert "pages" in params, "Missing parameter 'pages'"

def test_book_has_rating():
    assert hasattr(Book, "rating")
    descriptor = None
    for klass in Book.__mro__:
        if "rating" in klass.__dict__:
            descriptor = klass.__dict__["rating"]
            break
    assert isinstance(descriptor, property)

def test_book_has_inPrint():
    assert hasattr(Book, "inPrint")
    descriptor = None
    for klass in Book.__mro__:
        if "inPrint" in klass.__dict__:
            descriptor = klass.__dict__["inPrint"]
            break
    assert isinstance(descriptor, property)

def test_book_has_title():
    assert hasattr(Book, "title")
    descriptor = None
    for klass in Book.__mro__:
        if "title" in klass.__dict__:
            descriptor = klass.__dict__["title"]
            break
    assert isinstance(descriptor, property)

def test_book_has_pages():
    assert hasattr(Book, "pages")
    descriptor = None
    for klass in Book.__mro__:
        if "pages" in klass.__dict__:
            descriptor = klass.__dict__["pages"]
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

Author_strategy = st.builds(
    Author,
    email=
        safe_text,
    name=
        safe_text
)

Book_strategy = st.builds(
    Book,
    rating=
        st.floats(allow_nan=False, allow_infinity=False),
    inPrint=
        st.booleans(),
    title=
        safe_text,
    pages=
        st.integers()
)

# =============================================================================
# SECTION 2 — PROPERTY-BASED & STATE-BASED TESTS
# =============================================================================

@given(instance=Author_strategy)
@settings(max_examples=50)
def test_author_instantiation(instance):
    assert isinstance(instance, Author)

@given(instance=Author_strategy)
def test_author_email_type(instance):
    assert isinstance(instance.email, str)


@given(instance=Author_strategy)
def test_author_email_setter(instance):
    original = instance.email
    instance.email = original
    assert instance.email == original

@given(instance=Author_strategy)
def test_author_name_type(instance):
    assert isinstance(instance.name, str)


@given(instance=Author_strategy)
def test_author_name_setter(instance):
    original = instance.name
    instance.name = original
    assert instance.name == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------

@given(instance=Book_strategy)
@settings(max_examples=50)
def test_book_instantiation(instance):
    assert isinstance(instance, Book)

@given(instance=Book_strategy)
def test_book_rating_type(instance):
    assert isinstance(instance.rating, float)


@given(instance=Book_strategy)
def test_book_rating_setter(instance):
    original = instance.rating
    instance.rating = original
    assert instance.rating == original

@given(instance=Book_strategy)
def test_book_inPrint_type(instance):
    assert isinstance(instance.inPrint, bool)


@given(instance=Book_strategy)
def test_book_inPrint_setter(instance):
    original = instance.inPrint
    instance.inPrint = original
    assert instance.inPrint == original

@given(instance=Book_strategy)
def test_book_title_type(instance):
    assert isinstance(instance.title, str)


@given(instance=Book_strategy)
def test_book_title_setter(instance):
    original = instance.title
    instance.title = original
    assert instance.title == original

@given(instance=Book_strategy)
def test_book_pages_type(instance):
    assert isinstance(instance.pages, int)


@given(instance=Book_strategy)
def test_book_pages_setter(instance):
    original = instance.pages
    instance.pages = original
    assert instance.pages == original


# -------------------------------------------------------------------------
# ASSOCIATION TESTS
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# STATE-BASED OPERATION TESTS
# -------------------------------------------------------------------------

@given(instance=Book_strategy)
@settings(max_examples=30)
def test_book_getsummary_changes_state(instance):
    before = copy.deepcopy(instance)

    try:
        instance.getSummary(
            1
        )

        assert instance.__dict__ != before.__dict__

    except (AttributeError, NotImplementedError, TypeError):
        pass

@given(instance=Book_strategy)
@settings(max_examples=30)
def test_book_isavailable_changes_state(instance):
    before = copy.deepcopy(instance)

    try:
        instance.isAvailable()

        assert instance.__dict__ != before.__dict__

    except (AttributeError, NotImplementedError, TypeError):
        pass


# -------------------------------------------------------------------------
# OCL CONSTRAINT TESTS
# -------------------------------------------------------------------------