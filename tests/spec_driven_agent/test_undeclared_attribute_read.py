"""An attribute read off a model instance that nothing declares.

The largest runtime-crash class in the recorded corpus. A census of every
crash the runtime probe observed across ``verification/spec-iterations``
(390 workspaces) found 52 observed failures; the biggest single cause is a
member name the receiver's class does not have -- 15 occurrences over 14
runs, of which exactly one was already reported by a validator:

    'Clerk' object has no attribute 'warehouse_id'        4 runs
    'Loan' object has no attribute 'returnDate'           3 runs
    'Booking' object has no attribute 'extraCharges'      2 runs (also as
        TypeError: 'extraCharges' is an invalid keyword argument for Booking)
    'Product' object has no attribute 'quantity'
    'Room' object has no attribute 'bookings'
    'Warehouse' object has no attribute 'total_stock'
    'Loan' object has no attribute 'created_at' / 'updated_at'

Three causes, one shape: a column that belongs to a DIFFERENT class
(``warehouse_id`` is Product's), a field the scaffold never wrote
(``created_at``), and a value the model invented (``total_stock``). Every
one is an ``AttributeError`` the first time the line runs.

Calibration is the point of the "silent" tests below. Each is a shape that
appears on probed-WORKING corpus apps, so they pin false positives the
measured version of this check does not have:
``verification/corpus_gate.py --validator contract_checks.undeclared_attribute``
reports 0 findings on 93 working apps (recorded labels) and 0 on 139
(rescored), and 17 on 9 dead ones. That silence is not vacuous: over the
same sweep it typed 25,288 attribute reads to a domain class and tested
16,916 names against the declarations, plus 10,422 constructor keywords.
16,887 and 10,421 of those resolved.
"""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Multiplicity,
    PrimitiveDataType,
    Property,
)
from besser.spec_driven_agent.contract_checks import (
    build_data_contract,
    collect_undeclared_attribute_issues,
)
from besser.spec_driven_agent.validation.issues import _classify_issue

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")

ONE = Multiplicity(1, 1)
MANY = Multiplicity(0, 9999)


def _inventory_model() -> DomainModel:
    """Clerk/Customer under Person, Product in a Warehouse, plus an Order.

    Reduced from the live inventory case. ``warehouse`` is named on the
    Product end, so it is a property of Product and no other class.
    """
    person = Class(name="Person")
    person.attributes = {Property(name="firstName", type=StringType)}
    clerk = Class(name="Clerk")
    product = Class(name="Product")
    product.attributes = {Property(name="quantityInStock", type=IntegerType)}
    warehouse = Class(name="Warehouse")
    warehouse.attributes = {Property(name="capacity", type=IntegerType)}
    order = Class(name="Order")
    order.attributes = {Property(name="orderNumber", type=StringType)}

    stored = BinaryAssociation(name="stored", ends={
        Property(name="warehouse", type=warehouse, multiplicity=ONE),
        Property(name="products", type=product, multiplicity=MANY),
    })
    return DomainModel(
        name="Inventory",
        types={person, clerk, product, warehouse, order},
        associations={stored},
        generalizations={Generalization(general=person, specific=clerk)},
    )


SCAFFOLD = '''\
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Person(Base):
    __tablename__ = "person"
    id: Mapped[int] = mapped_column(primary_key=True)
    firstName: Mapped[str]

class Clerk(Person):
    __tablename__ = "clerk"
    id: Mapped[int] = mapped_column(ForeignKey("person.id"), primary_key=True)

class Warehouse(Base):
    __tablename__ = "warehouse"
    id: Mapped[int] = mapped_column(primary_key=True)
    capacity: Mapped[int]

class Product(Base):
    __tablename__ = "product"
    id: Mapped[int] = mapped_column(primary_key=True)
    quantityInStock: Mapped[int]
    warehouse_id: Mapped[int] = mapped_column(ForeignKey("warehouse.id"))

class Order(Base):
    __tablename__ = "order"
    id: Mapped[int] = mapped_column(primary_key=True)
    orderNumber: Mapped[str]

Product.warehouse: Mapped["Warehouse"] = relationship(
    "Warehouse", back_populates="products")
'''


def _app(tmp_path, **files) -> str:
    backend = tmp_path / "backend"
    backend.mkdir(parents=True, exist_ok=True)
    (backend / "sql_alchemy.py").write_text(SCAFFOLD, encoding="utf-8")
    for name, source in files.items():
        (backend / f"{name}.py").write_text(source, encoding="utf-8")
    return str(tmp_path)


def _issues(tmp_path, model=None, **files) -> list:
    contract = build_data_contract(
        model if model is not None else _inventory_model())
    return collect_undeclared_attribute_issues(_app(tmp_path, **files), contract)


# --------------------------------------------------------------------------- #
# The live defect
# --------------------------------------------------------------------------- #

def test_reads_a_column_that_belongs_to_another_class(tmp_path):
    """``...2507-2fqm5uj6``: POST /order/ answered
    ``EXC: AttributeError: 'Clerk' object has no attribute 'warehouse_id'``.
    ``warehouse_id`` is Product's foreign key; Clerk inherits only Person's
    columns. Four corpus runs died this way."""
    issues = _issues(tmp_path, order='''\
from sql_alchemy import Clerk

def create_order(order_data, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == order_data.handledBy).first()
    if db_clerk.warehouse_id is None:
        raise HTTPException(status_code=400, detail="no warehouse")
''')
    assert len(issues) == 1, issues
    [finding] = issues
    assert "backend/order.py line 5" in finding
    assert "Clerk.warehouse_id" in finding
    assert _classify_issue(finding).severity == "blocker"


def test_reads_an_audit_field_the_scaffold_never_wrote(tmp_path):
    """``...2507-j38du4bm``: a response body built from ``db_order.created_at``
    and ``db_order.updated_at``, neither of which the ORM class declares."""
    issues = _issues(tmp_path, order='''\
from sql_alchemy import Order

def read_order(order_id, database):
    db_order = database.query(Order).filter(Order.id == order_id).first()
    return {"createdAt": db_order.created_at, "updatedAt": db_order.updated_at}
''')
    assert len(issues) == 2, issues
    assert all("line 5" in finding for finding in issues)
    assert {"created_at", "updated_at"} == {
        finding.split("Order.")[1].split("`")[0] for finding in issues}


def test_reads_a_value_the_model_invented(tmp_path):
    """``...2507-iia3pn0_``: ``db_warehouse.total_stock`` on the right-hand
    side of its own assignment. The write would bind the name; the read
    that feeds it raises first."""
    issues = _issues(tmp_path, order='''\
from sql_alchemy import Warehouse

def create_order(order_data, database):
    db_warehouse = database.query(Warehouse).filter(Warehouse.id == 1).first()
    db_warehouse.total_stock = db_warehouse.total_stock + order_data.quantity
''')
    assert len(issues) == 1, issues
    assert "Warehouse.total_stock" in issues[0]


def test_the_constructor_keyword_twin(tmp_path):
    """``...2507-_hnqmog8``: POST /order/ answered ``EXC: TypeError:
    'warehouse_id' is an invalid keyword argument for Order``. SQLAlchemy
    raises before any attribute is read, so one wrong name has two
    exception types and a single cause."""
    issues = _issues(tmp_path, order='''\
from sql_alchemy import Order

def create_order(order_data, database):
    db_order = Order(orderNumber=order_data.orderNumber,
                     warehouse_id=order_data.warehouse_id)
    return db_order.orderNumber
''')
    assert len(issues) == 1, issues
    assert "Order.warehouse_id" in issues[0]
    assert "invalid keyword argument" in issues[0]
    assert _classify_issue(issues[0]).severity == "blocker"


def test_silent_on_a_constructor_keyword_the_class_declares(tmp_path):
    """Every generated create handler builds a row this way; a keyword the
    class has must never be claimed."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Order

def create_order(order_data, database):
    return Order(orderNumber=order_data.orderNumber)
''') == []


def test_silent_on_a_star_kwargs_construction(tmp_path):
    """``Order(**payload)`` names nothing, so nothing is provable."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Order

def create_order(order_data, database):
    return Order(**order_data.model_dump())
''') == []


# --------------------------------------------------------------------------- #
# Calibration: shapes that must stay silent
# --------------------------------------------------------------------------- #

def test_silent_on_a_column_the_class_does_declare(tmp_path):
    """``db_product.warehouse_id`` is the same name on the class that owns
    it. The check is about the receiver, never the name."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Product

def create_order(order_data, database):
    db_product = database.query(Product).filter(Product.id == 1).first()
    return db_product.warehouse_id
''') == []


def test_silent_on_a_column_inherited_from_a_base(tmp_path):
    """Clerk declares nothing but its own id; ``firstName`` is Person's."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Clerk

def create_order(order_data, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == 1).first()
    return db_clerk.firstName
''') == []


def test_silent_on_an_association_role(tmp_path):
    """``product.warehouse`` is a role. Roles are the inverted-end check's,
    and reporting them here would double every one of its findings."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Product

def create_order(order_data, database):
    db_product = database.query(Product).filter(Product.id == 1).first()
    return db_product.warehouse.capacity
''') == []


def test_silent_on_a_hasattr_guarded_read(tmp_path):
    """``...2507-d71pocck`` shipped a guarded read and served 11/11 workflow
    checks: the author already handles the field being absent."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Clerk

def create_order(order_data, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == 1).first()
    return hasattr(db_clerk, "warehouse_id") and db_clerk.warehouse_id
''') == []


def test_silent_on_a_read_only_a_branch_reaches(tmp_path):
    """``...2507-n19svhnc`` passes its probe with a real ``Loan.returnDate``
    crash behind ``if status == RETURNED``. The read is genuinely wrong and
    a blocker there would spend fix turns on an app that works."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Clerk

def create_order(order_data, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == 1).first()
    if order_data.strict:
        return db_clerk.warehouse_id
''') == []


def test_an_if_test_is_not_a_branch(tmp_path):
    """The condition itself evaluates whenever control reaches the
    statement. An earlier version treated a whole ``if`` as conditional and
    demoted the exact read the probe watched crash on ...2507-2fqm5uj6."""
    issues = _issues(tmp_path, order='''\
from sql_alchemy import Clerk

def create_order(order_data, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == 1).first()
    if db_clerk.warehouse_id is None:
        raise HTTPException(status_code=400, detail="no warehouse")
''')
    assert len(issues) == 1, issues


def test_silent_on_a_write(tmp_path):
    """Binding a new attribute on an instance never raises. It may not
    persist, but that is a different finding with a different remedy."""
    assert _issues(tmp_path, order='''\
from sql_alchemy import Order

def create_order(order_data, database):
    db_order = Order(orderNumber=order_data.orderNumber)
    db_order.total_stock = 4
''') == []


def test_silent_on_an_injected_dependency_that_shares_a_class_name(tmp_path):
    """``database: Session = Depends(get_db)`` in an app whose model has a
    class called ``Session``. Five corpus apps produced 44-145 findings
    each this way -- every ``database.query`` read as a member of the
    domain class -- before Depends-annotated parameters were excluded."""
    session = Class(name="Session")
    session.attributes = {Property(name="title", type=StringType)}
    speaker = Class(name="Speaker")
    presents = BinaryAssociation(name="presents", ends={
        Property(name="talks", type=session, multiplicity=MANY),
        Property(name="speaker", type=speaker, multiplicity=ONE),
    })
    model = DomainModel(name="Conf", types={session, speaker},
                        associations={presents})
    assert _issues(tmp_path, model=model, speaker='''\
from sqlalchemy.orm import Session

def create_speaker(speaker_data, database: Session = Depends(get_db)):
    database.add(speaker_data)
    database.flush()
    database.commit()
''') == []


def test_silent_when_the_model_declares_no_association(tmp_path):
    """No ends, no class membership to test against, so nothing is claimed."""
    lonely = Class(name="Note")
    lonely.attributes = {Property(name="text", type=StringType)}
    model = DomainModel(name="Notes", types={lonely})
    assert _issues(tmp_path, model=model, note='''\
from sql_alchemy import Note

def read_note(note_id, database):
    db_note = database.query(Note).filter(Note.id == note_id).first()
    return db_note.anything_at_all
''') == []


@pytest.mark.parametrize("body", [
    "return db_clerk.model_dump()",
    "return db_clerk.metadata",
    "return db_clerk._sa_instance_state",
])
def test_silent_on_members_a_base_class_contributes(tmp_path, body):
    """A domain class name is reused for the ORM model and the Pydantic
    schema, and both bases contribute a public API the app never writes."""
    assert _issues(tmp_path, order=f'''\
from sql_alchemy import Clerk

def read_clerk(clerk_id, database):
    db_clerk = database.query(Clerk).filter(Clerk.id == clerk_id).first()
    {body}
''') == []
