"""A table's create/edit form offers only what the backend Create schema accepts.

Measured on recorded hotel runs: Booking's form carried an editable ``bill``
lookup although Booking is the non-owning side of the 1:1 Booking-Bill
association (Bill holds the ``booking`` foreign key), so ``BookingCreate``
has no ``bill`` field; the value was sent and silently dropped (48 of 52
recorded ``bill`` findings). Form columns now use the same rule as the
pydantic generator, and a drift guard checks both agree on every
multiplicity shape.
"""
import ast
import json
import os

from besser.BUML.metamodel.gui import DataBinding, GUIModel, Module, Screen
from besser.BUML.metamodel.gui.dashboard import Table
from besser.BUML.metamodel.structural import (
    AssociationClass,
    BinaryAssociation,
    Class,
    DomainModel,
    FloatType,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.react import ReactGenerator
from besser.generators.structural_utils import create_schema_accepts_end, get_foreign_keys


def _end(name, cls, low, high, navigable=True):
    return Property(name=name, type=cls, multiplicity=Multiplicity(low, high), is_navigable=navigable)


def _hotel():
    booking = Class(name="Booking", attributes={Property(name="code", type=StringType)})
    bill = Class(name="Bill", attributes={Property(name="total", type=FloatType)})
    guest = Class(name="Guest", attributes={Property(name="name", type=StringType)})
    # Bill holds the FK: its booking end is mandatory, Booking's bill end optional.
    billing = BinaryAssociation(name="billing", ends={_end("bill", bill, 0, 1), _end("booking", booking, 1, 1)})
    contact = BinaryAssociation(name="contact", ends={_end("contact", guest, 1, 1), _end("bookings", booking, 0, "*")})
    return DomainModel(name="Hotel", types={booking, bill, guest}, associations={billing, contact})


def _form_columns(domain, class_name, tmp_path):
    cls = domain.get_class_by_name(class_name)
    table = Table(name=f"{class_name}_table", data_binding=DataBinding(domain_concept=cls), component_id="t")
    screen = Screen(name=class_name, description=class_name, view_elements={table}, is_main_page=True)
    gui = GUIModel(name="G", package="", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})
    ReactGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path / class_name)).generate()
    with open(os.path.join(str(tmp_path / class_name), "src", "pages", f"{class_name}.tsx"), encoding="utf-8") as f:
        page = f.read()
    options = page[page.index("options={") + len("options={"):]
    return {column.get("path") or column["field"]: column
            for column in json.JSONDecoder().raw_decode(options)[0]["formColumns"]}


def test_the_non_owning_side_of_a_one_to_one_has_no_form_field(tmp_path):
    domain = _hotel()

    booking = _form_columns(domain, "Booking", tmp_path)
    assert "bill" not in booking  # neither editable nor sent
    assert booking["contact"]["column_type"] == "lookup"  # its FK-owning N:1 end stays

    bill = _form_columns(domain, "Bill", tmp_path)
    assert bill["booking"]["column_type"] == "lookup"  # the FK owner keeps it, editable
    assert not bill["booking"].get("readOnly")


def _create_fields(domain, tmp_path):
    PydanticGenerator(domain, backend=True, output_dir=str(tmp_path)).generate()
    with open(os.path.join(str(tmp_path), "pydantic_classes.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def fields(name):
        node = classes[name]
        own = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                own |= fields(base.id)
        return own

    return {name[:-len("Create")]: fields(name) for name in classes if name.endswith("Create")}


def test_form_columns_follow_the_backend_create_schema_for_every_shape(tmp_path):
    """Drift guard: every association shape, both sides."""
    a, b, c, d, e, f = (Class(name=n) for n in ("A", "B", "C", "D", "E", "F"))
    node = Class(name="Node")
    child = Class(name="Child")
    for cls in (a, b, c, d, e, f, node, child):
        cls.attributes = {Property(name=f"{cls.name.lower()}_label", type=StringType)}
    link = AssociationClass(
        name="Link",
        attributes={Property(name="weight", type=IntegerType)},
        association=BinaryAssociation(name="links", ends={_end("es", e, 0, "*"), _end("fs", f, 0, "*")}),
    )
    associations = {
        BinaryAssociation(name="nm", ends={_end("bs", b, 0, "*"), _end("as_", a, 1, "*")}),
        BinaryAssociation(name="n1", ends={_end("owner_c", c, 1, 1), _end("bs_of_c", b, 0, "*")}),
        BinaryAssociation(name="one_one_req", ends={_end("d_req", d, 1, 1), _end("a_opt", a, 0, 1)}),
        BinaryAssociation(name="one_one_opt", ends={_end("c_one", c, 0, 1), _end("d_one", d, 0, 1)}),
        BinaryAssociation(name="one_way", ends={_end("hidden_a", a, 0, 1, navigable=False), _end("seen_e", e, 0, "*")}),
        BinaryAssociation(name="tree", ends={_end("parent", node, 0, 1), _end("children", node, 0, "*")}),
        link.association,
    }
    domain = DomainModel(
        name="Shapes",
        types={a, b, c, d, e, f, node, child, link},
        associations=associations,
        generalizations={Generalization(general=node, specific=child)},
    )

    create = _create_fields(domain, tmp_path)
    fkeys = get_foreign_keys(domain)
    for cls in (a, b, c, d, e, f, node, child):
        predicted = {
            end.name for end in cls.all_association_ends()
            if create_schema_accepts_end(end, fkeys, {"links"})
        }
        attributes = {attr.name for attr in cls.all_attributes()}
        assert predicted == create[cls.name] - attributes, cls.name


def test_an_id_is_a_form_field_only_when_it_is_the_declared_primary_key(tmp_path):
    """A plain ``id`` is the server's surrogate key (not in ``<X>Create``); a
    declared ``is_id`` key is client-supplied and stays editable."""
    surrogate = Class(name="Ticket", attributes={
        Property(name="id", type=IntegerType), Property(name="title", type=StringType)})
    declared = Class(name="Room", attributes={
        Property(name="id", type=IntegerType, is_id=True), Property(name="floor", type=IntegerType)})
    domain = DomainModel(name="Desk", types={surrogate, declared})

    create = _create_fields(domain, tmp_path / "pydantic")
    assert "id" not in create["Ticket"] and "id" in create["Room"]
    assert "id" not in _form_columns(domain, "Ticket", tmp_path)
    assert "id" in _form_columns(domain, "Room", tmp_path)
