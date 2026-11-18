"""A required ``str`` attribute rejects an empty or blank value in the backend Create schemas.

Imports the generated module and validates payloads through it, the way the
generated FastAPI routes do. Optional strings keep accepting any value.
"""

import importlib.util
import os
import sys

import pytest
from pydantic import ValidationError

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, Generalization, IntegerType,
    Multiplicity, Property, StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator


def _catalog_model() -> DomainModel:
    item = Class(name="Item")
    item.attributes = {
        Property(name="name", type=StringType),
        Property(name="note", type=StringType, is_optional=True),
        Property(name="quantity", type=IntegerType),
    }
    product = Class(name="Product")
    product.attributes = {Property(name="sku", type=StringType)}
    return DomainModel(
        name="Catalog", types={item, product},
        generalizations={Generalization(general=item, specific=product)},
    )


def _rental_model() -> DomainModel:
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    tool = Class(name="Tool", attributes={Property(name="label", type=StringType)})
    rents = BinaryAssociation(name="Rents", ends={
        Property(name="renters", type=customer, multiplicity=Multiplicity(0, "*")),
        Property(name="rented", type=tool, multiplicity=Multiplicity(0, "*")),
    })
    rental = AssociationClass(name="Rental", association=rents, attributes={
        Property(name="purpose", type=StringType),
        Property(name="memo", type=StringType, is_optional=True),
    })
    return DomainModel(name="Rentals", types={customer, tool, rental}, associations={rents})


def _import_generated(tmpdir, module_name: str, model=None):
    output_dir = tmpdir.mkdir(module_name)
    PydanticGenerator(model=model or _catalog_model(), output_dir=str(output_dir), backend=True).generate()
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(str(output_dir), "pydantic_classes.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_required_string_rejects_blank(tmpdir, blank):
    module = _import_generated(tmpdir, "blank_required_module")
    with pytest.raises(ValidationError, match="name"):
        module.ItemCreate(name=blank, quantity=1)


def test_required_string_keeps_a_real_value_unchanged(tmpdir):
    module = _import_generated(tmpdir, "blank_value_module")
    assert module.ItemCreate(name="  Lamp ", quantity=1).name == "  Lamp "


def test_optional_string_still_accepts_blank_and_none(tmpdir):
    module = _import_generated(tmpdir, "blank_optional_module")
    assert module.ItemCreate(name="Lamp", quantity=1, note="").note == ""
    assert module.ItemCreate(name="Lamp", quantity=1).note is None


def test_subclass_checks_its_own_and_inherited_required_strings(tmpdir):
    module = _import_generated(tmpdir, "blank_inherited_module")
    with pytest.raises(ValidationError, match="sku"):
        module.ProductCreate(name="Lamp", quantity=1, sku=" ")
    with pytest.raises(ValidationError, match="name"):
        module.ProductCreate(name=" ", quantity=1, sku="L-1")
    assert module.ProductCreate(name="Lamp", quantity=1, sku="L-1").sku == "L-1"


def test_association_class_link_payload_rejects_blank_required_string(tmpdir):
    """A link created from an end (``rented: [{target, purpose}]``) is validated like the record itself."""
    module = _import_generated(tmpdir, "blank_link_module", _rental_model())
    for blank in ("", "  "):
        with pytest.raises(ValidationError, match="purpose"):
            module.RentalLinkCreate(target=1, purpose=blank)
    link = module.RentalLinkCreate(target=1, purpose="repair", memo="")
    assert (link.purpose, link.memo) == ("repair", "")
