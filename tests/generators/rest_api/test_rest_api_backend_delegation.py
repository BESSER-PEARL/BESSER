"""Contract of RESTAPIGenerator(backend=True) after the modular split.

The backend mode renders the same modular API layer as BackendGenerator
(shared ``generate_modular_api``), but it must keep its historical scope: the
API files plus requirements.txt, and nothing else. ``sql_alchemy.py`` and
``pydantic_classes.py`` belong to the caller (BackendGenerator orchestrates
them), so ``backend=True`` must never overwrite them. The ``port`` argument
must land in the generated ``uvicorn.run`` call.
"""

import os

from besser.generators.rest_api import RESTAPIGenerator
from besser.generators.backend import BackendGenerator


def test_backend_mode_generates_only_the_api_layer(library_book_author_model, tmp_path):
    output_dir = str(tmp_path / "output")
    RESTAPIGenerator(
        model=library_book_author_model, backend=True, output_dir=output_dir
    ).generate()

    for expected in ("main_api.py", "database.py", "bal_stdlib.py", "requirements.txt"):
        assert os.path.isfile(os.path.join(output_dir, expected)), f"Missing {expected}"
    for class_name in ("library", "book", "author"):
        assert os.path.isfile(os.path.join(output_dir, "routers", f"{class_name}.py"))

    # backend=True never owned sql_alchemy.py / pydantic_classes.py — callers
    # (like BackendGenerator) generate them, and delegating must not start
    # overwriting files in a directory the caller manages.
    assert not os.path.isfile(os.path.join(output_dir, "sql_alchemy.py"))
    assert not os.path.isfile(os.path.join(output_dir, "pydantic_classes.py"))


def test_backend_mode_honors_the_port_argument(library_book_author_model, tmp_path):
    output_dir = str(tmp_path / "output")
    RESTAPIGenerator(
        model=library_book_author_model, backend=True, port=9001, output_dir=output_dir
    ).generate()
    with open(os.path.join(output_dir, "main_api.py"), encoding="utf-8") as f:
        assert "port=9001" in f.read()


def test_backend_generator_honors_the_port_argument(library_book_author_model, tmp_path):
    output_dir = str(tmp_path / "output")
    BackendGenerator(
        model=library_book_author_model, port=9002, output_dir=output_dir
    ).generate()
    with open(os.path.join(output_dir, "main_api.py"), encoding="utf-8") as f:
        assert "port=9002" in f.read()


def test_optional_link_attribute_is_optional_in_the_payload(tmp_path):
    """The form that submits an association-class link leaves an optional
    attribute blank, so demanding it in the payload rejects a request the UI
    considers complete."""
    from besser.BUML.metamodel.structural import (
        AssociationClass, BinaryAssociation, Class, DomainModel, FloatType,
        IntegerType, Multiplicity, Property,
    )
    from besser.generators.pydantic_classes import PydanticGenerator

    trip = Class(name="Trip", attributes={Property(name="id", type=IntegerType, is_id=True)})
    seat = Class(name="Seat", attributes={Property(name="code", type=IntegerType, is_id=True)})
    trip_seat = BinaryAssociation(name="trip_seat", ends={
        Property(name="trips", type=trip, multiplicity=Multiplicity(0, "*")),
        Property(name="seats", type=seat, multiplicity=Multiplicity(0, "*")),
    })
    reservation = AssociationClass(
        name="Reservation",
        attributes={
            Property(name="price", type=FloatType),
            Property(name="surcharge", type=FloatType, is_optional=True),
        },
        association=trip_seat,
    )
    model = DomainModel(name="TripModel", types={trip, seat, reservation}, associations={trip_seat})

    PydanticGenerator(model=model, output_dir=str(tmp_path), backend=True).generate()
    code = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")

    assert "class ReservationLinkCreate(BaseModel):" in code
    assert "    surcharge: Optional[float] = None" in code
    assert "    price: float\n" in code
