JSON Object Generator
=====================

The JSON Object generator serialises an :doc:`object model
<../buml_language/model_types/object>` — the *instances* of your classes, not
the classes themselves — into a single JSON document. Use it to export a
concrete system state for fixtures, seed data, test payloads, or to hand a
worked example of your model to another tool.

It is the instance-level counterpart of the :doc:`JSON Schema generator
<json_schema>`: JSON Schema describes the shape your data must have, the JSON
Object generator emits data that has that shape.

Usage
-----

Create a ``JSONObjectGenerator``, pass the ``ObjectModel``, and call
``generate``:

.. code-block:: python

    from besser.BUML.metamodel.structural import (
        Class, DomainModel, Property, StringType, IntegerType,
        BinaryAssociation, Multiplicity
    )
    from besser.BUML.metamodel.object import (
        ObjectModel, Object, DataValue, AttributeLink, LinkEnd, Link
    )
    from besser.generators.json import JSONObjectGenerator

    # --- Structural model ------------------------------------------------
    name_prop = Property(name="name", type=StringType)
    age_prop = Property(name="age", type=IntegerType)
    author_cls = Class(name="Author", attributes={name_prop, age_prop})

    title_prop = Property(name="title", type=StringType)
    book_cls = Class(name="Book", attributes={title_prop})

    author_end = Property(name="author", type=author_cls, multiplicity=Multiplicity(1, 1))
    book_end = Property(name="books", type=book_cls, multiplicity=Multiplicity(0, "*"))
    writes = BinaryAssociation(name="writes", ends={author_end, book_end})

    domain_model = DomainModel(
        name="Library", types={author_cls, book_cls}, associations={writes}
    )

    # --- Object model (the instances) ------------------------------------
    alice = Object(name="alice", classifier=author_cls, slots=[
        AttributeLink(
            value=DataValue(classifier=StringType, value="Alice", name="v1"),
            attribute=name_prop,
        ),
        AttributeLink(
            value=DataValue(classifier=IntegerType, value=30, name="v2"),
            attribute=age_prop,
        ),
    ])
    book1 = Object(name="book1", classifier=book_cls, slots=[
        AttributeLink(
            value=DataValue(classifier=StringType, value="Python 101", name="v3"),
            attribute=title_prop,
        ),
    ])
    link = Link(name="writes_link", association=writes, connections=[
        LinkEnd(name="author", association_end=author_end, object=alice),
        LinkEnd(name="books", association_end=book_end, object=book1),
    ])

    object_model = ObjectModel(name="LibraryObjects", objects={alice, book1})

    # --- Generate ---------------------------------------------------------
    generator = JSONObjectGenerator(model=object_model)
    generator.generate()

Constructor parameters:

- ``model``: the ``ObjectModel`` to serialise. Passing anything else raises
  ``TypeError`` — this generator does not accept a ``DomainModel``.
- ``output_dir``: output directory (default: ``<<current_directory>>/output``).

The output file is named after the object model, with spaces replaced by
underscores: the example above writes ``LibraryObjects.json``. A model with no
name falls back to ``object_model.json``.

Output format
-------------

.. code-block:: json

    {
      "name": "LibraryObjects",
      "objects": [
        {
          "id": "alice",
          "class": "Author",
          "attributes": {
            "name": "Alice",
            "age": 30
          },
          "relationships": {
            "author": [
              "book1"
            ]
          }
        },
        {
          "id": "book1",
          "class": "Book",
          "attributes": {
            "title": "Python 101"
          },
          "relationships": {
            "books": [
              "alice"
            ]
          }
        }
      ]
    }

The document is a flat list rather than a nested tree, so a cycle in your
instance graph cannot produce infinite output:

- ``name`` — the object model's name. A ``description`` key is added when the
  model carries one in its metadata.
- ``objects`` — every object, sorted case-insensitively by name.
- ``id`` — the object's name, and the key other objects reference it by.
- ``class`` — the name of the object's classifier.
- ``attributes`` — one entry per attribute slot. Omitted when the object has
  none.
- ``relationships`` — links to other objects, by ``id``. Omitted when the
  object participates in no link.

.. note::
   Each entry under ``relationships`` is keyed by the name of the **object's
   own** association end (falling back to the association's name when the end
   is unnamed), not by the name of the end it points at. In the example above,
   ``alice`` is attached to the ``author`` end of the ``writes`` association,
   so her link to ``book1`` is listed under ``"author"``.

Value serialisation
-------------------

Attribute values are converted to JSON-native types:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - B-UML value
     - JSON
   * - ``date`` / ``datetime`` / ``time``
     - ISO-8601 string
   * - ``timedelta``
     - number of seconds (float)
   * - Enumeration literal
     - the literal's name, as a string
   * - ``set`` / ``list`` / ``tuple``
     - JSON array (sets are sorted for determinism)
   * - Any other object with a ``name_``
     - that name, as a string
   * - Everything else
     - passed through to ``json.dump`` unchanged

Output is written with ``indent=2`` and ``ensure_ascii=False``, so non-ASCII
text stays readable rather than being escaped.

Using it from the Web Modeling Editor
-------------------------------------

The generator is registered as ``jsonobject`` in the editor's generator
registry. It reads the project's object diagram rather than the class diagram,
and the editor downloads the result as ``object_model.json``.
