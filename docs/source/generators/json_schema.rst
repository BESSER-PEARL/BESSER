JSON Schema Generator
=====================

The JSON Schema generator produces JSON Schema definitions based on a given B-UML model. This schema can then be used to validate JSON objects against the model's structure and constraints. The generator supports two modes: **regular** mode for standard JSON schemas and **smart data** mode for Smart Data Models compliant schemas.

Regular JSON Schema Generation
------------------------------
This example demonstrates how to generate a JSON Schema for a Network Data Schema model. The model describes
typical entities found in mobile networks, such as network cells, user equipment (UE), sessions, and configuration
parameters, along with their relationships and attributes.

To generate the JSON Schema, create a ``JSONSchemaGenerator`` object, provide the :doc:`../buml_language/model_types/structural`,
(``network_model`` in the example) and call the ``generate`` method as shown:

.. code-block:: python
    
    from besser.BUML.metamodel.structural import (
        Class, Property, Method, Parameter,
        BinaryAssociation, Generalization, DomainModel,
        Enumeration, EnumerationLiteral, Multiplicity,
        StringType, IntegerType, FloatType, BooleanType,
        TimeType, DateType, DateTimeType, TimeDeltaType,
        AnyType, Constraint, AssociationClass
    )

    # Enumerations
    communicationType: Enumeration = Enumeration(
        name="communicationType",
        literals={
                EnumerationLiteral(name="TDD"),
                EnumerationLiteral(name="FDD")
        }
    )

    # Classes
    NRCellCU = Class(name="NRCellCU")
    NRCellDU = Class(name="NRCellDU")
    BWP = Class(name="BWP")
    UE = Class(name="UE")
    PDUSession = Class(name="PDUSession")
    SynchronizationSignal = Class(name="SynchronizationSignal")
    TDD = Class(name="TDD")

    # NRCellCU class attributes and methods
    NRCellCU_nRcellId: Property = Property(name="nRcellId", type=IntegerType)
    NRCellCU_physicalCellId: Property = Property(name="physicalCellId", type=IntegerType)
    NRCellCU.attributes={NRCellCU_physicalCellId, NRCellCU_nRcellId}

    # NRCellDU class attributes and methods
    NRCellDU_nRCellid: Property = Property(name="nRCellid", type=IntegerType)
    NRCellDU_physicalCellId: Property = Property(name="physicalCellId", type=IntegerType)
    NRCellDU_policy: Property = Property(name="policy", type=communicationType)
    NRCellDU_ssb: Property = Property(name="ssb", type=IntegerType)
    NRCellDU.attributes={NRCellDU_nRCellid, NRCellDU_policy, NRCellDU_physicalCellId, NRCellDU_ssb}

    # BWP class attributes and methods
    BWP_subCarrierSpacing: Property = Property(name="subCarrierSpacing", type=IntegerType)
    BWP_numberofRBs: Property = Property(name="numberofRBs", type=IntegerType)
    BWP.attributes={BWP_numberofRBs, BWP_subCarrierSpacing}

    # UE class attributes and methods
    UE_id: Property = Property(name="id", type=IntegerType)
    UE_rnti: Property = Property(name="rnti", type=StringType)
    UE_position: Property = Property(name="position", type=IntegerType)
    UE.attributes={UE_id, UE_position, UE_rnti}

    # PDUSession class attributes and methods
    PDUSession_id: Property = Property(name="id", type=IntegerType)
    PDUSession.attributes={PDUSession_id}

    # SynchronizationSignal class attributes and methods
    SynchronizationSignal_rsrq: Property = Property(name="rsrq", type=FloatType)
    SynchronizationSignal_sinr: Property = Property(name="sinr", type=FloatType)
    SynchronizationSignal_rsrp: Property = Property(name="rsrp", type=IntegerType)
    SynchronizationSignal.attributes={SynchronizationSignal_rsrq, SynchronizationSignal_rsrp, SynchronizationSignal_sinr}

    # TDD class attributes and methods
    TDD_band: Property = Property(name="band", type=IntegerType)
    TDD_arfcn: Property = Property(name="arfcn", type=IntegerType)
    TDD.attributes={TDD_arfcn, TDD_band}

    # Relationships
    NRCellDU_BWP: BinaryAssociation = BinaryAssociation(
        name="NRCellDU_BWP",
        ends={
            Property(name="nrcelldu", type=NRCellDU, multiplicity=Multiplicity(1, 1)),
            Property(name="bwp", type=BWP, multiplicity=Multiplicity(1, 1))
        }
    )
    NRCellDU_UE: BinaryAssociation = BinaryAssociation(
        name="NRCellDU_UE",
        ends={
            Property(name="du", type=NRCellDU, multiplicity=Multiplicity(1, 1)),
            Property(name="userEquipments", type=UE, multiplicity=Multiplicity(0, 9999))
        }
    )
    UE_PDUSession: BinaryAssociation = BinaryAssociation(
        name="UE_PDUSession",
        ends={
            Property(name="ue", type=UE, multiplicity=Multiplicity(1, 1)),
            Property(name="pdus", type=PDUSession, multiplicity=Multiplicity(0, 9999))
        }
    )
    UE_SynchronizationSignal: BinaryAssociation = BinaryAssociation(
        name="UE_SynchronizationSignal",
        ends={
            Property(name="ue", type=UE, multiplicity=Multiplicity(1, 1)),
            Property(name="synchronizationSignal", type=SynchronizationSignal, multiplicity=Multiplicity(1, 9999))
        }
    )
    NRCellCU_NRCellDU: BinaryAssociation = BinaryAssociation(
        name="NRCellCU_NRCellDU",
        ends={
            Property(name="nrcellcu", type=NRCellCU, multiplicity=Multiplicity(1, 1)),
            Property(name="nrCellDU_by_duId", type=NRCellDU, multiplicity=Multiplicity(1, 9999))
        }
    )
    TDD_NRCellDU: BinaryAssociation = BinaryAssociation(
        name="TDD_NRCellDU",
        ends={
            Property(name="tdd", type=TDD, multiplicity=Multiplicity(1, 1)),
            Property(name="nrcelldu", type=NRCellDU, multiplicity=Multiplicity(1, 1))
        }
    )

    # Domain Model
    network_model = DomainModel(
        name="Network-DataSchema",
        types={NRCellCU, NRCellDU, BWP, UE, PDUSession, SynchronizationSignal, TDD, communicationType},
        associations={NRCellDU_BWP, NRCellDU_UE, UE_PDUSession, UE_SynchronizationSignal, NRCellCU_NRCellDU, TDD_NRCellDU},
        generalizations={}
    )

    from besser.generators.json import JSONSchemaGenerator

    generator: JSONSchemaGenerator = JSONSchemaGenerator(model=network_model, mode='regular')
    generator.generate()

The ``json_schema.json`` file containing the JSON Schema will be generated in the ``<<current_directory>>/output``
folder. This schema will include definitions for all the classes (NRCellCU, NRCellDU, BWP, UE, PDUSession, SynchronizationSignal, TDD) and enumerations (communicationType) defined in your model.

This schema can now be used by any JSON Schema validator to ensure that the JSON objects conform to the model's structure and constraints.

Smart Data Models Generation
-----------------------------

The generator also supports **Smart Data Models** mode, which creates schemas compliant with the `Smart Data Models initiative <https://smartdatamodels.org/>`_. This format is particularly useful for IoT, smart cities, and data interoperability scenarios.

To generate Smart Data Models compatible schemas:

.. code-block:: python
    
    from besser.generators.json import JSONSchemaGenerator

    # Output directory for generated schemas
    output_dir = "output_directory"

    # Create the generator in Smart Data mode
    generator = JSONSchemaGenerator(network_model, output_dir, mode='smart_data')

    # Generate the schemas
    generator.generate()

In Smart Data Models mode, the generator will:

- Create a separate schema file for each class in your model
- Generate schemas that reference the `GSMA-Commons <https://smart-data-models.github.io/data-models/common-schema.json#/definitions/GSMA-Commons>`_ and `Location-Commons <https://smart-data-models.github.io/data-models/common-schema.json#/definitions/Location-Commons>`_ definitions
- Include standard fields like ``id``, ``type``, ``dateCreated``, ``dateModified``, ``location``, and ``address``
- Create example JSON files for each schema
- Generate the required ``ADOPTERS.yaml`` and ``notes.yaml`` files

The output structure for Smart Data Models will be:

.. code-block::

    output/
    ├── ADOPTERS.yaml
    ├── notes.yaml
    ├── NRCellCU/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    ├── NRCellDU/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    ├── BWP/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    ├── UE/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    ├── PDUSession/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    ├── SynchronizationSignal/
    │   ├── schema.json
    │   └── examples/
    │       ├── example.json
    │       ├── example-normalized.json
    │       ├── example.jsonld
    │       └── example-normalized.jsonld
    └── TDD/
        ├── schema.json
        └── examples/
            ├── example.json
            ├── example-normalized.json
            ├── example.jsonld
            └── example-normalized.jsonld

Each class directory contains:

- **schema.json**: The JSON Schema for the class.
- **examples/**: A folder with example files in various formats:
  - `example.json`: A key-value representation of the schema.
  - `example-normalized.json`: An NGSI v2 normalized example.
  - `example.jsonld`: A JSON-LD example with context.
  - `example-normalized.jsonld`: An NGSI-LD normalized example.

Additionally, the root directory includes:

- **ADOPTERS.yaml**: A file listing adopters of the data model.
- **notes.yaml**: A file containing notes about the data model.

Generator Parameters
--------------------

The ``JSONSchemaGenerator`` constructor accepts the following parameters:

- ``model`` (DomainModel): An instance of the DomainModel class representing the B-UML model.
- ``output_dir`` (str, optional): The output directory where the generated code will be saved. Defaults to ``<<current_directory>>/output``.
- ``mode`` (str, optional): The generation mode, either ``'regular'`` or ``'smart_data'``. Defaults to ``'regular'``.

