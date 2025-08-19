#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)
from besser.generators.json.json_schema_generator import JSONSchemaGenerator

# Enumerations
communicationType: Enumeration = Enumeration(
    name="communicationType",
    literals={
            EnumerationLiteral(name="FDD"),
            EnumerationLiteral(name="TDD")
    }
)

# Classes
TDD = Class(name="TDD")
NRCellCU = Class(name="NRCellCU")
NRCellDU = Class(name="NRCellDU")
BWP = Class(name="BWP")
PDUSession = Class(name="PDUSession")
UE = Class(name="UE")
SynchronizationSignal = Class(name="SynchronizationSignal")

# TDD class attributes and methods
TDD_arfcn: Property = Property(name="arfcn", type=IntegerType)
TDD_band: Property = Property(name="band", type=IntegerType)
TDD.attributes={TDD_band, TDD_arfcn}

# NRCellCU class attributes and methods
NRCellCU_nRcellId: Property = Property(name="nRcellId", type=IntegerType)
NRCellCU_physicalCellId: Property = Property(name="physicalCellId", type=IntegerType)
NRCellCU.attributes={NRCellCU_physicalCellId, NRCellCU_nRcellId}

# NRCellDU class attributes and methods
NRCellDU_nRCellid: Property = Property(name="nRCellid", type=IntegerType)
NRCellDU_physicalCellId: Property = Property(name="physicalCellId", type=IntegerType)
NRCellDU_ssb: Property = Property(name="ssb", type=IntegerType)
NRCellDU_policy: Property = Property(name="policy", type=communicationType)
NRCellDU.attributes={NRCellDU_nRCellid, NRCellDU_ssb, NRCellDU_physicalCellId, NRCellDU_policy}

# BWP class attributes and methods
BWP_subCarrierSpacing: Property = Property(name="subCarrierSpacing", type=IntegerType)
BWP_numberofRBs: Property = Property(name="numberofRBs", type=IntegerType)
BWP.attributes={BWP_subCarrierSpacing, BWP_numberofRBs}

# PDUSession class attributes and methods
PDUSession_id: Property = Property(name="id", type=IntegerType)
PDUSession.attributes={PDUSession_id}

# UE class attributes and methods
UE_position: Property = Property(name="position", type=IntegerType)
UE_id: Property = Property(name="id", type=IntegerType)
UE_rnti: Property = Property(name="rnti", type=StringType)
UE.attributes={UE_rnti, UE_id, UE_position}

# SynchronizationSignal class attributes and methods
SynchronizationSignal_rsrp: Property = Property(name="rsrp", type=IntegerType)
SynchronizationSignal_sinr: Property = Property(name="sinr", type=FloatType)
SynchronizationSignal_rsrq: Property = Property(name="rsrq", type=FloatType)
SynchronizationSignal.attributes={SynchronizationSignal_sinr, SynchronizationSignal_rsrp, SynchronizationSignal_rsrq}

# Relationships
NRCellCU_NRCellDU: BinaryAssociation = BinaryAssociation(
    name="NRCellCU_NRCellDU",
    ends={
        Property(name="nrCellDU_by_duId", type=NRCellDU, multiplicity=Multiplicity(1, 9999)),
        Property(name="nrcellcu", type=NRCellCU, multiplicity=Multiplicity(1, 1))
    }
)
NRCellDU_BWP: BinaryAssociation = BinaryAssociation(
    name="NRCellDU_BWP",
    ends={
        Property(name="nrcelldu", type=NRCellDU, multiplicity=Multiplicity(1, 1)),
        Property(name="bwp", type=BWP, multiplicity=Multiplicity(1, 1))
    }
)
UE_SynchronizationSignal: BinaryAssociation = BinaryAssociation(
    name="UE_SynchronizationSignal",
    ends={
        Property(name="ue", type=UE, multiplicity=Multiplicity(1, 1)),
        Property(name="synchronizationSignal", type=SynchronizationSignal, multiplicity=Multiplicity(1, 9999))
    }
)
UE_PDUSession: BinaryAssociation = BinaryAssociation(
    name="UE_PDUSession",
    ends={
        Property(name="ue", type=UE, multiplicity=Multiplicity(1, 1)),
        Property(name="pdus", type=PDUSession, multiplicity=Multiplicity(0, 9999))
    }
)
NRCellDU_UE: BinaryAssociation = BinaryAssociation(
    name="NRCellDU_UE",
    ends={
        Property(name="du", type=NRCellDU, multiplicity=Multiplicity(1, 1)),
        Property(name="userEquipments", type=UE, multiplicity=Multiplicity(0, 9999))
    }
)
TDD_NRCellDU: BinaryAssociation = BinaryAssociation(
    name="TDD_NRCellDU",
    ends={
        Property(name="nrcelldu", type=NRCellDU, multiplicity=Multiplicity(1, 1)),
        Property(name="tdd", type=TDD, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={TDD, NRCellCU, NRCellDU, BWP, PDUSession, UE, SynchronizationSignal, communicationType},
    associations={NRCellCU_NRCellDU, NRCellDU_BWP, UE_SynchronizationSignal, UE_PDUSession, NRCellDU_UE, TDD_NRCellDU},
    generalizations={}
)

# Test the generator
generator = JSONSchemaGenerator(domain_model, output_dir="./test_output", mode="smart_data")
generator.generate()

print("Test completed! Check the ./test_output directory for generated files.")
