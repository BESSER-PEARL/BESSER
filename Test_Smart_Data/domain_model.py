# Generated B-UML Model
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
        Property(name="ue_1", type=UE, multiplicity=Multiplicity(1, 1)),
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
        Property(name="nrcelldu_1", type=NRCellDU, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Network-DataSchema",
    types={NRCellCU, NRCellDU, BWP, UE, PDUSession, SynchronizationSignal, TDD, communicationType},
    associations={NRCellDU_BWP, NRCellDU_UE, UE_PDUSession, UE_SynchronizationSignal, NRCellCU_NRCellDU, TDD_NRCellDU},
    generalizations={}
)


from besser.generators.json import JSONSchemaGenerator


# Chemin de sortie pour les schémas générés
output_dir = "output_directory"

# Créer le générateur en mode Smart Data
generator = JSONSchemaGenerator(domain_model, output_dir, mode='smart_data')

# Générer les schémas
generator.generate()
