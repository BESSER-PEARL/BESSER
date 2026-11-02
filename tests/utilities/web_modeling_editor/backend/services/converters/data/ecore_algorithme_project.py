"""
B-UML Domain Model: algorithme
Generated from Ecore model
Date: 2026-11-02 10:10:18
"""

from besser.BUML.metamodel.structural.structural import (
    DomainModel,
    Class,
    Property,
    Method,
    Parameter,
    Association,
    BinaryAssociation,
    Generalization,
    Multiplicity,
    Enumeration,
    EnumerationLiteral,
    Metadata,
    Constraint,
    StringType, IntegerType, FloatType, BooleanType,
    DateType, DateTimeType, TimeType, TimeDeltaType
)
from besser.BUML.metamodel.project import Project as BUMLProject


# ==================== ENUMERATIONS ====================

TypeRessource = Enumeration(name="TypeRessource")
TypeRessource_lit_SCRIPT = EnumerationLiteral(name="SCRIPT", owner=TypeRessource)
TypeRessource.add_literal(TypeRessource_lit_SCRIPT)
TypeRessource_lit_CALCUL = EnumerationLiteral(name="CALCUL", owner=TypeRessource)
TypeRessource.add_literal(TypeRessource_lit_CALCUL)


# ==================== CLASSES ====================

PortSortie = Class(name="PortSortie")
Port = Class(name="Port", is_abstract=True)
Documentation = Class(name="Documentation")
AlgorithmeElement = Class(name="AlgorithmeElement", is_abstract=True)
Algorithme = Class(name="Algorithme")
PortEntree = Class(name="PortEntree")
Ressource = Class(name="Ressource")

# ==================== ATTRIBUTES ====================

# Port attributes
nom_Port = Property(
    name="nom",
    type=StringType,
    owner=Port,
    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1)
)
Port.add_attribute(nom_Port)

# Documentation attributes
texte_Documentation = Property(
    name="texte",
    type=StringType,
    owner=Documentation,
    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1)
)
Documentation.add_attribute(texte_Documentation)

# Algorithme attributes
nom_Algorithme = Property(
    name="nom",
    type=StringType,
    owner=Algorithme,
    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1)
)
Algorithme.add_attribute(nom_Algorithme)

# Ressource attributes
chemin_Ressource = Property(
    name="chemin",
    type=StringType,
    owner=Ressource,
    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1)
)
Ressource.add_attribute(chemin_Ressource)
type_Ressource = Property(
    name="type",
    type=TypeRessource,
    owner=Ressource,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=1), is_optional=True
)
Ressource.add_attribute(type_Ressource)


# ==================== METHODS ====================


# ==================== GENERALIZATIONS ====================

gen_0 = Generalization(
    general=AlgorithmeElement,
    specific=Documentation
)
gen_1 = Generalization(
    general=Port,
    specific=PortSortie
)
gen_2 = Generalization(
    general=AlgorithmeElement,
    specific=Ressource
)
gen_3 = Generalization(
    general=Port,
    specific=PortEntree
)

# ==================== ASSOCIATIONS ====================

# Association: Algorithme_algorithmeElements
end_0_0 = Property(
    name="algorithmeElements",
    type=AlgorithmeElement,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999), is_composite=True
)
end_0_1 = Property(
    name="inverse_Algorithme_algorithmeElements",
    type=Algorithme,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999), is_navigable=False
)
assoc_0 = BinaryAssociation(
    name="Algorithme_algorithmeElements",
    ends={end_0_0, end_0_1}
)

# Association: Documentation_elements
end_1_0 = Property(
    name="inverse_Documentation_elements",
    type=Documentation,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999), is_navigable=False
)
end_1_1 = Property(
    name="elements",
    type=AlgorithmeElement,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999)
)
assoc_1 = BinaryAssociation(
    name="Documentation_elements",
    ends={end_1_0, end_1_1}
)

# Association: Ressource_ports
end_2_0 = Property(
    name="inverse_Ressource_ports",
    type=Ressource,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999), is_navigable=False
)
end_2_1 = Property(
    name="ports",
    type=Port,
    owner=None,
    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999), is_composite=True
)
assoc_2 = BinaryAssociation(
    name="Ressource_ports",
    ends={end_2_0, end_2_1}
)


# ==================== CONSTRAINTS ====================


# ==================== DOMAIN MODEL ====================

metadata = Metadata(
    description="Converted from algorithme.ecore"
)

domain_model = DomainModel(
    name="algorithme",
    types={PortSortie, Port, Documentation, AlgorithmeElement, Algorithme, PortEntree, Ressource} | {TypeRessource},
    associations={assoc_0, assoc_1, assoc_2},
    generalizations={gen_0, gen_1, gen_2, gen_3},
    constraints={},
    metadata=metadata
)


# ==================== PROJECT DEFINITION ====================
# This section allows the model to be directly opened in the B-UML editor

buml_project = BUMLProject(
    name="algorithme",
    models=[domain_model],
    owner="User",
    metadata=metadata
)
