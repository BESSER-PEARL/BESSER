####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)

# Classes
Person = Class(name="Person")
Organization = Class(name="Organization")
Place = Class(name="Place")

# Person class attributes and methods
Person_name: Property = Property(name="name", type=StringType)
Person_age: Property = Property(name="age", type=IntegerType)
Person_m_knows: Method = Method(name="knows", parameters={Parameter(name='person', type=Person)})
Person.attributes={Person_age, Person_name}
Person.methods={Person_m_knows}

# Organization class attributes and methods
Organization_name: Property = Property(name="name", type=StringType)
Organization_industry: Property = Property(name="industry", type=StringType)
Organization.attributes={Organization_industry, Organization_name}

# Place class attributes and methods
Place_name: Property = Property(name="name", type=StringType)
Place_countryCode: Property = Property(name="countryCode", type=StringType)
Place.attributes={Place_countryCode, Place_name}

# Relationships
worksAt: BinaryAssociation = BinaryAssociation(
    name="worksAt",
    ends={
        Property(name="person", type=Person, multiplicity=Multiplicity(1, 1)),
        Property(name="worksAt", type=Organization, multiplicity=Multiplicity(1, 1))
    }
)
location: BinaryAssociation = BinaryAssociation(
    name="location",
    ends={
        Property(name="organization", type=Organization, multiplicity=Multiplicity(1, 1)),
        Property(name="location", type=Place, multiplicity=Multiplicity(1, 1))
    }
)

# Domain Model
domain_model = DomainModel(
    name="KG_Imported_Diagram",
    types={Person, Organization, Place},
    associations={worksAt, location},
    generalizations={}
)
