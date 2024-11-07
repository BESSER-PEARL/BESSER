# Generated B-UML Model
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType
)

# Enumerations
dsd: Enumeration = Enumeration(
    name="dsd",
    literals={
            EnumerationLiteral(name="set3")
    }
)

SetEnum: Enumeration = Enumeration(
    name="SetEnum",
    literals={
            EnumerationLiteral(name="set1")
    }
)

# Classes
Classname = Class(name="Classname")
Library = Class(name="Library")
Worker = Class(name="Worker")
dfg = Class(name="dfg")
Book = Class(name="Book")
Class1 = Class(name="Class1")
Class2 = Class(name="Class2")
Cartoon = Class(name="Cartoon")

# Classname class attributes and methods
Classname_dd: Property = Property(name="dd", type=IntegerType)
Classname.attributes={Classname_dd}

# Library class attributes and methods
Library_iteeem: Property = Property(name="iteeem", type=SetEnum)
Library_item: Property = Property(name="item", type=StringType, visibility="private")
Library.attributes={Library_iteeem, Library_item}

# Worker class attributes and methods
Worker_fild: Property = Property(name="fild", type=IntegerType)
Worker_m_notify: Method = Method(name="notify", parameters={})
Worker_m_findbook: Method = Method(name="findbook", parameters={}, type=Class1)
Worker.attributes={Worker_fild}
Worker.methods={Worker_m_notify, Worker_m_findbook}

# dfg class attributes and methods
dfg_field: Property = Property(name="field", type=StringType)
dfg.attributes={dfg_field}

# Book class attributes and methods
Book_name: Property = Property(name="name", type=StringType)
Book.attributes={Book_name}

# Class1 class attributes and methods

# Class2 class attributes and methods
Class2_field: Property = Property(name="field", type=StringType)
Class2.attributes={Class2_field}

# Cartoon class attributes and methods

# Relationships
Classname_dfg_association_1: BinaryAssociation = BinaryAssociation(
    name="Classname_dfg_association_1",
    ends={
        Property(name="AgregEdge", type=dfg, multiplicity=Multiplicity(1, 1)),
        Property(name="Classname_end", type=Classname, multiplicity=Multiplicity(1, 1), is_navigable=False)
    }
)
Worker_Library_association: BinaryAssociation = BinaryAssociation(
    name="Worker_Library_association",
    ends={
        Property(name="employed", type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=Worker, multiplicity=Multiplicity(1, 1))
    }
)
Book_Library_association: BinaryAssociation = BinaryAssociation(
    name="Book_Library_association",
    ends={
        Property(name="child", type=Library, multiplicity=Multiplicity(5, 9)),
        Property(name="parent", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)
dfg_Classname_composite: BinaryAssociation = BinaryAssociation(
    name="dfg_Classname_composite",
    ends={
        Property(name="CompositeeeEdge", type=Classname, multiplicity=Multiplicity(1, 1)),
        Property(name="dfg_end", type=dfg, multiplicity=Multiplicity(1, 1), is_composite=True)
    }
)
Classname_dfg_association: BinaryAssociation = BinaryAssociation(
    name="Classname_dfg_association",
    ends={
        Property(name="Classname_end", type=Classname, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property(name="dfg_end", type=dfg, multiplicity=Multiplicity(1, 1))
    }
)

# Generalizations
gen_Cartoon_Book = Generalization(general=Book, specific=Cartoon)
gen_Classname_dfg = Generalization(general=dfg, specific=Classname)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Classname, Library, Worker, dfg, Book, Class1, Class2, Cartoon, dsd, SetEnum},
    associations={Classname_dfg_association_1, Worker_Library_association, Book_Library_association, dfg_Classname_composite, Classname_dfg_association},
    generalizations={gen_Cartoon_Book, gen_Classname_dfg}
)
