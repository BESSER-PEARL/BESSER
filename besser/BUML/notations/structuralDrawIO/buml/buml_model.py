# Generated B-UML Model
from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType
)

# Enumerations
SetEnum: Enumeration = Enumeration(
    name="SetEnum",
    literals={EnumerationLiteral(name="set1")}
)

dsd: Enumeration = Enumeration(
    name="dsd",
    literals={EnumerationLiteral(name="set3")}
)

# Classes
Class1 = Class(name="Class1")
Class2 = Class(name="Class2")
Book = Class(name="Book")
Library = Class(name="Library")
dfg = Class(name="dfg")
Worker = Class(name="Worker")
Cartoon = Class(name="Cartoon")
Classname = Class(name="Classname")

# Class1 class attributes and methods

# Class2 class attributes and methods
Class2_field: Property = Property(name="field", type=StringType, visibility="public")
Class2.attributes={Class2_field}

# Book class attributes and methods
Book_name: Property = Property(name="name", type=StringType, visibility="public")
Book.attributes={Book_name}

# Library class attributes and methods
Library_iteeem: Property = Property(name="iteeem", type=SetEnum, visibility="public")
Library_item: Property = Property(name="item", type=StringType, visibility="private")
Library.attributes={Library_iteeem, Library_item}

# dfg class attributes and methods
dfg_field: Property = Property(name="field", type=StringType, visibility="public")
dfg.attributes={dfg_field}

# Worker class attributes and methods
Worker_fild: Property = Property(name="fild", type=IntegerType, visibility="public")
Worker_m_findbook: Method = Method(name="findbook", visibility="public", parameters={}, type=Class1)
Worker_m_notify: Method = Method(name="notify", visibility="public", parameters={})
Worker.attributes={Worker_fild}
Worker.methods={Worker_m_findbook, Worker_m_notify}

# Cartoon class attributes and methods

# Classname class attributes and methods
Classname_dd: Property = Property(name="dd", type=IntegerType, visibility="public")
Classname.attributes={Classname_dd}

# Relationships
dfg_Classname_composite: BinaryAssociation = BinaryAssociation(name="dfg_Classname_composite", ends={Property(name="CompositeeeEdge", type=Classname, multiplicity=Multiplicity(1, 9999),is_navigable=True, is_composite=False), Property(name="dfg_end", type=dfg, multiplicity=Multiplicity(1, 9999),is_navigable=True, is_composite=True)})
Classname_dfg_association: BinaryAssociation = BinaryAssociation(name="Classname_dfg_association", ends={Property(name="dfg_end", type=dfg, multiplicity=Multiplicity(1, 9999),is_navigable=True, is_composite=False), Property(name="Classname_end", type=Classname, multiplicity=Multiplicity(1, 1),is_navigable=False, is_composite=False)})
Classname_dfg_association_1: BinaryAssociation = BinaryAssociation(name="Classname_dfg_association_1", ends={Property(name="AgregEdge", type=dfg, multiplicity=Multiplicity(1, 9999),is_navigable=True, is_composite=False), Property(name="Classname_end", type=Classname, multiplicity=Multiplicity(1, 9999),is_navigable=False, is_composite=False)})
Worker_Library_association: BinaryAssociation = BinaryAssociation(name="Worker_Library_association", ends={Property(name="employed", type=Library, multiplicity=Multiplicity(1, 1),is_navigable=True, is_composite=False), Property(name="has", type=Worker, multiplicity=Multiplicity(1, 9999),is_navigable=True, is_composite=False)})
Book_Library_association: BinaryAssociation = BinaryAssociation(name="Book_Library_association", ends={Property(name="parent", type=Book, multiplicity=Multiplicity(0, 9999),is_navigable=True, is_composite=False), Property(name="child", type=Library, multiplicity=Multiplicity(5, 9),is_navigable=True, is_composite=False)})

# Generalizations
gen_Classname_dfg = Generalization(general=dfg,specific=Classname)
gen_Cartoon_Book = Generalization(general=Book,specific=Cartoon)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Class1, Class2, Book, Library, dfg, Worker, Cartoon, Classname, SetEnum, dsd},
    associations={dfg_Classname_composite, Classname_dfg_association, Classname_dfg_association_1, Worker_Library_association, Book_Library_association},
    generalizations={gen_Classname_dfg, gen_Cartoon_Book}
)
