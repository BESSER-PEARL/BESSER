# Generated B-UML Model
from besser.BUML.metamodel.structural import *

# Enumerations
BOOKCATEGORY = Enumeration(name="BOOKCATEGORY", literals={EnumerationLiteral(name="ROMANCE"), EnumerationLiteral(name="HISTORY"), EnumerationLiteral(name="SCIENCE")})

# Classes
Classname: Class = Class(name="Classname")
Book: Class = Class(name="Book")
Library: Class = Class(name="Library")

# Classname class attributes and methods
Classname_field: Property = Property(name="field", type=PrimitiveDataType("int"), visibility="public")
Classname_m_test: Method = Method(name="test", visibility="public", parameters={Parameter(name="attr1", type=PrimitiveDataType("str"))}, type=PrimitiveDataType("str"))
Classname_m_notify: Method = Method(name="notify", visibility="public", parameters={Parameter(name="sms", type=PrimitiveDataType("str"), default_value="message")})
Classname.attributes={Classname_field}
Classname.methods={Classname_m_test, Classname_m_notify}

# Book class attributes and methods
Book_title: Property = Property(name="title", type=PrimitiveDataType("str"), visibility="public")
Book_release: Property = Property(name="release", type=PrimitiveDataType("date"), visibility="public")
Book_pages: Property = Property(name="pages", type=PrimitiveDataType("int"), visibility="public")
Book_category: Property = Property(name="category", type=BOOKCATEGORY, visibility="public")
Book.attributes={Book_title, Book_release, Book_pages, Book_category}

# Library class attributes and methods
Library_address: Property = Property(name="address", type=PrimitiveDataType("str"), visibility="public")
Library_name: Property = Property(name="name", type=PrimitiveDataType("str"), visibility="public")
Library.attributes={Library_address, Library_name}

# Relationships
Book_Book_association: BinaryAssociation = BinaryAssociation(name="Book_Book_association", ends={Property(name="parent", type=Book, multiplicity=Multiplicity(1, 9999)), Property(name="child", type=Book, multiplicity=Multiplicity(1, 9999))})
Library_Book_association: BinaryAssociation = BinaryAssociation(name="Library_Book_association", ends={Property(name="child", type=Library, multiplicity=Multiplicity(1, 1)), Property(name="belongs_to", type=Book, multiplicity=Multiplicity(1, 1))})

# Generalizations
gen_Classname_Library = Generalization(general=Library, specific=Classname)

# Domain Model
domain_model = DomainModel(
    name="Generated Model",
    types={Classname, Book, Library},
    associations={Book_Book_association, Library_Book_association},
    generalizations={gen_Classname_Library},
    enumerations={BOOKCATEGORY}
)
