import string

from besser.BUML.metamodel.structural import NamedElement, Property, Type,Association



class AttributeLink:
    def __init__(self, name,value=None, prop_type=None, is_id=False,property = None):
        """An attribute link is a named slot in an instance, which holds the value of an attribute
       Args:
           name (str): the name of the property
           prop_type: type of the property.
           is_id: Boolean to identify if property is ID
           property: property from the class diagram
       Attributes:
           value: value of the property
           attribute: property.
       """

        if prop_type is None:
            prop_type = self.checkType (value)
        if property is None:
            self.attribute = Property(name= name, property_type=prop_type,is_id= is_id)
        self.value = DataValue(value)
    @property
    def name(self) -> string:
        """str: Get the name of the property."""
        return self.attribute.name

    @property
    def get_attribute(self)->Property:
        """str: Get the property."""
        return self.attribute
    def __repr__(self) -> string:
        """str: printing the value of attribute link."""
        return "Attribute Link " + str(self.attribute) + " value: " + str(self.value)
    def checkType(self, value) ->string:
        """str: return the type of value (int, string, real)"""
        if value.isdigit():
            try:
                int (value)
                return "int"
            except:
                return "real"

        else:
            return "str"

class Instance(NamedElement):
    """The instance defines an entity to which a set of operations can be applied and
        which has a state that stores the effects of the operations.
           Attributes:
               classifier: classname of the instance
               slots: list of properties
               ownedLink: the set of Links that are owned by the Instance.
               linkEnd: The set of LinkEnds of the connected Links that are attached to the
Instance.
           """

    def __init__(self):
        self.classifier:Type = None
        self.slots = None
        self.ownedLink = None
        self.linkEnd = None



class Object(Instance):
    """ An object is an instance that originates from a class.
     Attributes:
          slots: list of properties
          linkEnd: The set of LinkEnds of the connected Links that are attached to the
Instance.
    """
    def __init__(self):
        super().__init__()
        self.slots: list[AttributeLink] = []
        self.linkEnd: list[LinkEnd] = []
    def add_slot(self, slot):
        """ Method to add attribute link to slots"""
        self.slots.append(slot)

    def add_to_link_end(self, link):
        """ Method to add link to linkend"""

        self.linkEnd.append(link)

    def get_slots(self):
        """ slots: Method to retrieve slots"""

        return self.slots

    def __repr__(self):
         """str: printing the value of object."""
         toRet =""
         for s in self.slots:
            toRet = toRet + str(s) +'\n'
         return toRet


class DataValue(Instance):
    """ An DataValue is an instance with no identity.
         Attributes:
              value: value of the property
    Instance.
        """
    def __init__(self,value):
        super().__init__()
        self.__value = value

    @property
    def value(self):
       """Method to retrieve Value"""
       return self.__value


    @value.setter
    def value(self, val):
        """Method to set Value"""

        self.__value = val

class LinkEnd(NamedElement):
    """ A link end is an end point of a link.
    args:
    name : name of the link
    ins: Instance of the link
    Attributes:
        associationEnd: property pointing to link
        instance : instance of link
            """
    def __init__(self,name,ins):
        super().__init__(name)
        self.associationEnd: Property = None
        self.instance= ins

class Link(NamedElement):
    """ A link contains list of linkends.
    attributes:
    connection: list of linkends
    """
    def __init__(self):
        self.connection : list[LinkEnd] = []
    def add_to_connection(self,linkEnd):
        """Method to add linkend"""
        self.connection.append(linkEnd)
