from besser.BUML.metamodel.structural import NamedElement, Property, Type,Association

class ObjectProperty(Property):
    def __init__(self, name, value =None, prop_type=None, is_id = False):
        if prop_type is None and value is not None:
            prop_type = self.checkType (value)

        super().__init__(name, property_type=prop_type, is_id = is_id)
        self.value = value
    def set_value (self,val):
        self.value = val
    def get_value(self):
        return self.value

    def __repr__(self):
        return f'ObjectProperty({self.name}, {self.visibility}, {self.type}, {self.multiplicity}, is_composite={self.is_composite}, is_id={self.is_id}, value = {self.value})'

    def checkType(self, value):
        if value.isdigit():
            try:
                int (value)
                return "int"
            except:
                return "real"

        else:
            return "str"


class AttributeLink:
    def __init__(self, name, value=None, prop_type=None, is_id=False):
        self.attribute = ObjectProperty(name, value, prop_type, is_id)
    def get_attribute(self):
        return self.attribute
    def __str__(self):
        return "Attribute Link " + str(self.attribute)
class Instance(NamedElement):

    classifier = None
    slots = None
    ownedLink = None
    linkEnd = None
    def __init__(self):
        self.classifier: list[Type] =[]
        self.slots: list[AttributeLink] =[]
        self.ownedLink: list[Link]=[]
        self.linkEnd : list[LinkEnd]=[]
    def add_to_link(self,link):
        self.ownedLink.append(link)
    def add_slot(self,slot):
        self.slots.append(slot)
    def get_slots(self):
        return self.slots

class Object(Instance):
    def __init__(self):
        super().__init__()
    def __str__(self):
        toRet =""
        for s in self.slots:
            toRet = toRet + str(s) +'\n'
        return toRet


class DataValue(Instance):
    def __init__(self):
        super().__init__()

class LinkEnd(NamedElement):
    def __init__(self,name,ins):
        super().__init__(name)
        self.associationEnd = list[Property]
        self.instance= ins
class Link(NamedElement):
    def __init__(self):
        self.connection : list[LinkEnd] = []
    def add_to_connection(self,linkEnd):
        self.connection.append(linkEnd)


