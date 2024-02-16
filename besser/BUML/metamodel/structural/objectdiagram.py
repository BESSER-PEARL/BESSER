from besser.BUML.metamodel.structural import NamedElement, Property, Type,Association



class AttributeLink:
    def __init__(self, name,value=None, prop_type=None, is_id=False,property = None):

        if prop_type is None:
            prop_type = self.checkType (value)
        if property is None:
            self.attribute = Property(name= name, property_type=prop_type,is_id= is_id)
        self.value = DataValue(value)
    def get_name(self):
        return self.attribute.name
    def get_attribute(self):
        return self.attribute
    def __str__(self):
        return "Attribute Link " + str(self.attribute) + " value: " + str(self.value)
    def checkType(self, value):
        if value.isdigit():
            try:
                int (value)
                return "int"
            except:
                return "real"

        else:
            return "str"

class Instance(NamedElement):

    classifier = None
    slots = None
    ownedLink = None
    linkEnd = None
    def __init__(self):
        self.classifier:Type



class Object(Instance):
    def __init__(self):
        super().__init__()
        self.slots: list[AttributeLink] = []
        self.linkEnd: list[LinkEnd] = []
    def add_slot(self, slot):
        self.slots.append(slot)

    def add_to_link_end(self, link):
        self.linkEnd.append(link)
    def get_slots(self):
        return self.slots
    def __str__(self):
        toRet =""
        for s in self.slots:
            toRet = toRet + str(s) +'\n'
        return toRet


class DataValue(Instance):
    def __init__(self,value):
        super().__init__()
        self.value = value
    def set_value(self, val):
          self.value = val
    def get_value(self):
       return self.value

class LinkEnd(NamedElement):
    def __init__(self,name,ins):
        super().__init__(name)
        self.associationEnd: Property = None
        self.association: Association = None
        self.instance= ins

class Link(NamedElement):
    def __init__(self):
        self.connection : list[LinkEnd] = []
    def add_to_connection(self,linkEnd):
        self.connection.append(linkEnd)


