from besser.BUML.metamodel.structural import NamedElement, Property, Type,Association

class AttributeLink:
    def __init__(self):
        self.attribute = Property()
class Instance(NamedElement):
    def __init__(self):
        self.classifier: list[Type]
        self.slot: list[AttributeLink]
        self.ownedLink: list[Link]
        self.linkEnd : list[LinkEnd]

class Object(Instance):
    def __init__(self):
        super().__init__()

class DataValue(Instance):
    def __init__(self):
        super().__init__()


class Link(NamedElement):
    def __init__(self):
        self.connection = list[LinkEnd]
class LinkEnd(NamedElement):
    def __init__(self):
        self.associationEnd = list[Property]

