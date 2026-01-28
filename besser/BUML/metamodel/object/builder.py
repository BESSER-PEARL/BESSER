from besser.BUML.metamodel.object import Object, AttributeLink, DataValue

class ObjectBuilder:
    def __init__(self, classifier):
        self.classifier = classifier
        self._name = None
        self._attributes = {}
        self._links = []

    def name(self, name):
        """Set the name of the object."""
        self._name = name
        return self

    def attributes(self, **kwargs):
        """Set attributes of the object."""
        self._attributes.update(kwargs)
        return self

    def link(self, target_obj, end_name:str):
        """Create a link to another object using the association end name."""
        # Find the end and association by end_name
        end_ = None
        for end in self.classifier.all_association_ends():
            if end.name == end_name:
                end_ = end
                break
        if not end_:
            raise ValueError(f"Association end '{end_name}' not found in class '{self.classifier.name}'")

        self._links.append((target_obj, end_))
        return self

    def build(self):
        """Build the Object instance."""
        if not self._name:
            raise ValueError("Object must have a name")

        # Build Object
        obj = Object(name=self._name, classifier=self.classifier)

        # Build AttributeLinks
        for attr_name, value in self._attributes.items():
            # Special handling for "name" attribute to avoid shadowing the object's name property
            # Create AttributeLink directly instead of using setattr
            if attr_name == "name":
                # Find the "name" property in the classifier
                name_property = None
                for attr in self.classifier.attributes | self.classifier.inherited_attributes():
                    if attr.name == "name":
                        name_property = attr
                        break
                
                if name_property:
                    # Create AttributeLink directly for the "name" attribute
                    data_value = DataValue(classifier=name_property.type, value=value)
                    attr_link = AttributeLink(value=data_value, attribute=name_property)
                    obj.add_slot(attr_link)
                else:
                    # If "name" is not a classifier attribute, use setattr (will go through __setattr__)
                    setattr(obj, attr_name, value)
            else:
                setattr(obj, attr_name, value)

        # Build Links
        for target, tgt_end in self._links:
            setattr(obj, tgt_end.name, target)

        return obj
