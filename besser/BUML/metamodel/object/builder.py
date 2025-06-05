from besser.BUML.metamodel.object import Object

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
            setattr(obj, attr_name, value)

        # Build Links
        for target, tgt_end in self._links:
            setattr(obj, tgt_end.name, target)

        return obj
