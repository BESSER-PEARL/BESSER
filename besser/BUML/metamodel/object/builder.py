from besser.BUML.metamodel.object import Object, AttributeLink, Link, LinkEnd, DataValue

class ObjectBuilder:
    def __init__(self, classifier):
        self.classifier = classifier
        self._name = None
        self._attributes = {}
        self._links = []

    def name(self, name):
        self._name = name
        return self

    def attributes(self, **kwargs):
        self._attributes.update(kwargs)
        return self

    def link_to(self, target_obj, association):
        self._links.append((target_obj, association))
        return self

    def build(self):
        if not self._name:
            raise ValueError("Object must have a name")

        # Build AttributeLinks
        slots = []
        for attr_name, value in self._attributes.items():
            prop = next((a for a in self.classifier.attributes if a.name == attr_name), None)
            if not prop:
                raise ValueError(f"Attribute '{attr_name}' not found in class '{self.classifier.name}'")
            data_value = DataValue(classifier=prop.type, value=value)
            slots.append(AttributeLink(attribute=prop, value=data_value))

        obj = Object(name=self._name, classifier=self.classifier, slots=slots)

        # Build Links
        for target, assoc in self._links:
            # Find an association end whose type matches the source or one of its parents
            src_types = [self.classifier] + list(self.classifier.all_parents())
            src_end = next((end for end in assoc.ends if end.type in src_types), None)
            if not src_end:
                raise ValueError(
                    f"The class '{self.classifier.name}' is not part of the association '{assoc.name}'"
                    )
            # Find an association end whose type matches the target or one of its parents
            tgt_types = [target.classifier] + list(target.classifier.all_parents())
            tgt_end = next((end for end in assoc.ends if end.type in tgt_types), None)
            if not tgt_end:
                raise ValueError(
                    f"The class '{target.classifier.name}' is not part of the association '{assoc.name}'"
                    )

            # Create the link
            link = Link(
                name=f"{self._name}_to_{target.name}",
                association=assoc,
                connections=[
                    LinkEnd(name=src_end.name, association_end=src_end, object=obj),
                    LinkEnd(name=tgt_end.name, association_end=tgt_end, object=target)
                ]
            )

        return obj