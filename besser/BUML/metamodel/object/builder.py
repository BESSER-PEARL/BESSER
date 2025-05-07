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
            ############# TO COMPLETE.... Check if the value actually has the same classifier type
            attr_link = AttributeLink(attribute=prop, value=data_value)
            slots.append(attr_link)

        obj = Object(name=self._name, classifier=self.classifier, slots=slots)

        # Build Links from associations
        for target, assoc in self._links:
            # Encuentra los ends correctos automáticamente
            src_end = next(end for end in assoc.ends if end.type == self.classifier)
            tgt_end = next(end for end in assoc.ends if end.type == target.classifier)

            obj_end = LinkEnd(name=src_end.name, association_end=src_end, object=obj)
            target_end = LinkEnd(name=tgt_end.name, association_end=tgt_end, object=target)

            link = Link(
                name=f"{self._name}_to_{target.name}",
                association=assoc,
                connections=[obj_end, target_end]
            )

            # Save Link
            if not hasattr(obj, "_outgoing_links"):
                obj._outgoing_links = []
            obj._outgoing_links.append(link)

        return obj
