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

    def link_to(self, target_obj, end_name:str):
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
        if not self._name:
            raise ValueError("Object must have a name")

        # Build AttributeLinks
        slots = []
        for attr_name, value in self._attributes.items():
            prop = next((a for a in self.classifier.attributes if a.name == attr_name), None)
            if not prop:
                parents = self.classifier.all_parents()
                prop = next((a for p in parents for a in p.attributes if a.name == attr_name), None)
            if not prop:
                raise ValueError(f"Attribute '{attr_name}' not found in class '{self.classifier.name}'")
            data_value = DataValue(classifier=prop.type, value=value)
            slots.append(AttributeLink(attribute=prop, value=data_value))

        obj = Object(name=self._name, classifier=self.classifier, slots=slots)

        # Build Links
        for target, tgt_end in self._links:
            # Find the source end
            association = tgt_end.owner
            src_end = next((end for end in association.ends if end != tgt_end), None)

            # Create the link
            link = Link(
                name=f"{self._name}_to_{target.name}",
                association=association,
                connections=[
                    LinkEnd(name=src_end.name, association_end=src_end, object=obj),
                    LinkEnd(name=tgt_end.name, association_end=tgt_end, object=target)
                ]
            )

        return obj