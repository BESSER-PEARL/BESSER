import string
import datetime
from typing import Union
from besser.BUML.metamodel.structural import (
    NamedElement,
    Property,
    Type,
    Association,
    PrimitiveDataType,
    UNLIMITED_MAX_MULTIPLICITY,
)

class AttributeLink():
    """An attribute link is a named slot in an instance, which holds the value of an attribute
    
    Args:
        name (str): the name of the attribute link
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the attribute link.
        value (DataValue): the value of the attribute.
        attribute (Property): the attribute or property from the structural metamodel.
    """

    def __init__(self, value: "DataValue", attribute: Property):
        self.__attribute: Property = attribute
        self.value: DataValue = value

    @property
    def value(self) -> "DataValue":
        """DataValue: Get value of the attribute."""
        return self.__value

    @value.setter
    def value(self, value: "DataValue"):
        """DataValue: Set the value of the attribute.
        
        Raises:
            TypeError: If the value's classifier type does not match the attribute's type.
        """
        # Validate that the value's type matches the attribute's type
        if value.classifier != self.__attribute.type:
            raise TypeError(f"Type mismatch: attribute '{self.__attribute.name}' expects {self.__attribute.type.name}, "
                           f"but got {value.classifier.name}")
        self.__value = value

    @property
    def attribute(self) -> Property:
        """Property: Get the attribute."""
        return self.__attribute

    @attribute.setter
    def attribute(self, attribute: Property):
        """Property: Set the attribute."""
        self.__attribute = attribute

    def __repr__(self) -> str:
        return f'Attribute Link({self.value}, {self.__attribute})'

class Instance(NamedElement):
    """The instance defines an entity to which a set of operations can be applied and which has a state that stores the effects of the operations.
    
    Args:
        name (str): the name of the instance
        classifier (Type): the classifier of the instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.

    Attributes:
        name (str): inherited from NamedElement, represents the name of the instance.
        classifier (Type): the classifier of the instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
    """

    def __init__(self, name: str, classifier: Type):
        super().__init__(name)
        self.classifier: Type = classifier

    @property
    def classifier(self) -> Type:
        """Type: Get the classifier."""
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier: Type):
        """Type: Set the classifier."""
        self.__classifier = classifier

class Object(Instance):
    """An object is an instance that originates from a class.

    Args:
        name (str): the name of the object instance
        classifier (Type): the classifier of the object instance. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
        slots (list[AttributeLink]): list of properties of the instance

    Attributes:
        name (str): inherited from NamedElement, represents the name of the object instance.
        classifier (Type): Inherited from Instance, represents the classifier of the object.
        slots (list[AttributeLink]): list of properties of the instance
    """
    def __init__(self, name: str, classifier: Type, slots: list[AttributeLink] = None):
        super().__init__(name, classifier)
        self.slots = slots if slots is not None else []
        self.__links: set[Link] = set()

    @property
    def name(self):
        """str: Get the name of the object instance."""
        return self.__getattr__("name")

    @name.setter
    def name(self, name: str):
        """str: Set the name of the object instance."""
        object.__setattr__(self, "_name", name)

    @property
    def name_(self):
        """str: Get the name of the object instance."""
        return self._name

    @name_.setter
    def name_(self, name: str):
        """str: Set the name of the object instance."""
        object.__setattr__(self, "_name", name)

    @property
    def slots(self) -> list[AttributeLink]:
        """list[AttributeLink]: Get the list of slots (attributes) of the object instance."""
        return self.__slots

    @slots.setter
    def slots(self, slots: list[AttributeLink]):
        """list[AttributeLink]: Set the list of slots (attributes) of the object instance."""
        self.__slots = slots

    def add_slot(self, slot: AttributeLink):
        """Add a slot (attribute) to the object instance."""
        self.__slots.append(slot)

    @property
    def links(self) -> set:
        """set[Link]: Get the set of links associated with the object instance."""
        return self.__links

    def _add_link(self, link):
        """Add a link to the object instance."""
        self.__links.add(link)

    def _delete_link(self, link):
        """Delete a link from the object instance."""
        self.__links.discard(link)

    def link_ends(self) -> set:
        """Get the set of link ends associated with the object instance."""
        ends = set()
        for link in self.__links:
            l_ends = link.connections
            ends.update(l_ends)
            l_aends = list(l_ends)
            if not(len(l_aends) == 2 and l_aends[0].object == l_aends[1].object):
                for end in l_ends:
                    if end.object == self:
                        ends.discard(end)
        return ends

    def __getattr__(self, item):
        """Get the value of an attribute or link end by its name."""
        for attr in self.__slots:
            if attr.attribute.name == item:
                return attr.value.value

        if item == "name":
            return self.name_

        matches = [le.object for le in self.link_ends() if le.name == item]
        if not matches:
            raise AttributeError(
                f"'{self.name_}' object, instance of the '{self.classifier.name}' class, "
                f"has no attribute or link '{item}'"
            )

        return matches[0] if len(matches) == 1 else set(matches)

    def __setattr__(self, key, value):
        """Set the value of an attribute or create a link end if the key matches an association end."""
        if self._is_internal_attr(key):
            object.__setattr__(self, key, value)
            return

        if self._is_fully_initialized():
            if self._set_slot_value(key, value):
                return
            if self._set_classifier_attribute(key, value):
                return
            if self._handle_association_end(key, value):
                return

            if key != "name":
                raise AttributeError(
                    f"'{self.name_}' object, instance of the '{self.classifier.name}' class, "
                    f"has no attribute or link '{key}'"
                )

        object.__setattr__(self, key, value)

    def _is_internal_attr(self, key: str) -> bool:
        """Check if the key is an internal attribute of the Object class."""
        return key in {"name_", "classifier", "slots", "_Object__links", "_Object__slots"}

    def _is_fully_initialized(self) -> bool:
        """Check if the Object instance is fully initialized."""
        return "_Object__slots" in self.__dict__ and "_Instance__classifier" in self.__dict__

    def _set_slot_value(self, key: str, value) -> bool:
        """Set the value of an attribute link if it exists in the slots."""
        for attr_link in self.__slots:
            if attr_link.attribute.name == key:
                if isinstance(value, DataValue):
                    attr_link.value = value
                else:
                    attr_link.value.value = value
                return True
        return False

    def _set_classifier_attribute(self, key: str, value) -> bool:
        """Set the value of an attribute if it exists in the classifier's attributes."""
        for attr in self.classifier.attributes | self.classifier.inherited_attributes():
            if attr.name == key:
                data_value = DataValue(classifier=attr.type, value=value)
                self.slots.append(AttributeLink(value=data_value, attribute=attr))
                return True
        return False

    def _handle_association_end(self, key: str, value) -> bool:
        """Handle the case where the key matches an association end."""
        for tgt_end in self.classifier.all_association_ends():
            if tgt_end.name != key:
                continue

            association = tgt_end.owner
            src_end = next((end for end in association.ends if end != tgt_end), None)
            old_links = [e_link.owner for e_link in self.link_ends() if e_link.association_end.name == key]

            if isinstance(value, Object):
                self._create_link(src_end, tgt_end, value, association)
            elif isinstance(value, set):
                for item in value:
                    self._create_link(src_end, tgt_end, item, association)
            else:
                raise TypeError(
                    f"Invalid value type for association end '{key}': expected Object or set of Objects, "
                    f"but received {type(value).__name__} with value '{value}'."
                )

            for old_link in old_links:
                self._delete_link(old_link)
            return True
        return False

    def _create_link(self, src_end, tgt_end, target, association):
        """Create a link between the source end and the target object."""
        Link(
            name=f"{self.name_}_to_{target.name}",
            association=association,
            connections=[
                LinkEnd(name=src_end.name, association_end=src_end, object=self),
                LinkEnd(name=tgt_end.name, association_end=tgt_end, object=target)
            ]
        )

    def __repr__(self):
        return f'Object({self.name}, {self.classifier}, {self.slots})'

class DataValue(Instance):
    """ An DataValue represent the value of a property or attribute of an Object.
    
    Args:
        classifier (Type): the classifier of the DataValue. It could be for example a Class or a PrimitiveDataType of the structural metamodel.
        value: value of the property Instance.
    
    Attributes:
        classifier (Type): Inherited from Instance, represents the classifier of the DataValue instance.
        value: value of the property Instance.
    """

    def __init__(self, classifier: Type, value, name=""):
        super().__init__(name, classifier)
        self.value = value

    @property
    def value(self):
        """Method to retrieve Value"""
        return self.__value

    @value.setter
    def value(self, val):
        # Only validate for known primitive types
        if isinstance(self.classifier, PrimitiveDataType):
            expected_type = self._primitive_type_to_python_type(self.classifier.name)
            if expected_type is not None and not isinstance(val, expected_type):
                raise TypeError(
                    f"Invalid value type: expected a value of type '{self.classifier.name}', "
                    f"but received a value of type '{type(val).__name__}' with value '{val}'."
                )
        self.__value = val

    def _primitive_type_to_python_type(self, typename: str):
        """Maps a BESSER primitive type name to the corresponding Python type."""
        mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "time": datetime.time,
            "date": datetime.date,
            "datetime": datetime.datetime,
            "timedelta": datetime.timedelta,
            "any": object,
        }
        return mapping.get(typename)

class LinkEnd(NamedElement):
    """ A link end is an end point of a link.

    Args:
        name (str): the name of the LinkEnd
        association_end (Property): the end represeted by the LinkEnd
        object (Object): the object pointed to by the LinkEnd
        owner (Link): the Link that owns this LinkEnd
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the LinkEnd
        association_end (Property): the end of the link
        object (Object): the object pointed to by the LinkEnd
        owner (Link): the Link that owns this LinkEnd
    """

    def __init__(self, name:str, association_end: Property, object: Object, owner: "Link" = None):
        super().__init__(name)
        self.association_end: Property = association_end
        self.object: Object = object
        self._owner = None
        self.owner: "Link" = owner

    @property
    def association_end(self):
        """Property: Method to retrieve the association end"""
        return self.__association_end

    @association_end.setter
    def association_end(self, association_end: Property):
        """Property: Method to set the association end"""
        self.__association_end = association_end

    @property
    def object(self):
        """Object: Method to retrieve the object"""
        return self.__object

    @object.setter
    def object(self, object: Object):
        """Object: Method to set the object"""
        self.__object = object
    
    @property
    def owner(self):
        """Link: Method to retrieve the owner link"""
        return self._owner
    
    @owner.setter
    def owner(self, owner: "Link"):
        """Link: Method to set the owner link"""
        if self._owner is not None:
            self._owner.connections.remove(self)
        self._owner = owner

    def __repr__(self):
        return f'LinkEnd({self.name}, end_name={self.association_end.name}, object_linked={self.object.name})'

class Link(NamedElement):
    """ A link represent a relationship between objects.

    Args:
        name (str): the name of the Link
        association (Association): the Association that represents the Link
        connections: list of link ends.
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the Link
        association (Association): the Association that represents the Link
        connections: list of link ends.
    """

    def __init__(self, name: str, association: Association, connections: list[LinkEnd]):
        super().__init__(name)
        self.association: Association = association
        self.connections: list[LinkEnd] = connections
    
    @property
    def association(self):
        """Association: Method to retrieve the association"""
        return self.__association

    @association.setter
    def association(self, association: Association):
        """Association: Method to set the association"""
        self.__association = association

    @property
    def connections(self):
        """list[LinkEnd]: Method to retrieve the connections"""
        return self.__connections

    @connections.setter
    def connections(self, connections: list[LinkEnd]):
        """list[LinkEnd]: Method to set the connections"""
        old_conns = getattr(self, "_Link__connections", [])
        for conn in old_conns:
             conn.object._delete_link(link=self)
        for end in connections:
            end.object._add_link(link=self)
            end.owner = self
        self.__connections = connections

    def add_to_connection(self, linkEnd):
        """Method to add linkend"""
        linkEnd.owner = self
        self.connections.append(linkEnd)

    def __repr__(self):
        return f'Link({self.name}, {self.association.name}, {self.connections})'

class ObjectModel(NamedElement):
    """ An object model is the root element that comprises a number of objects.

    Args:
        name (str): the name of the object model
        objects (set[Object]): set of objects in the model
    
    Attributes:
        name (str): inherited from NamedElement, represents the name of the model
        objects (set[Object]): set of objects in the model
    """

    def __init__(self, name: str, objects: set[Object] = None):
        super().__init__(name)
        self.objects: set[Object] = objects if objects is not None else set()

    @property
    def instances(self) -> set[Union[Object, DataValue]]:
        """set[Union[Object, DataValue]: Method to retrieve the intances (Objects + DataValues)."""
        all_instances = set(self.__objects)
        for obj in self.__objects:
            for slot in obj.slots:
                if isinstance(slot.value, DataValue):
                    all_instances.add(slot.value)
        return all_instances

    @property
    def links(self) -> set[Link]:
        """set[Link]: Method to retrieve the links."""
        all_links = set()
        for obj in self.__objects:
            all_links.update(obj.links)
        return all_links

    @property
    def objects(self) -> set[Object]:
        """set[Object]: Method to retrieve the objects."""
        return self.__objects

    @objects.setter
    def objects(self, objects: set[Object]):
        """set[Object]: Method to set the objects"""
        self.__objects = objects

    def add_object(self, obj: Object):
        """Object: Method to add an object to the set of objects."""
        self.__objects.add(obj)

    def validate(self, raise_exception: bool = True) -> dict:
        """
        Validate the object model according to the UML object diagram constraints.

        Args:
            raise_exception (bool): If True, raise ValueError when validation fails.

        Returns:
            dict: Validation result with success flag, errors, and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        self._validate_unique_object_names(errors)
        self._validate_links(errors)
        self._validate_multiplicities(errors)

        result = {"success": len(errors) == 0, "errors": errors, "warnings": warnings}
        if errors and raise_exception:
            raise ValueError("\n".join(errors))
        return result

    def _validate_unique_object_names(self, errors: list[str]):
        """Ensure each object has a unique name inside the model."""
        seen_names: dict[str, Object] = {}
        for obj in self.__objects:
            # Use name_ property to get the object's identifier, not any "name" attribute
            obj_name = obj.name_
            if obj_name in seen_names:
                errors.append(
                    f"Duplicate object name '{obj_name}' found in object model '{self.name}'."
                )
            else:
                seen_names[obj_name] = obj

    def _validate_links(self, errors: list[str]):
        """Validate that each link is well-formed and typed correctly."""
        for link in self.links:
            association = link.association
            if association is None:
                errors.append(f"Link '{link.name}' is missing an association.")
                continue

            if len(link.connections) != len(association.ends):
                errors.append(
                    f"Link '{link.name}' must instantiate all ends of association '{association.name}' "
                    f"(expected {len(association.ends)}, got {len(link.connections)})."
                )

            seen_ends: set[Property] = set()
            for conn in link.connections:
                if conn.association_end not in association.ends:
                    errors.append(
                        f"Link '{link.name}' references end '{conn.association_end.name}' "
                        f"that does not belong to association '{association.name}'."
                    )
                    continue

                if conn.association_end in seen_ends:
                    errors.append(
                        f"Link '{link.name}' has multiple instances of association end '{conn.association_end.name}'."
                    )
                seen_ends.add(conn.association_end)

                if conn.object not in self.__objects:
                    errors.append(
                        f"Link '{link.name}' references object '{conn.object.name}' "
                        f"which is not part of object model '{self.name}'."
                    )
                elif not self._conforms_to_type(conn.object.classifier, conn.association_end.type):
                    errors.append(
                        f"Object '{conn.object.name}' does not conform to type '{conn.association_end.type.name}' "
                        f"required by association end '{conn.association_end.name}'."
                    )

    def _validate_multiplicities(self, errors: list[str]):
        """Verify that each object satisfies multiplicity constraints for every navigable association end."""
        for obj in self.__objects:
            if not hasattr(obj.classifier, "all_association_ends"):
                continue
            # Count connections grouped by association end
            end_counts: dict[Property, int] = {}
            for link_end in obj.link_ends():
                end_counts[link_end.association_end] = end_counts.get(link_end.association_end, 0) + 1

            for assoc_end in obj.classifier.all_association_ends():
                current = end_counts.get(assoc_end, 0)
                min_mult = assoc_end.multiplicity.min
                max_mult = assoc_end.multiplicity.max
                if current < min_mult or current > max_mult:
                    errors.append(
                        f"Object '{obj.name}' violates multiplicity "
                        f"{self._format_multiplicity(assoc_end)} for association end '{assoc_end.name}' "
                        f"of association '{assoc_end.owner.name}' (found {current} link"
                        f"{'s' if current != 1 else ''})."
                    )

    @staticmethod
    def _conforms_to_type(classifier: Type, expected_type: Type) -> bool:
        """Return True if classifier matches or specializes expected_type."""
        if classifier == expected_type:
            return True
        if hasattr(classifier, "all_parents") and expected_type in classifier.all_parents():
            return True
        return False

    @staticmethod
    def _format_multiplicity(assoc_end: Property) -> str:
        """Render multiplicity bounds readable for error messages."""
        max_value = assoc_end.multiplicity.max
        max_repr = "*" if max_value == UNLIMITED_MAX_MULTIPLICITY else max_value
        return f"{assoc_end.multiplicity.min}..{max_repr}"

    def __repr__(self):
        return f'ObjectModel({self.name}, {self.objects})'
