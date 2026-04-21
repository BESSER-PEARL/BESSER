"""Platform Customization metamodel.

Defines overrides applied to a class diagram when generating a platform editor
via `besser.generators.platform.PlatformGenerator`. A customization references
classes and associations *by name* so it survives re-generation and remains
decoupled from BUML object identity.

v1 scope:
    - per-class: container flag, default node width/height
    - per-association: edge color (hex, hsl, or any CSS color string)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from besser.BUML.metamodel.structural import NamedElement


@dataclass
class ClassCustomization:
    """Per-class overrides consumed by the platform generator.

    Attributes:
        is_container: If True, instances of this class render as a container
            rectangle in the generated editor; dropping a child instance
            inside auto-creates the containment association.
        default_width: Default node width in pixels. None lets the generator
            choose a sensible default based on is_container.
        default_height: Default node height in pixels. Same default rule.
    """

    is_container: bool = False
    default_width: Optional[int] = None
    default_height: Optional[int] = None

    def __post_init__(self):
        if self.default_width is not None and self.default_width <= 0:
            raise ValueError("default_width must be a positive integer")
        if self.default_height is not None and self.default_height <= 0:
            raise ValueError("default_height must be a positive integer")


@dataclass
class AssociationCustomization:
    """Per-association overrides consumed by the platform generator.

    Attributes:
        edge_color: CSS color string used to stroke edges that represent this
            association. Accepts any form the browser understands (hex, hsl,
            rgb, named colors). None means "use default".
    """

    edge_color: Optional[str] = None

    def __post_init__(self):
        if self.edge_color is not None and not isinstance(self.edge_color, str):
            raise ValueError("edge_color must be a string or None")
        if self.edge_color is not None and self.edge_color.strip() == "":
            raise ValueError("edge_color cannot be empty; use None to clear")


class PlatformCustomizationModel(NamedElement):
    """Top-level container for platform generation customizations.

    Pairs with a `DomainModel` (class diagram) at generation time. References
    target classes and associations by their name, so the same customization
    can be reused across class-diagram edits that preserve names.

    Args:
        name: The customization model's name.
        class_overrides: Mapping of class name to `ClassCustomization`.
        association_overrides: Mapping of association name to
            `AssociationCustomization`.
        timestamp: Creation timestamp (defaults to now).

    Example:
        >>> cust = PlatformCustomizationModel(
        ...     name="RegionSensor",
        ...     class_overrides={
        ...         "Region": ClassCustomization(is_container=True, default_width=400, default_height=300),
        ...         "Sensor": ClassCustomization(default_width=80, default_height=80),
        ...     },
        ...     association_overrides={"has": AssociationCustomization(edge_color="#22c55e")},
        ... )
    """

    def __init__(
        self,
        name: str,
        class_overrides: Optional[Dict[str, ClassCustomization]] = None,
        association_overrides: Optional[Dict[str, AssociationCustomization]] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(name=name, timestamp=timestamp)
        self.class_overrides = {} if class_overrides is None else class_overrides
        self.association_overrides = {} if association_overrides is None else association_overrides

    @property
    def class_overrides(self) -> Dict[str, ClassCustomization]:
        return self.__class_overrides

    @class_overrides.setter
    def class_overrides(self, overrides: Dict[str, ClassCustomization]):
        if not isinstance(overrides, dict):
            raise ValueError("class_overrides must be a dict keyed by class name")
        for cls_name, override in overrides.items():
            if not isinstance(cls_name, str) or cls_name.strip() == "":
                raise ValueError("class_overrides keys must be non-empty class names")
            if not isinstance(override, ClassCustomization):
                raise ValueError(
                    f"class_overrides[{cls_name!r}] must be a ClassCustomization instance"
                )
        self.__class_overrides = overrides

    @property
    def association_overrides(self) -> Dict[str, AssociationCustomization]:
        return self.__association_overrides

    @association_overrides.setter
    def association_overrides(self, overrides: Dict[str, AssociationCustomization]):
        if not isinstance(overrides, dict):
            raise ValueError("association_overrides must be a dict keyed by association name")
        for assoc_name, override in overrides.items():
            if not isinstance(assoc_name, str) or assoc_name.strip() == "":
                raise ValueError("association_overrides keys must be non-empty association names")
            if not isinstance(override, AssociationCustomization):
                raise ValueError(
                    f"association_overrides[{assoc_name!r}] must be an AssociationCustomization instance"
                )
        self.__association_overrides = overrides

    def get_class_customization(self, class_name: str) -> ClassCustomization:
        """Return the customization for `class_name`, or a default if none set."""
        return self.__class_overrides.get(class_name, ClassCustomization())

    def get_association_customization(self, association_name: str) -> AssociationCustomization:
        """Return the customization for `association_name`, or a default if none set."""
        return self.__association_overrides.get(association_name, AssociationCustomization())

    def __repr__(self) -> str:
        return (
            f"PlatformCustomizationModel(name={self.name!r}, "
            f"class_overrides={len(self.__class_overrides)}, "
            f"association_overrides={len(self.__association_overrides)})"
        )


__all__ = [
    "ClassCustomization",
    "AssociationCustomization",
    "PlatformCustomizationModel",
]
