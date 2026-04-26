"""Platform Customization metamodel.

Defines overrides applied to a class diagram when generating a platform editor
via `besser.generators.platform.PlatformGenerator`. A customization references
classes and associations *by name* so it survives re-generation and remains
decoupled from BUML object identity.

v2 scope:
    - per-class: container flag, default size, node shape, fill / border color,
      border width / style / radius, font size / weight / color, label position
    - per-association: edge color, line width / style, source / target arrow
      style, label visibility / font size / color
    - diagram-level: background color, grid visibility / size, snap-to-grid,
      theme (light / dark / auto)

All new fields default to None / False so a model with no overrides produces
output identical to v1.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

from besser.BUML.metamodel.structural import NamedElement


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeShape(str, Enum):
    RECTANGLE = "rectangle"
    ROUNDED_RECT = "rounded_rect"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"


class LineStyle(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


class ArrowStyle(str, Enum):
    NONE = "none"
    FILLED_TRIANGLE = "filled_triangle"
    OPEN_TRIANGLE = "open_triangle"
    DIAMOND = "diamond"
    OPEN_DIAMOND = "open_diamond"
    CIRCLE = "circle"


class FontWeight(str, Enum):
    NORMAL = "normal"
    BOLD = "bold"


class LabelPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    INSIDE = "inside"


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


_E = TypeVar("_E", bound=Enum)


def _coerce_enum(enum_cls: Type[_E], value: Any, field_name: str) -> Optional[_E]:
    """Accept either an Enum member or its string value; pass through None."""
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, KeyError) as exc:
        valid = ", ".join(m.value for m in enum_cls)
        raise ValueError(
            f"{field_name} must be one of [{valid}], got {value!r}"
        ) from exc


def _validate_int_range(value: Optional[int], lo: int, hi: int, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int or None")
    if value < lo or value > hi:
        raise ValueError(f"{field_name} must be in [{lo}, {hi}], got {value}")


def _validate_color(value: Optional[str], field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    if value.strip() == "":
        raise ValueError(f"{field_name} cannot be empty; use None to clear")


# ---------------------------------------------------------------------------
# Per-class
# ---------------------------------------------------------------------------


@dataclass
class ClassCustomization:
    """Per-class overrides consumed by the platform generator.

    All fields are optional: None means "fall through to the generator's
    current default look".
    """

    is_container: bool = False
    default_width: Optional[int] = None
    default_height: Optional[int] = None

    node_shape: Optional[NodeShape] = None
    fill_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: Optional[int] = None
    border_style: Optional[LineStyle] = None
    border_radius: Optional[int] = None

    font_size: Optional[int] = None
    font_weight: Optional[FontWeight] = None
    font_color: Optional[str] = None
    label_position: Optional[LabelPosition] = None

    def __post_init__(self):
        if self.default_width is not None and self.default_width <= 0:
            raise ValueError("default_width must be a positive integer")
        if self.default_height is not None and self.default_height <= 0:
            raise ValueError("default_height must be a positive integer")

        self.node_shape = _coerce_enum(NodeShape, self.node_shape, "node_shape")
        self.border_style = _coerce_enum(LineStyle, self.border_style, "border_style")
        self.font_weight = _coerce_enum(FontWeight, self.font_weight, "font_weight")
        self.label_position = _coerce_enum(LabelPosition, self.label_position, "label_position")

        _validate_color(self.fill_color, "fill_color")
        _validate_color(self.border_color, "border_color")
        _validate_color(self.font_color, "font_color")

        _validate_int_range(self.border_width, 0, 8, "border_width")
        _validate_int_range(self.border_radius, 0, 32, "border_radius")
        _validate_int_range(self.font_size, 8, 24, "font_size")


# ---------------------------------------------------------------------------
# Per-association
# ---------------------------------------------------------------------------


@dataclass
class AssociationCustomization:
    """Per-association overrides consumed by the platform generator.

    `is_container_association` flips the runtime behavior: when true *and* the
    source class of the association is a container, dropping a target-class
    instance inside the container instance auto-creates this association and
    nests the child node visually. When false (default), the association is
    rendered as a normal edge regardless of the source class's container flag.
    """

    edge_color: Optional[str] = None
    line_width: Optional[int] = None
    line_style: Optional[LineStyle] = None
    source_arrow_style: Optional[ArrowStyle] = None
    target_arrow_style: Optional[ArrowStyle] = None
    label_visible: Optional[bool] = None
    label_font_size: Optional[int] = None
    label_font_color: Optional[str] = None
    is_container_association: bool = False

    def __post_init__(self):
        _validate_color(self.edge_color, "edge_color")
        _validate_color(self.label_font_color, "label_font_color")

        _validate_int_range(self.line_width, 1, 6, "line_width")
        _validate_int_range(self.label_font_size, 8, 18, "label_font_size")

        self.line_style = _coerce_enum(LineStyle, self.line_style, "line_style")
        self.source_arrow_style = _coerce_enum(
            ArrowStyle, self.source_arrow_style, "source_arrow_style"
        )
        self.target_arrow_style = _coerce_enum(
            ArrowStyle, self.target_arrow_style, "target_arrow_style"
        )

        if self.label_visible is not None and not isinstance(self.label_visible, bool):
            raise ValueError("label_visible must be a bool or None")
        if not isinstance(self.is_container_association, bool):
            raise ValueError("is_container_association must be a bool")


# ---------------------------------------------------------------------------
# Diagram-level
# ---------------------------------------------------------------------------


@dataclass
class DiagramCustomization:
    """Diagram-level overrides applied to the whole generated canvas."""

    background_color: Optional[str] = None
    grid_visible: Optional[bool] = None
    grid_size: Optional[int] = None
    snap_to_grid: Optional[bool] = None
    theme: Optional[Theme] = None

    def __post_init__(self):
        _validate_color(self.background_color, "background_color")
        _validate_int_range(self.grid_size, 8, 64, "grid_size")
        self.theme = _coerce_enum(Theme, self.theme, "theme")
        if self.grid_visible is not None and not isinstance(self.grid_visible, bool):
            raise ValueError("grid_visible must be a bool or None")
        if self.snap_to_grid is not None and not isinstance(self.snap_to_grid, bool):
            raise ValueError("snap_to_grid must be a bool or None")


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class PlatformCustomizationModel(NamedElement):
    """Top-level container for platform generation customizations.

    Pairs with a `DomainModel` (class diagram) at generation time. References
    target classes and associations by their name, so the same customization
    can be reused across class-diagram edits that preserve names.
    """

    def __init__(
        self,
        name: str,
        class_overrides: Optional[Dict[str, ClassCustomization]] = None,
        association_overrides: Optional[Dict[str, AssociationCustomization]] = None,
        diagram_customization: Optional[DiagramCustomization] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(name=name, timestamp=timestamp)
        self.class_overrides = {} if class_overrides is None else class_overrides
        self.association_overrides = (
            {} if association_overrides is None else association_overrides
        )
        self.diagram_customization = diagram_customization

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

    @property
    def diagram_customization(self) -> Optional[DiagramCustomization]:
        return self.__diagram_customization

    @diagram_customization.setter
    def diagram_customization(self, value: Optional[DiagramCustomization]):
        if value is not None and not isinstance(value, DiagramCustomization):
            raise ValueError(
                "diagram_customization must be a DiagramCustomization instance or None"
            )
        self.__diagram_customization = value

    def get_class_customization(self, class_name: str) -> ClassCustomization:
        """Return the customization for `class_name`, or a default if none set."""
        return self.__class_overrides.get(class_name, ClassCustomization())

    def get_association_customization(self, association_name: str) -> AssociationCustomization:
        """Return the customization for `association_name`, or a default if none set."""
        return self.__association_overrides.get(association_name, AssociationCustomization())

    def get_diagram_customization(self) -> DiagramCustomization:
        """Return the diagram-level customization, or a default if none set."""
        return self.__diagram_customization or DiagramCustomization()

    def __repr__(self) -> str:
        return (
            f"PlatformCustomizationModel(name={self.name!r}, "
            f"class_overrides={len(self.__class_overrides)}, "
            f"association_overrides={len(self.__association_overrides)}, "
            f"diagram_customization={'set' if self.__diagram_customization else 'none'})"
        )


__all__ = [
    "ArrowStyle",
    "AssociationCustomization",
    "ClassCustomization",
    "DiagramCustomization",
    "FontWeight",
    "LabelPosition",
    "LineStyle",
    "NodeShape",
    "PlatformCustomizationModel",
    "Theme",
]
