"""Jinja filters for the Platform generator.

These helpers translate `PlatformCustomization` enum values into the strings
the generated React/xyflow code expects, so each piece of conversion logic
lives in exactly one place.
"""

from typing import Optional

from besser.BUML.metamodel.platform_customization import ArrowStyle, LineStyle


# CSS `stroke-dasharray` patterns. None means "default solid line"; an explicit
# "0" forces solid even when something upstream guesses otherwise.
_DASH_PATTERNS = {
    LineStyle.SOLID: "0",
    LineStyle.DASHED: "8 4",
    LineStyle.DOTTED: "2 4",
}


def dash_for(line_style) -> Optional[str]:
    """Return the SVG `stroke-dasharray` for a `LineStyle` value, or None."""
    if line_style is None:
        return None
    if isinstance(line_style, str):
        try:
            line_style = LineStyle(line_style)
        except ValueError:
            return None
    return _DASH_PATTERNS.get(line_style)


# Mapping from our ArrowStyle vocabulary to the marker descriptor consumed by
# the generated React code. xyflow ships only Arrow / ArrowClosed; the other
# entries reference custom <marker> defs injected once into InstanceCanvas.
_MARKER_DESCRIPTORS = {
    ArrowStyle.NONE: None,
    ArrowStyle.FILLED_TRIANGLE: {"type": "arrowclosed"},
    ArrowStyle.OPEN_TRIANGLE: {"type": "arrow"},
    ArrowStyle.DIAMOND: {"type": "url", "id": "platform-diamond"},
    ArrowStyle.OPEN_DIAMOND: {"type": "url", "id": "platform-open-diamond"},
    ArrowStyle.CIRCLE: {"type": "url", "id": "platform-circle"},
}


def marker_for(arrow_style):
    """Return the marker descriptor for an `ArrowStyle`, or None if 'none'."""
    if arrow_style is None:
        return None
    if isinstance(arrow_style, str):
        try:
            arrow_style = ArrowStyle(arrow_style)
        except ValueError:
            return None
    return _MARKER_DESCRIPTORS.get(arrow_style)


def enum_value(value) -> Optional[str]:
    """Return the .value of an enum, or the raw value if it's already a string."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)
