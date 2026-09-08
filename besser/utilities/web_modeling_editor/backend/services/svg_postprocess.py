"""Post-processing for SVG diagrams returned by the headless render service.

The WME Node renderer (Apollon + ELK) sizes the root ``<svg>`` to the diagram's
node/edge geometry, but it does **not** account for association-end *labels*
(role names and multiplicities) that are drawn outside that box. For a
self-referential association the loop is routed on the node's left edge and its
labels are right-anchored, so they extend into negative ``x`` and get clipped
when the SVG is used as a raster image (``<img>``), e.g. embedded in a README.

``fit_svg_viewbox_to_content`` recomputes the true content bounds (node boxes
plus estimated label extents) and, when anything overflows, rewrites the root
``width``/``height`` and adds a matching ``viewBox`` so nothing is cut off. When
the content already fits, the SVG is returned unchanged.
"""

import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# The renderer draws labels in a 16px Helvetica-family font. We can't measure
# glyphs server-side, so we approximate the advance width per character. The
# value is deliberately generous: over-estimating only pads the canvas, whereas
# under-estimating would clip a label again.
_AVG_CHAR_WIDTH = 9.0
_FONT_SIZE = 16.0
_PADDING = 8.0
# Ignore sub-pixel overflow so SVGs that already fit are returned untouched.
_TOLERANCE = 1.0


def _to_float(value, default=None):
    """Parse an SVG length as a float, returning ``default`` for None/percent/invalid."""
    if value is None:
        return default
    value = value.strip()
    if not value or value.endswith("%"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_translate(transform):
    """Extract the (tx, ty) of a ``translate(...)`` transform, or (0, 0)."""
    if not transform:
        return 0.0, 0.0
    match = re.search(r"translate\(\s*([-\d.]+)[ ,]+([-\d.]+)\s*\)", transform)
    if not match:
        match = re.search(r"translate\(\s*([-\d.]+)\s*\)", transform)
        if match:
            return float(match.group(1)), 0.0
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def _local(tag):
    """Strip the XML namespace from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _text_extent(element, ox, oy):
    """Bounding box (x0, y0, x1, y1) of a ``<text>`` element, or None if unpositionable."""
    x = _to_float(element.get("x"))
    if x is None:  # percentage-positioned text lives inside a node box already
        return None
    ax = ox + x + _to_float(element.get("dx"), 0.0)
    text = "".join(element.itertext())
    width = len(text) * _AVG_CHAR_WIDTH
    anchor = element.get("text-anchor", "start")
    if anchor == "end":
        x0, x1 = ax - width, ax
    elif anchor == "middle":
        x0, x1 = ax - width / 2, ax + width / 2
    else:
        x0, x1 = ax, ax + width

    y = _to_float(element.get("y"))
    if y is None:
        y0, y1 = oy - _FONT_SIZE, oy + _FONT_SIZE
    else:
        ay = oy + y + _to_float(element.get("dy"), 0.0)
        y0, y1 = ay - _FONT_SIZE, ay + _FONT_SIZE
    return x0, y0, x1, y1


def _collect_bounds(element, ox, oy, boxes):
    """Walk the tree, accumulating absolute bounding boxes of nodes and labels."""
    tx, ty = _parse_translate(element.get("transform"))
    ox += tx
    oy += ty

    tag = _local(element.tag)
    child_ox, child_oy = ox, oy
    if tag == "svg":
        # A nested <svg x y> establishes a new coordinate origin for its children.
        cx = ox + _to_float(element.get("x"), 0.0)
        cy = oy + _to_float(element.get("y"), 0.0)
        width = _to_float(element.get("width"))
        height = _to_float(element.get("height"))
        if width is not None and height is not None:
            boxes.append((cx, cy, cx + width, cy + height))
        child_ox, child_oy = cx, cy
    elif tag == "text":
        extent = _text_extent(element, ox, oy)
        if extent is not None:
            boxes.append(extent)

    for child in element:
        _collect_bounds(child, child_ox, child_oy, boxes)


def fit_svg_viewbox_to_content(svg: str) -> str:
    """Expand the root ``<svg>`` viewport so overflowing labels are not clipped.

    Returns the SVG unchanged when its content already fits or when the markup
    can't be parsed (best-effort safety net — never raises).
    """
    if not svg or "<svg" not in svg:
        return svg

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        logger.warning("SVG post-process skipped (parse error): %s", exc)
        return svg

    root_w = _to_float(root.get("width"))
    root_h = _to_float(root.get("height"))
    if root_w is None or root_h is None:
        return svg  # nothing reliable to anchor the current viewport to

    boxes = [(0.0, 0.0, root_w, root_h)]
    for child in root:
        _collect_bounds(child, 0.0, 0.0, boxes)

    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)

    overflow = (
        min_x < -_TOLERANCE
        or min_y < -_TOLERANCE
        or max_x > root_w + _TOLERANCE
        or max_y > root_h + _TOLERANCE
    )
    if not overflow:
        return svg

    min_x -= _PADDING
    min_y -= _PADDING
    max_x += _PADDING
    max_y += _PADDING
    new_w = round(max_x - min_x, 2)
    new_h = round(max_y - min_y, 2)
    view_box = f"{round(min_x, 2)} {round(min_y, 2)} {new_w} {new_h}"

    # Rewrite only the root opening tag; leave the body byte-for-byte intact.
    end = svg.index(">") + 1
    head, body = svg[:end], svg[end:]
    head = re.sub(r'\swidth="[^"]*"', f' width="{new_w}"', head, count=1)
    head = re.sub(r'\sheight="[^"]*"', f' height="{new_h}"', head, count=1)
    if "viewBox" in head:
        head = re.sub(r'\sviewBox="[^"]*"', f' viewBox="{view_box}"', head, count=1)
    else:
        head = head[:-1] + f' viewBox="{view_box}">'
    return head + body
