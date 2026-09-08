"""Tests for the SVG viewport safety net (``fit_svg_viewbox_to_content``).

The headless renderer sizes the canvas to node/edge geometry but omits
association-end labels, so a self-referential association's left-anchored labels
get clipped. The post-processor must expand the root viewport to include them,
while leaving already-fitting diagrams untouched.
"""

import re

from besser.utilities.web_modeling_editor.backend.services.svg_postprocess import (
    fit_svg_viewbox_to_content,
)

# A node at x=25..185, plus a self-loop group at x=15 whose "manager" label is
# right-anchored and therefore extends to the left, past x=0.
_SELF_LOOP_SVG = (
    '<svg width="201" height="101" xmlns="http://www.w3.org/2000/svg">'
    '<svg x="25" y="15" width="160" height="70" class="Employee"><g></g></svg>'
    '<svg x="15" y="38" width="10" height="23" class="manages"><g>'
    '<text x="2" y="0" dy="-10" text-anchor="end">manager</text>'
    '</g></svg>'
    '</svg>'
)

_FITTING_SVG = (
    '<svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">'
    '<svg x="20" y="15" width="160" height="70" class="A"><g></g></svg>'
    '</svg>'
)


def _viewbox(svg):
    match = re.search(r'viewBox="([^"]+)"', svg)
    return [float(v) for v in match.group(1).split()] if match else None


def test_left_overflowing_label_expands_viewbox():
    out = fit_svg_viewbox_to_content(_SELF_LOOP_SVG)
    vb = _viewbox(out)
    assert vb is not None
    min_x, _min_y, width, _height = vb
    # The clipped "manager" label starts well left of 0; the viewBox must too.
    assert min_x < 0
    # And the canvas must widen to cover it.
    assert width > 201


def test_content_within_bounds_is_unchanged():
    assert fit_svg_viewbox_to_content(_FITTING_SVG) == _FITTING_SVG


def test_malformed_input_returned_unchanged():
    assert fit_svg_viewbox_to_content("not an svg") == "not an svg"
    broken = '<svg width="10" height="10"><rect></svg>'  # unbalanced tags
    # Must never raise, even on unparseable markup.
    assert isinstance(fit_svg_viewbox_to_content(broken), str)
