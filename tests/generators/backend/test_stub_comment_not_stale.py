"""The 501 stub's rationale must not survive replacing the raise it explains.

router_methods.py.j2 used to emit, above every unimplemented method's 501, a
comment carrying both the method's identity AND the reason for the 501:

    # {{ class.name }}.{{ clean_name }}: no body in the model - be honest:
    501, never a fake "executed" success.
    raise HTTPException(status_code=501, detail="...")

When a Spec-Driven Agent run implements the method it naturally replaces the
`raise` and leaves the comment above it untouched -- a self-contradicting
"no body in the model" claim sitting above real, working code. Measured in
9-13 of 15 runs, 1-6 stale instances per affected file (zero functional
impact, but it misleads the next pass, including the requirements judge,
which reads the file as text).

Fix: the rationale now lives only in detail=, textually inside the statement
that gets replaced, so it disappears along with it. The comment that CAN
survive an edit is identity-only (`# Class.method`, no implementation-status
claim), which is still useful (it is what keeps a body-less stub quotable by
modify_file without "old_text occurs N times" -- see
test_method_stubs_are_distinguishable.py) but is not a lie if left behind.

action_inventory.py's `_placeholder_reason` is the static stub detector this
must not blind: it is an AST check keyed on `status_code=501` (a constant),
not on comment text, but it is exercised here directly rather than assumed.
"""
import ast
import os

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, PrimitiveDataType, Property,
)
from besser.generators.backend import BackendGenerator
from besser.generators.backend import api_generator as _api_generator_module
from besser.spec_driven_agent.planning.action_inventory import _placeholder_reason

TEMPLATE_PATH = os.path.join(
    os.path.dirname(_api_generator_module.__file__), "templates", "router_methods.py.j2",
)

_STALE_PHRASES = ("no body in the model", "be honest")


def test_no_stale_rationale_anywhere_in_the_template():
    """Covers all 3 raise sites in the template source directly (two of the
    three are presently unreachable dead branches -- see the module note in
    router_methods.py.j2 -- so a generated-output check alone would miss
    them; this catches a partial revert of any of the three)."""
    source = open(TEMPLATE_PATH, encoding="utf-8").read()
    for phrase in _STALE_PHRASES:
        assert phrase not in source, f"stale rationale phrase {phrase!r} still in the template"


def _two_stub_model():
    booking = Class(name="Booking")
    booking.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    booking.methods = {Method(name="produceBill"), Method(name="cancel")}
    return DomainModel(name="Hotel", types={booking})


def test_generated_stub_has_no_stale_comment_and_the_detector_still_fires(tmp_path):
    BackendGenerator(model=_two_stub_model(), output_dir=str(tmp_path)).generate()
    path = os.path.join(str(tmp_path), "routers", "booking_methods.py")
    content = open(path, encoding="utf-8").read()

    for phrase in _STALE_PHRASES:
        assert phrase not in content, f"stale rationale phrase {phrase!r} in generated output"

    tree = ast.parse(content)
    stub_functions = [
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("execute_booking_")
    ]
    assert len(stub_functions) == 2, [n.name for n in stub_functions]
    for fn in stub_functions:
        assert _placeholder_reason(fn) == "HTTP 501", (
            f"{fn.name}: static stub detector no longer recognises this as unimplemented"
        )


def test_implementing_the_stub_leaves_no_misleading_comment(tmp_path):
    """The actual reported failure mode: an editor deletes ONLY the raise
    block (the literal edit a modify_file/replace_file_lines call makes) and
    leaves every comment above it untouched. Simulated directly rather than
    asserted about, since that is exactly what the fix must survive."""
    BackendGenerator(model=_two_stub_model(), output_dir=str(tmp_path)).generate()
    path = os.path.join(str(tmp_path), "routers", "booking_methods.py")
    content = open(path, encoding="utf-8").read()

    raise_block = (
        "    raise HTTPException(\n"
        "        status_code=501,\n"
        "        detail=\"Method 'cancel' of Booking is modeled but has no implementation - 501, never a fake success\",\n"
        "    )"
    )
    assert raise_block in content, "the raise block's exact shape changed; update this test's expected text"

    implemented = (
        "    _booking_object.status = 'CANCELLED'\n"
        "    database.commit()\n"
        "    return {'status': 'cancelled'}"
    )
    edited = content.replace(raise_block, implemented)
    assert edited != content

    for phrase in _STALE_PHRASES:
        assert phrase not in edited, (
            f"stale rationale phrase {phrase!r} survives above the implemented method"
        )
    # The identity comment is expected to survive (that's the point -- it is
    # not a lie), and the detector must now correctly say this is NOT a stub.
    assert "# Booking.cancel" in edited
    tree = ast.parse(edited)
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "execute_booking_cancel")
    assert _placeholder_reason(fn) is None, "implemented method still reads as an unimplemented stub"
