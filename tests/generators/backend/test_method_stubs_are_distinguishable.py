"""Every generated 501 stub must name its method.

Measured 2026-09-17 on a five-method class: the stub bodies were byte-identical
for 14 lines, so a modify_file quoting one of them hit "old_text occurs 5 times
and is ambiguous" - the model then either quoted a def line from memory (miss)
or rewrote the whole file. The identity belongs in the comment the model
quotes, not only in the detail string at the end.
"""

import os
from collections import Counter

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Method, PrimitiveDataType, Property,
)
from besser.generators.backend import BackendGenerator


def test_each_501_stub_names_its_method(tmp_path):
    booking = Class(name="Booking")
    booking.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    booking.methods = {Method(name="produceBill"), Method(name="cancel")}
    model = DomainModel(name="Hotel", types={booking})

    BackendGenerator(model=model, output_dir=str(tmp_path)).generate()

    content = open(os.path.join(str(tmp_path), "routers", "booking_methods.py"), encoding="utf-8").read()
    for method in ("produceBill", "cancel"):
        assert f"# Booking.{method}" in content, method
        assert f"Method '{method}' of Booking is modeled but has no implementation" in content, method
    assert "# Method body not defined in the model" not in content
    # 2026-09-19: the rationale used to sit in the standalone comment itself
    # ("...: no body in the model - be honest..."), so an agent that replaced
    # the raise but left the comment shipped a self-contradicting claim above
    # real code. It now lives only in detail=, deleted along with the raise.
    assert "no body in the model" not in content
    assert content.count("status_code=501") == 2       # still honest stubs


# Exception scaffolding carries no method identity and is never an edit target,
# so a repeat of it is not what makes modify_file refuse a real edit.
_SCAFFOLDING = {
    "try:", "except HTTPException:", "except Exception as e:", "raise",
    "sys.stdout = sys.__stdout__", ")", "}", "]",
}


def _ambiguous_windows(content: str, size: int = 4) -> list[str]:
    """Windows of `size` lines that occur more than once and could plausibly be
    quoted as a modify_file anchor."""
    lines = content.split("\n")
    counts: Counter = Counter()
    for i in range(len(lines) - size + 1):
        window = lines[i:i + size]
        body = [ln.strip() for ln in window if ln.strip()]
        if len(body) < 2 or all(ln in _SCAFFOLDING for ln in body):
            continue
        counts["\n".join(window)] += 1
    return [w for w, n in counts.items() if n > 1]


def test_no_four_line_window_of_a_stub_is_ambiguous(tmp_path):
    """Every quotable 4-line window must pin exactly one method, or modify_file
    refuses the edit with "old_text occurs N times and is ambiguous"."""
    booking = Class(name="Booking")
    booking.attributes = {Property(name="id", type=PrimitiveDataType("int"), is_id=True)}
    booking.methods = {Method(name="produceBill"), Method(name="cancel")}
    model = DomainModel(name="Hotel", types={booking})

    BackendGenerator(model=model, output_dir=str(tmp_path)).generate()

    path = os.path.join(str(tmp_path), "routers", "booking_methods.py")
    with open(path, encoding="utf-8") as f:
        content = f.read()

    ambiguous = _ambiguous_windows(content)
    assert ambiguous == [], "\n\n--- also ---\n\n".join(ambiguous)
