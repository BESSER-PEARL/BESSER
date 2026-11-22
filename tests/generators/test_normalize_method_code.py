"""Tests for normalize_method_code: user method bodies embedded in generated files."""
from besser.generators.structural_utils import normalize_method_code


def test_mixed_tabs_and_spaces_compile():
    # Real-world editor input: tab-indented lines, a stray "spaces+tab" blank
    # line and one 4-space-indented line in the same body (previously an
    # IndentationError that prevented the generated app from starting).
    code = (
        "def pay_invoice(self):\n"
        "\tif self.paid:\n"
        "\t\treturn False\n"
        "    \t\n"
        "\tself.paid = True\n"
        "    self.booking.status = 'confirmed'\n"
        "\treturn True\n"
    )
    out = normalize_method_code(code, "pay_invoice")
    compile(out, "<test>", "exec")
    assert "\t" not in out


def test_broken_code_becomes_commented_stub():
    out = normalize_method_code("def broken(self:\n  return", "broken")
    compile(out, "<test>", "exec")
    assert "NotImplementedError" in out
    assert "# def broken(self:" in out


def test_valid_code_passes_through_unchanged_semantics():
    code = "def ok(self):\n    return 42\n"
    out = normalize_method_code(code, "ok")
    namespace = {}
    exec(compile(out, "<test>", "exec"), namespace)
    class _Obj:  # noqa: N801 - minimal stand-in instance
        pass
    assert namespace["ok"](_Obj()) == 42


def test_empty_code_is_returned_as_is():
    assert normalize_method_code("", "x") == ""
    assert normalize_method_code(None, "x") == ""
