"""What a method button tells the user when the call fails."""

CLASSES = {"Order": [("reference", "str")]}


def test_a_method_without_implementation_gets_a_readable_message(build_app):
    """The backend answers 501 with a sentence for developers; the user reads a plain one."""
    app = build_app(CLASSES, {"Orders": [
        {"type": "table", "attributes": {"id": "orders", "data-source": "cls-Order"}}]})
    button = app.file("components", "MethodButton.tsx")

    assert "const notImplemented = statusCode === 501;" in button
    handled = button[button.index("if (notImplemented) {"):]
    assert handled.index('errorMessage = "This action is not available yet.";') < handled.index("err.response?.data")
    # The "Server Error (501): ..." prefix is not added to it.
    assert "if (!notImplemented && statusCode && statusCode >= 500" in button
