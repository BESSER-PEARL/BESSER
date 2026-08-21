"""OCL parser error handling.

The raw ANTLR diagnostics ("mismatched input 'then' expecting <EOF>", "missing
')' at '<EOF>'", etc.) expose internal parser state and use token names that
mean little to someone writing an OCL constraint. This module sits on top of
ANTLR's error listener and rewrites the diagnostics into hints phrased in OCL
terms — "the 'then' keyword must follow 'if <condition>'", "missing closing
')'", etc. — so users can diagnose their constraints without reading the
grammar.

Relates to BESSER-PEARL/BESSER#202.
"""

from antlr4.error.ErrorListener import ErrorListener


# Human-friendly token labels. Anything not listed here keeps the ANTLR name.
_TOKEN_LABELS = {
    "<EOF>": "end of expression",
    "ID": "an identifier",
    "NUMBER": "a number",
    "STRING_LITERAL": "a string literal",
    "BOOLEAN_LITERAL": "a boolean literal",
    "DATE": "a date literal",
}


def _label(token_text: str) -> str:
    """Return a human-friendly label for a raw token string."""
    if not token_text:
        return "end of expression"
    # Strip ANTLR's surrounding quotes on keyword tokens (e.g. "'if'" -> "'if'")
    return _TOKEN_LABELS.get(token_text, token_text)


def _friendly(offending: str, antlr_msg: str) -> str | None:
    """Translate a raw ANTLR diagnostic into an OCL-level hint.

    Returns None when no targeted hint applies, so the caller can fall back to
    the raw ANTLR message.
    """
    # 1. Dangling if/then/else keyword pieces.
    if offending == "then" and "expecting <EOF>" in antlr_msg:
        return ("Unexpected 'then'. The 'then' keyword must follow "
                "'if <condition>'. Did you forget the 'if'?")
    if offending == "else" and "expecting <EOF>" in antlr_msg:
        return ("Unexpected 'else' outside of an 'if ... then ... else ... "
                "endif' expression.")
    if offending == "endif" and "expecting <EOF>" in antlr_msg:
        return "Unexpected 'endif' without a matching 'if'."

    # 2. Unterminated if/then/else.
    if "missing 'endif'" in antlr_msg or "expecting 'endif'" in antlr_msg:
        return ("Missing 'endif'. Every 'if ... then ... else ...' expression "
                "must be closed with 'endif'.")
    if offending == "<EOF>" and "expecting 'else'" in antlr_msg:
        return ("Expression ends before the 'else' branch. 'if ... then ...' "
                "requires an 'else ... endif' to complete the expression.")
    if offending == "<EOF>" and "expecting 'then'" in antlr_msg:
        return ("Expression ends before the 'then' branch. 'if <condition>' "
                "must be followed by 'then ... else ... endif'.")

    # 3. Parenthesis balance.
    if "missing ')'" in antlr_msg:
        return "Missing closing ')'."
    if offending == ")" and "expecting <EOF>" in antlr_msg:
        return "Unexpected ')' — there is no matching opening '('."

    # 4. Structural errors at the very top of an OCL file.
    if "expecting 'context'" in antlr_msg:
        return ("OCL constraints must start with 'context <ClassName> inv: "
                "<expression>'.")

    # 5. Incomplete expression trailing an operator.
    if offending == "<EOF>":
        return "Expression ends unexpectedly — the operand on the right is missing."

    # 6. Unknown token, usually a keyword collision or unsupported construct.
    if "no viable alternative" in antlr_msg:
        return (f"Unexpected token near '{offending}' — this construct is not "
                "recognized by B-OCL.")

    return None


class BOCLErrorListener(ErrorListener):
    """Collects syntax errors and rewrites them into OCL-level hints."""

    def __init__(self):
        super().__init__()
        self.errors = []

    # pylint: disable=unused-argument
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        offending_text = ""
        if offendingSymbol is not None and hasattr(offendingSymbol, "text"):
            offending_text = offendingSymbol.text or ""

        hint = _friendly(offending_text, msg)
        location = f"line {line}, column {column}"

        if hint is not None:
            self.errors.append(f"{location}: {hint}")
        else:
            # Fallback: keep the raw ANTLR diagnostic but still surface the
            # location in a consistent format.
            self.errors.append(f"{location}: {msg}")

    def has_errors(self):
        return len(self.errors) > 0


class BOCLSyntaxError(Exception):
    """Raised when OCL parsing fails."""

    def __init__(self, errors):
        self.errors = errors
        if len(errors) == 1:
            summary = errors[0]
        else:
            summary = "Multiple OCL syntax errors:\n  - " + "\n  - ".join(errors)
        super().__init__(summary)
