from antlr4.error.ErrorListener import ErrorListener


class BOCLErrorListener(ErrorListener):
    """Collects syntax errors instead of printing to stderr."""

    def __init__(self):
        super().__init__()
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")

    def has_errors(self):
        return len(self.errors) > 0


class BOCLSyntaxError(Exception):
    """Raised when OCL parsing fails."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("OCL syntax errors:\n" + "\n".join(errors))
