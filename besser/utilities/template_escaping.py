"""Jinja filters that escape free text for the string literal it is rendered into.

Each filter returns the *inside* of a string literal: the template keeps its own
quotes, so ``'{{ text | py_str }}'`` renders the same bytes as ``'{{ text }}'``
for text with no quote, backslash, line break or interpolation character.
"""

from jinja2 import Environment

# Line breaks and tabs are escaped in every target: a raw newline ends (or breaks)
# a single-line string literal in Python, Turtle, HCL, Dart and YAML.
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape(text, escapes: dict) -> str:
    """Return ``text`` as a string with each character in ``escapes`` replaced."""
    if text is None:
        return ""
    return "".join(escapes.get(char, char) for char in str(text))


def py_str(text) -> str:
    """Escape for a single-quoted Python string literal (not an f-string).

    Double quotes are left as they are, matching the hand-written escaping this replaces.
    """
    return _escape(text, {"\\": "\\\\", "'": "\\'", **_CONTROL_ESCAPES})


def ttl_str(text) -> str:
    """Escape for a double-quoted RDF Turtle string literal (Turtle ``ECHAR``)."""
    return _escape(text, {"\\": "\\\\", '"': '\\"', **_CONTROL_ESCAPES})


def hcl_str(text) -> str:
    """Escape for a double-quoted HCL (Terraform) string, including ``${`` and ``%{`` templates."""
    escaped = _escape(text, {"\\": "\\\\", '"': '\\"', **_CONTROL_ESCAPES})
    return escaped.replace("${", "$${").replace("%{", "%%{")


def dart_str(text) -> str:
    """Escape for a single-quoted Dart string, including ``$`` interpolation."""
    return _escape(text, {"\\": "\\\\", "'": "\\'", "$": "\\$", **_CONTROL_ESCAPES})


def yaml_str(text) -> str:
    """Escape for a double-quoted YAML scalar."""
    return _escape(text, {"\\": "\\\\", '"': '\\"', **_CONTROL_ESCAPES})


ESCAPING_FILTERS = {
    "py_str": py_str,
    "ttl_str": ttl_str,
    "hcl_str": hcl_str,
    "dart_str": dart_str,
    "yaml_str": yaml_str,
}


def register_escaping_filters(env: Environment) -> Environment:
    """Add the string-literal escaping filters to ``env`` and return it."""
    env.filters.update(ESCAPING_FILTERS)
    return env
