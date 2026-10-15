"""Turning free text (labels, IRIs' local names) into identifiers.

One core, :func:`sanitize_identifier`, behind the three conventions BESSER uses.
They differ in policy, not in mechanism, so each is a thin wrapper that fixes the
options:

* :func:`besser.utilities.buml_code_builder.common.safe_var_name` -- names in
  generated Python code; keeps any Unicode letter, tidies underscores, lowercases.
* ``besser.BUML.notations.kg_to_buml.owl2uml.naming.sanitize`` -- B-UML element names
  from ontologies; **ASCII only**, underscores left alone (auxiliary class names such
  as ``_some_p_D`` depend on their leading one).
* ``besser.BUML.notations.kg_to_buml._common.sanitize_python_identifier`` -- the same
  ASCII rule with tidying, keyword escaping and uniqueness.

ASCII is a requirement for anything that reaches OCL, not a preference: the B-OCL
identifier token is ``[a-zA-Z_][a-zA-Z0-9_]*``, so a class or property called
"Café" would make ``self.café`` unparseable and the invariant would be dropped.
"""

import keyword
import string

__all__ = ["sanitize_identifier"]

_ASCII_IDENTIFIER_CHARS = frozenset(string.ascii_letters + string.digits + "_")


def sanitize_identifier(
    name: str,
    *,
    ascii_only: bool = False,
    collapse_underscores: bool = False,
    strip_underscores: bool = False,
    lowercase: bool = False,
    digit_prefix: str = "_",
    empty: str = "_",
    escape_keywords: bool = False,
) -> str:
    """Turn ``name`` into an identifier.

    Steps, in this order: replace every character that cannot be part of an
    identifier with ``_``; prefix a leading digit with ``digit_prefix``; collapse
    runs of ``_``; strip leading and trailing ``_``; lowercase; append ``_`` to a
    Python keyword. Each of the last four is optional.

    Args:
        name: The text to convert.
        ascii_only: Keep only ``[A-Za-z0-9_]``. Otherwise any character
            ``str.isalnum()`` accepts is kept.
        collapse_underscores: Collapse ``__`` runs to one ``_``.
        strip_underscores: Remove leading and trailing ``_`` (``empty`` if nothing is left).
        lowercase: Lowercase the result.
        digit_prefix: What goes in front of a leading digit.
        empty: Returned for an empty ``name``, and for a name that is nothing but
            underscores when ``strip_underscores`` is set.
        escape_keywords: Append ``_`` to a Python keyword.
    """
    if not name:
        return empty
    if ascii_only:
        result = "".join(c if c in _ASCII_IDENTIFIER_CHARS else "_" for c in name)
    else:
        result = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if result[0].isdigit():
        result = digit_prefix + result
    if collapse_underscores:
        while "__" in result:
            result = result.replace("__", "_")
    if strip_underscores:
        result = result.strip("_") or empty
    if lowercase:
        result = result.lower()
    if escape_keywords and keyword.iskeyword(result):
        result += "_"
    return result
