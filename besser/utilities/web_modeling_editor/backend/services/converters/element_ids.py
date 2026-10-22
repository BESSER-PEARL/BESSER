"""Element ids: which ids the editor can use."""

import re

# Editor ids are 36-character uuids; the cap stops a hostile id from bloating every reference to it.
MAX_ELEMENT_ID_LENGTH = 128

# The editor keys plain objects by id and the GUI parser splits a field reference at '.', so
# only these characters are kept ('__proto__' matches but would hit the object prototype).
_ELEMENT_ID_PATTERN = re.compile(rf"[A-Za-z0-9_:-]{{1,{MAX_ELEMENT_ID_LENGTH}}}")


def is_valid_element_id(value) -> bool:
    """True when ``value`` can be used as an editor element id."""
    return isinstance(value, str) and bool(_ELEMENT_ID_PATTERN.fullmatch(value)) and value != "__proto__"
