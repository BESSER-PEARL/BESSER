"""
Text parsing utilities for converting JSON to BUML format.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def sanitize_text(text):
    """Normalize unicode and strip control characters from user-provided text.

    The result is stored verbatim in metamodel objects, so no escaping happens
    here: quoting and backslash-escaping belong to code emission
    (``besser.utilities.buml_code_builder.common._escape_python_string``).
    Escaping at this point would be applied again by the code builders and
    grow on every JSON -> BUML -> JSON round trip.
    """
    if not isinstance(text, str):
        return text
    # Normalize unicode representations
    text = unicodedata.normalize("NFKD", text)
    # Strip control characters but keep Unicode letters and symbols
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
