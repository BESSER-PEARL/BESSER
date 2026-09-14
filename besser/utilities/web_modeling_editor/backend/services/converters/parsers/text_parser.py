"""
Text parsing utilities for converting JSON to BUML format.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def sanitize_text(text):
    """Sanitize text by removing control characters and normalizing."""
    if not isinstance(text, str):
        return text
    # Normalize unicode representations
    text = unicodedata.normalize("NFKD", text)
    # Strip control characters but keep Unicode letters and symbols
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Escape backslashes first, then single quotes, so a trailing backslash
    # cannot form \\' and re-open a string literal.
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    return text
