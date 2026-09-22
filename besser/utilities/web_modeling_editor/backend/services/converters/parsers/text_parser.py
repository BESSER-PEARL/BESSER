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
    text = text.replace("'", "\\'")
    # Escape single quotes for code safety
    return text
