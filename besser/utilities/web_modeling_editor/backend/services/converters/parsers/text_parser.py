"""
Text parsing utilities for converting JSON to BUML format.
"""

import unicodedata


def sanitize_text(text):
    """Sanitize text by removing special characters and normalizing."""
    if not isinstance(text, str):
        return text
    # Normalize and strip accents or special symbols
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("'", "\\'")
    #text = text.replace("'", " ")
    # Escape single quotes for code safety
    return text
