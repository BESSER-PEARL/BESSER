"""Id helpers shared by the KG importer, converters and code builder."""

from __future__ import annotations

import hashlib

__all__ = ["short_digest"]


def short_digest(raw: str, length: int, *, algo: str = "sha1") -> str:
    """Hex digest of ``raw``, truncated to ``length`` characters.

    The shared tail of every id/signature helper in the KG code. What goes into
    ``raw`` (separators, sort order) and the ``algo``/``length`` are deliberately
    left to each caller: those ids are echoed to the editor and back (issue ids,
    ``kgSignature``), so changing one invalidates a round-trip.

    Encoding uses ``errors="replace"`` so a lone surrogate in a client-supplied
    id degrades to ``?`` instead of raising ``UnicodeEncodeError`` (a 500).
    """
    return hashlib.new(algo, raw.encode("utf-8", errors="replace")).hexdigest()[:length]
