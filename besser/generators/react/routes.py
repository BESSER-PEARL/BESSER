"""Routes of the pages of a generated React app."""
import re


def page_route(name) -> str:
    """The route of the page called ``name``.

    ``/`` and the name in lower case, each run of characters that are not
    letters, digits or ``_`` as one ``-``: ``Q&A / Help`` -> ``/q-a-help``. A
    name with none of those gives ``/``.
    """
    return "/" + re.sub(r"[^\w]+", "-", str(name or "").lower()).strip("-")
