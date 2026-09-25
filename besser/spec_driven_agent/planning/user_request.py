"""The user's request, and the one sanctioned way to shorten it.

Every stage that DECIDES on the request reads it whole. A head-clip is the
worst shape for that text: stack declarations conventionally come LAST
("Additionally, you must use: Frontend -> React"), so a selector that sees
only ``instructions[:500]`` can scaffold a long spec with no frontend.
``tests/spec_driven_agent/test_request_not_clipped.py`` fails on any bound
that is not declared - here, with a stated reason, or with a ``bounded:``
comment on a display/fingerprint slice.
"""


def user_request(
    instructions: str | None, *, excerpt: int | None = None, reason: str = "",
) -> str:
    """The request for a consuming stage - whole unless ``excerpt`` is given.

    ``excerpt`` is for a bounded input with a stated budget and requires
    ``reason``, which the invariant test checks is a literal at every call
    site; it is documentation, not runtime behaviour. A cut request ends
    with a marker so the model knows it continues.
    """
    text = instructions or ""
    if excerpt is None or len(text) <= excerpt:
        return text
    return f"{text[:excerpt]}\n…[request truncated at {excerpt} chars]"
