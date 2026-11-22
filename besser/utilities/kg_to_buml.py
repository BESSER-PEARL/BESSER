"""Deprecated alias of :mod:`besser.utilities.kg_llm_to_buml`.

The LLM-assisted KG conversion moved so that its name no longer collides with
the deterministic :mod:`besser.BUML.notations.kg_to_buml` package. Import from
``besser.utilities.kg_llm_to_buml`` instead; this module will be removed in a
future release.
"""

import warnings

from besser.utilities.kg_llm_to_buml import *  # noqa: F401,F403
from besser.utilities.kg_llm_to_buml import __all__  # noqa: F401

warnings.warn(
    "besser.utilities.kg_to_buml is deprecated; import from "
    "besser.utilities.kg_llm_to_buml instead.",
    DeprecationWarning,
    stacklevel=2,
)
