from .utils import *
from .image_to_buml import *
from .buml_code_builder import *
from .buml_code_builder.domain_model_builder import domain_model_to_code
from .buml_code_builder.agent_model_builder import agent_model_to_code
from .buml_code_builder.gui_model_builder import gui_model_to_code
from .buml_code_builder.project_builder import project_to_code


def __getattr__(name: str):
    """
    Lazy-load kg_to_buml symbols to avoid import-time circular dependencies.
    """
    try:
        from . import kg_to_buml as _kg_to_buml
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise AttributeError(name) from exc
    if hasattr(_kg_to_buml, name):
        return getattr(_kg_to_buml, name)
    raise AttributeError(name)
