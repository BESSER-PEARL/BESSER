"""BESSER utilities.

The names below are loaded on first use (PEP 562), so importing one
submodule, e.g. ``besser.utilities.path_utils``, does not also import the
model-to-code builders and the image and KG converters.
"""
import importlib
import sys
import types

_EXPORTS = {
    # utils
    "ModelSerializer": "besser.utilities.utils",
    "sort_by_timestamp": "besser.utilities.utils",
    "timestamp_to_iso": "besser.utilities.utils",
    "explicit_timestamp_iso": "besser.utilities.utils",
    "parse_iso_timestamp": "besser.utilities.utils",
    "DomainModel": "besser.utilities.utils",
    # image_to_buml
    "image_to_buml": "besser.utilities.image_to_buml",
    "image_to_plantuml": "besser.utilities.image_to_buml",
    "plantuml_to_buml": "besser.utilities.image_to_buml",
    "post_openai_chat": "besser.utilities.image_to_buml",
    "LLMUpstreamError": "besser.utilities.image_to_buml",
    "DEFAULT_OPENAI_MODEL": "besser.utilities.image_to_buml",
    "UNLIMITED_MULTIPLICITY": "besser.utilities.image_to_buml",
    # buml_code_builder
    "domain_model_to_code": "besser.utilities.buml_code_builder",
    "agent_model_to_code": "besser.utilities.buml_code_builder",
    "gui_model_to_code": "besser.utilities.buml_code_builder",
    "project_to_code": "besser.utilities.buml_code_builder",
    "bpmn_model_to_code": "besser.utilities.buml_code_builder",
    "component_model_to_code": "besser.utilities.buml_code_builder",
    "deployment_model_to_code": "besser.utilities.buml_code_builder",
    "kg_model_to_code": "besser.utilities.buml_code_builder",
    "nn_model_to_code": "besser.utilities.buml_code_builder",
    "state_machine_to_code": "besser.utilities.buml_code_builder",
    # kg_llm_to_buml (its __all__, which ``from besser.utilities import *`` returned)
    **{name: "besser.utilities.kg_llm_to_buml" for name in (
        "LLMAnswerError", "clean_json_response", "clean_plantuml_response",
        "convert_spec_json_to_buml", "kg_to_buml", "kg_to_plantuml",
        "parse_json_safely", "spec_to_domain_model")},
}

# Submodules the package exposed as attributes when it imported them eagerly.
_SUBMODULES = {"utils", "identifiers", "llm_utils", "provenance", "subprocess_env",
               "buml_code_builder", "web_modeling_editor"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(importlib.import_module(module), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        # The LLM KG conversion's symbols (``kg_to_buml``, ``kg_to_plantuml``, ...).
        try:
            kg_llm_to_buml = importlib.import_module(f"{__name__}.kg_llm_to_buml")
        except Exception as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        if not hasattr(kg_llm_to_buml, name):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(kg_llm_to_buml, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


class _UtilitiesModule(types.ModuleType):
    def __setattr__(self, name, value):
        # Importing a submodule binds it on this package; ``image_to_buml`` must
        # stay the function of that name, as the star import used to leave it.
        if name in _EXPORTS and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _UtilitiesModule
