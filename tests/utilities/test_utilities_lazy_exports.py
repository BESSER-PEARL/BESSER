"""``besser.utilities`` loads its names on first use.

It used to star-import its submodules, so importing any one of them (even
``besser.utilities.path_utils``) imported every model-to-code builder and the
image converter, and with them most of the metamodel: the source of several
"import here to avoid a cycle" workarounds. Every name it exported still
resolves to the same object.
"""
import importlib
import subprocess
import sys

import pytest

import besser.utilities as utilities

EXPORTED_FROM = {
    "sort_by_timestamp": "besser.utilities.utils",
    "ModelSerializer": "besser.utilities.utils",
    "timestamp_to_iso": "besser.utilities.utils",
    "explicit_timestamp_iso": "besser.utilities.utils",
    "parse_iso_timestamp": "besser.utilities.utils",
    "DomainModel": "besser.BUML.metamodel.structural",
    "image_to_buml": "besser.utilities.image_to_buml",
    "image_to_plantuml": "besser.utilities.image_to_buml",
    "plantuml_to_buml": "besser.BUML.notations.structuralPlantUML",
    "post_openai_chat": "besser.utilities.llm_utils",
    "LLMUpstreamError": "besser.utilities.llm_utils",
    "DEFAULT_OPENAI_MODEL": "besser.utilities.image_to_buml",
    "UNLIMITED_MULTIPLICITY": "besser.utilities.image_to_buml",
    "domain_model_to_code": "besser.utilities.buml_code_builder.domain_model_builder",
    "agent_model_to_code": "besser.utilities.buml_code_builder.agent_model_builder",
    "gui_model_to_code": "besser.utilities.buml_code_builder.gui_model_builder",
    "project_to_code": "besser.utilities.buml_code_builder.project_builder",
    "bpmn_model_to_code": "besser.utilities.buml_code_builder.bpmn_model_builder",
    "component_model_to_code": "besser.utilities.buml_code_builder.component_model_builder",
    "deployment_model_to_code": "besser.utilities.buml_code_builder.deployment_model_builder",
    "kg_model_to_code": "besser.utilities.buml_code_builder.kg_model_builder",
    "nn_model_to_code": "besser.utilities.buml_code_builder.nn_model_builder",
    "state_machine_to_code": "besser.utilities.buml_code_builder.state_machine_builder",
}


@pytest.mark.parametrize("name, module", sorted(EXPORTED_FROM.items()))
def test_every_exported_name_still_resolves(name, module):
    assert getattr(utilities, name) is getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("name", ["utils", "identifiers", "llm_utils", "provenance", "subprocess_env",
                                  "buml_code_builder", "web_modeling_editor"])
def test_submodules_resolve_as_attributes(name):
    assert getattr(utilities, name) is importlib.import_module(f"besser.utilities.{name}")


def test_from_import_star_and_kg_names_still_work():
    namespace = {}
    exec("from besser.utilities import *", namespace)
    # The old star import yielded kg_llm_to_buml's __all__; those names remain.
    from besser.utilities import kg_llm_to_buml
    assert set(kg_llm_to_buml.__all__) <= set(namespace)
    assert namespace["sort_by_timestamp"] is utilities.sort_by_timestamp
    assert utilities.kg_to_buml is kg_llm_to_buml.kg_to_buml
    assert not hasattr(utilities, "no_such_name")


def test_importing_one_submodule_does_not_load_the_builders():
    # A fresh interpreter: this test process has long since imported everything.
    code = ("import sys, besser.utilities.path_utils; "
            "print(sorted(m for m in sys.modules if m.startswith(("
            "'besser.utilities.buml_code_builder', 'besser.utilities.image_to_buml', "
            "'besser.BUML.metamodel.structural'))))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
