"""Shell tools are off unless a caller asks for them.

``LLMOrchestrator`` defaulted ``allow_shell_tools=True`` and
``enable_toolchain_validation=True``, and ``LLMGenerator`` never passed either
argument. The hosted backend explicitly gates shell tools off
(``BESSER_LLM_ENABLE_SHELL_TOOLS``, default False) because ``run_command`` and
``install_dependencies`` execute arbitrary commands in the workspace -- so the
library path silently handed out exactly the capability the hosted gate exists
to withhold (2026-09-14).

Both defaults are now off, and both are exposed on ``LLMGenerator`` so a local
run can opt in deliberately. The hosted runner passes its own values and must
keep getting what it asks for.
"""

import inspect

import pytest

from besser.generators.llm.llm_generator import LLMGenerator
from besser.generators.llm.orchestrator import LLMOrchestrator


def _default(func, name):
    return inspect.signature(func).parameters[name].default


# ------------------------------------------------------- the defaults


def test_the_orchestrator_withholds_shell_tools_by_default():
    assert _default(LLMOrchestrator.__init__, "allow_shell_tools") is False


def test_the_orchestrator_does_not_shell_out_to_compilers_by_default():
    assert _default(LLMOrchestrator.__init__, "enable_toolchain_validation") is False


def test_the_generator_withholds_shell_tools_by_default():
    assert _default(LLMGenerator.__init__, "allow_shell_tools") is False


def test_the_generator_does_not_shell_out_to_compilers_by_default():
    assert _default(LLMGenerator.__init__, "enable_toolchain_validation") is False


# ------------------------------------ the generator must THREAD them through


def test_the_generator_actually_passes_the_flags_on():
    """The bug was not the default -- it was that the argument never reached
    the orchestrator at all, so the orchestrator's own default decided."""
    source = inspect.getsource(LLMGenerator.generate)
    assert "allow_shell_tools=" in source, (
        "LLMGenerator.generate must pass allow_shell_tools to LLMOrchestrator; "
        "without it the orchestrator default silently decides"
    )
    assert "enable_toolchain_validation=" in source


@pytest.mark.parametrize("flag", ["allow_shell_tools", "enable_toolchain_validation"])
def test_an_explicit_opt_in_is_preserved(flag, tmp_path):
    """Turning them on must still work -- this is a default change, not a removal."""
    from besser.BUML.metamodel.structural import (
        Class, DomainModel, IntegerType, Property,
    )
    cls = Class(name="Thing", attributes={Property(name="id", type=IntegerType, is_id=True)})
    model = DomainModel(name="M", types={cls})

    gen = LLMGenerator(
        model=model, instructions="build it", api_key="test-key-not-used",
        output_dir=str(tmp_path), **{flag: True},
    )
    assert getattr(gen, flag) is True


def test_the_hosted_runner_still_gates_shell_tools_off():
    """The deployment's own default must not have moved."""
    from besser.utilities.web_modeling_editor.backend.constants.constants import (
        LLM_ENABLE_SHELL_TOOLS,
    )
    assert LLM_ENABLE_SHELL_TOOLS is False


def test_the_hosted_runner_passes_its_own_value_explicitly():
    """It must not rely on the orchestrator default, in either direction."""
    import besser.utilities.web_modeling_editor.backend.services.spec_driven.runner as runner
    source = inspect.getsource(runner)
    assert "allow_shell_tools=LLM_ENABLE_SHELL_TOOLS" in source
