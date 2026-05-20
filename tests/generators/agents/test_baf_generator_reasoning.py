"""Integration tests for BAFGenerator with the reasoning extension.

These tests build a fixture agent that uses every reasoning primitive
(``Tool``, ``Skill``, ``Workspace``, ``ReasoningState``), run the
generator, and assert that the produced ``<agent_name>.py`` file:

    * exists and is non-empty;
    * imports the BAF reasoning runtime symbols;
    * registers each primitive via the canonical BAF builder calls;
    * is syntactically valid Python (``ast.parse``).

The tests deliberately avoid importing the BAF runtime — only the text
of the generated file is inspected.
"""

import ast
import os

import pytest

from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    LLMOpenAI,
    WebSocketPlatform,
)
from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode


@pytest.fixture
def reasoning_agent_model() -> Agent:
    """Build an agent with one of each reasoning primitive."""
    agent = Agent("reasoning_demo")
    agent.platforms.append(WebSocketPlatform())

    llm = LLMOpenAI(agent=agent, name="gpt-4o-mini", parameters={})

    agent.new_tool(
        name="ping",
        description="Ping the server.",
        code="def ping():\n    return 'pong'\n",
    )
    agent.new_skill(
        name="GreetByName",
        content="Always greet the user by name when introduced.",
        description="Greeting playbook.",
    )
    agent.new_workspace(
        name="cinema",
        path="/tmp/cinema",
        description="Cinema files.",
        writable=True,
        max_read_bytes=50_000,
    )
    agent.new_reasoning_state(
        name="reason",
        llm="gpt-4o-mini",
        initial=True,
        max_steps=15,
        enable_task_planning=True,
        stream_steps=False,
    )
    return agent


def _generate(agent: Agent, output_dir: str) -> str:
    """Run BAFGenerator and return the path of the generated .py file."""
    BAFGenerator(
        model=agent,
        output_dir=output_dir,
        generation_mode=GenerationMode.CODE_ONLY,
    ).generate()
    return os.path.join(output_dir, f"{agent.name}.py")


class TestBAFGeneratorReasoning:
    """``BAFGenerator.generate()`` emits the reasoning runtime calls."""

    def test_generated_file_exists_and_non_empty(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_generated_file_is_valid_python(self, reasoning_agent_model, tmp_path):
        """The generated agent module parses cleanly."""
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        # Will raise SyntaxError if the template emits broken code.
        ast.parse(code)

    def test_imports_baf_reasoning_state_factory(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        assert "from baf.library.state import new_reasoning_state" in code

    def test_emits_tool_registration(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        # The tool's source must be inlined and then registered.
        assert "def ping" in code
        assert "agent.new_tool(" in code

    def test_emits_skill_registration(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        assert "agent.new_skill(" in code
        assert "GreetByName" in code

    def test_emits_workspace_registration(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        assert "agent.new_workspace(" in code
        assert "cinema" in code

    def test_emits_reasoning_state_factory_call(self, reasoning_agent_model, tmp_path):
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        assert "new_reasoning_state(" in code
        # The reasoning state should reference the agent and the LLM by var.
        assert "agent=agent" in code
        # Either max_steps=15 (the value we set) shows up explicitly.
        assert "max_steps=15" in code

    def test_does_not_emit_set_body_on_reasoning_state(
        self, reasoning_agent_model, tmp_path
    ):
        """Reasoning states never get a hand-written body wired up."""
        path = _generate(reasoning_agent_model, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        # The runtime API does not expose a ``set_body`` for reasoning
        # states; the generator must not attempt to call it.
        assert "reason.set_body" not in code
        assert "reason.set_fallback_body" not in code

    def test_plain_agent_does_not_import_reasoning(self, tmp_path):
        """Agents without reasoning primitives should not pull in the runtime."""
        agent = Agent("plain")
        agent.platforms.append(WebSocketPlatform())
        agent.new_state("only", initial=True)

        path = _generate(agent, str(tmp_path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "from baf.library.state import new_reasoning_state" not in code
        assert "agent.new_tool(" not in code
        assert "agent.new_skill(" not in code
        assert "agent.new_workspace(" not in code
