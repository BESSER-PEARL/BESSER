"""Integration tests for BAFGenerator with the reasoning extension.

These tests build a fixture agent that uses every reasoning primitive
(``Tool``, ``Skill``, ``Workspace``, ``ReasoningState``), run the
generator, and assert that the produced output:

    * ``<agent_name>.py`` exists and is non-empty;
    * ``<agent_name>.py`` imports the BAF reasoning runtime symbols;
    * ``tools.py`` exists and contains each tool's function code;
    * ``skills/<name>.md`` exists for each skill;
    * ``<agent_name>.py`` calls ``load_tools``/``load_skills`` instead of
      registering primitives inline;
    * ``<agent_name>.py`` is syntactically valid Python (``ast.parse``).

The tests deliberately avoid importing the BAF runtime — only the text
of the generated files is inspected.
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


def _generate_test_mode(agent: Agent, output_dir: str) -> str:
    """Run BAFGenerator in test mode and return the generated agent path."""
    BAFGenerator(
        model=agent,
        output_dir=output_dir,
        generation_mode=GenerationMode.CODE_ONLY,
        test_mode=True,
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
        agent_path = _generate(reasoning_agent_model, str(tmp_path))
        with open(agent_path, "r", encoding="utf-8") as f:
            agent_code = f.read()
        # Agent script delegates to load_tools, not inline new_tool calls.
        assert "agent.load_tools(" in agent_code
        assert "tools.py" in agent_code
        # Tool source code lives in the separate tools.py module.
        tools_path = os.path.join(str(tmp_path), "tools.py")
        assert os.path.isfile(tools_path)
        with open(tools_path, "r", encoding="utf-8") as f:
            tools_code = f.read()
        assert "def ping" in tools_code

    def test_emits_skill_registration(self, reasoning_agent_model, tmp_path):
        agent_path = _generate(reasoning_agent_model, str(tmp_path))
        with open(agent_path, "r", encoding="utf-8") as f:
            agent_code = f.read()
        # Agent script delegates to load_skills, not inline new_skill calls.
        assert "agent.load_skills(" in agent_code
        assert "skills" in agent_code
        # Each skill is written to skills/<name>.md.
        skill_file = os.path.join(str(tmp_path), "skills", "GreetByName.md")
        assert os.path.isfile(skill_file)
        with open(skill_file, "r", encoding="utf-8") as f:
            skill_content = f.read()
        assert "Always greet the user by name when introduced." in skill_content

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
        assert "agent.load_tools(" not in code
        assert "agent.load_skills(" not in code
        assert "agent.new_workspace(" not in code

    def test_test_mode_still_generates_reasoning_assets(self, reasoning_agent_model, tmp_path):
        """Test-mode generation must include tools/skills and workspace dirs."""
        _generate_test_mode(reasoning_agent_model, str(tmp_path))

        tools_path = os.path.join(str(tmp_path), "tools.py")
        assert os.path.isfile(tools_path)

        skills_path = os.path.join(str(tmp_path), "skills", "GreetByName.md")
        assert os.path.isfile(skills_path)

        workspace_dir = os.path.join(str(tmp_path), "tmp", "cinema")
        assert os.path.isdir(workspace_dir)

