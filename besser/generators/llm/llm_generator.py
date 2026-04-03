"""
LLM-augmented code generator for BESSER.

Implements ``GeneratorInterface`` so it integrates seamlessly with the
rest of BESSER — same pattern as PythonGenerator, DjangoGenerator, etc.

The LLM has access to all BESSER generators as callable tools.  For parts
covered by existing generators, it calls them (reliable base code).  For
everything else, it writes code from scratch using the domain model.

Usage::

    from besser.generators.llm import LLMGenerator

    # Minimal — just a model and instructions
    gen = LLMGenerator(
        model=domain_model,
        instructions="Build a FastAPI backend with JWT auth and PostgreSQL",
        api_key="sk-ant-...",
    )
    gen.generate()

    # Full — with GUI model, agent, and custom LLM model
    gen = LLMGenerator(
        model=domain_model,
        instructions="Build a full-stack web app with dark mode",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        llm_model="claude-opus-4-20250514",
        gui_model=gui_model,
        agent_model=agent_model,
        agent_config=agent_config,
        output_dir="./my_app",
    )
    gen.generate()
"""

import logging
import os
from typing import Any

from besser.generators import GeneratorInterface
from besser.generators.llm.llm_client import (
    ClaudeLLMClient,
    LLMProvider,
    create_llm_client,
    _resolve_api_key,
)
from besser.generators.llm.orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)


class LLMGenerator(GeneratorInterface):
    """
    LLM-augmented code generator.

    Uses an LLM to orchestrate BESSER's deterministic generators
    and customize their output based on natural language instructions.

    Supports multiple LLM providers via the ``provider`` parameter:
    - ``"anthropic"`` (default) — Anthropic Claude models
    - ``"openai"`` — OpenAI GPT / o3 models (requires ``pip install openai``)

    For components where BESSER has a generator (FastAPI, Django, React, etc.),
    the LLM calls it to get reliable base code, then modifies the output.
    For components where no generator exists (NestJS, Next.js, custom frameworks),
    the LLM writes code from scratch using the domain model as specification.

    Args:
        model: The BUML domain model.
        instructions: Natural language description of what to build.
        api_key: API key for the selected provider (or set via environment variable).
        llm_model: Model ID (default depends on provider).
        provider: LLM provider to use: ``"anthropic"`` or ``"openai"``.
        gui_model: Optional GUI model (for React/Flutter/WebApp generators).
        agent_model: Optional agent model (for BAF generator).
        agent_config: Optional agent config dict.
        output_dir: Where to write generated files (default: ./output).
        max_turns: Maximum agent loop iterations (default: 50).
        max_cost_usd: Maximum estimated cost in USD before stopping (default: 5.0).
        max_runtime_seconds: Maximum wall-clock time in seconds (default: 1200 = 20 min).
        base_url: Custom API base URL (for gateways/proxies).
    """

    def __init__(
        self,
        model,
        instructions: str,
        api_key: str | None = None,
        llm_model: str | None = None,
        provider: str = "anthropic",
        gui_model=None,
        agent_model=None,
        agent_config: dict | None = None,
        output_dir: str | None = None,
        max_turns: int | None = None,
        max_cost_usd: float = 5.0,
        max_runtime_seconds: int = 1200,
        base_url: str | None = None,
    ):
        super().__init__(model, output_dir)
        self.instructions = instructions
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.max_turns = max_turns
        self.max_cost_usd = max_cost_usd
        self.max_runtime_seconds = max_runtime_seconds

        # Create client via factory (handles API key resolution per provider)
        self.llm_client: LLMProvider = create_llm_client(
            provider=provider,
            api_key=api_key,
            model=llm_model,
            base_url=base_url,
        )

    def generate(self) -> str:
        """
        Run LLM-orchestrated code generation.

        The LLM analyzes the model and instructions, calls BESSER generators
        where applicable, and writes custom code for everything else.

        Returns:
            Path to the output directory containing all generated files.

        Raises:
            ValueError: If instructions are empty or output directory is empty after generation.
        """
        if not self.instructions or not self.instructions.strip():
            raise ValueError("Instructions cannot be empty")

        output = self.build_generation_dir()
        self._orchestrator = LLMOrchestrator(
            llm_client=self.llm_client,
            domain_model=self.model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            agent_config=self.agent_config,
            output_dir=output,
            max_turns=self.max_turns,
            max_cost_usd=self.max_cost_usd,
            max_runtime_seconds=self.max_runtime_seconds,
        )
        result = self._orchestrator.run(self.instructions)

        # Validate that something was actually generated
        generated_files = []
        for root, _, files in os.walk(result):
            for f in files:
                if not f.startswith(".besser_"):
                    generated_files.append(os.path.relpath(os.path.join(root, f), result))

        if not generated_files:
            logger.warning("LLM generation produced no output files")

        logger.info(
            "LLM generation complete: %d files in %s",
            len(generated_files), result,
        )
        return result

    def fix_error(self, error_message: str) -> str:
        """
        Fix an error from running the generated code.

        Call this after ``generate()`` with the error/traceback from your
        terminal. The LLM analyzes it and either fixes the code or tells
        you what to do (e.g., "rebuild Docker", "reset database").

        Usage::

            gen = LLMGenerator(model=m, instructions="...")
            gen.generate()

            # User runs the app, gets an error
            response = gen.fix_error("ImportError: No module named 'auth'")
            print(response)  # "Fixed: added auth.py with JWT logic"

        Returns:
            The LLM's explanation of what it did or what you should do.
        """
        if not hasattr(self, '_orchestrator') or self._orchestrator is None:
            raise RuntimeError("Call generate() first before fix_error()")
        return self._orchestrator.fix_error(error_message)
