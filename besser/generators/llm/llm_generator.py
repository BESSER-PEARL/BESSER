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
from besser.generators.llm.llm_client import ClaudeLLMClient, _resolve_api_key
from besser.generators.llm.orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)


class LLMGenerator(GeneratorInterface):
    """
    LLM-augmented code generator.

    Uses Anthropic Claude to orchestrate BESSER's deterministic generators
    and customize their output based on natural language instructions.

    For components where BESSER has a generator (FastAPI, Django, React, etc.),
    the LLM calls it to get reliable base code, then modifies the output.
    For components where no generator exists (NestJS, Next.js, custom frameworks),
    the LLM writes code from scratch using the domain model as specification.

    Args:
        model: The BUML domain model.
        instructions: Natural language description of what to build.
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
        llm_model: Claude model ID (default: claude-sonnet-4-20250514).
        gui_model: Optional GUI model (for React/Flutter/WebApp generators).
        agent_model: Optional agent model (for BAF generator).
        agent_config: Optional agent config dict.
        output_dir: Where to write generated files (default: ./output).
        max_turns: Maximum agent loop iterations (default: 50).
        base_url: Custom Anthropic API base URL (for gateways/proxies).
            Also reads from ANTHROPIC_BASE_URL env var.
    """

    def __init__(
        self,
        model,
        instructions: str,
        api_key: str | None = None,
        llm_model: str | None = None,
        gui_model=None,
        agent_model=None,
        agent_config: dict | None = None,
        output_dir: str | None = None,
        max_turns: int | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, output_dir)
        self.instructions = instructions
        self.gui_model = gui_model
        self.agent_model = agent_model
        self.agent_config = agent_config
        self.max_turns = max_turns

        # Resolve API key and create client
        resolved_key = _resolve_api_key(api_key=api_key)
        self.llm_client = ClaudeLLMClient(
            api_key=resolved_key,
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
        orchestrator = LLMOrchestrator(
            llm_client=self.llm_client,
            domain_model=self.model,
            gui_model=self.gui_model,
            agent_model=self.agent_model,
            agent_config=self.agent_config,
            output_dir=output,
            max_turns=self.max_turns,
        )
        result = orchestrator.run(self.instructions)

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
