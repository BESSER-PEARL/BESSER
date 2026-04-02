"""
LLM-augmented code generation for BESSER.

Uses an LLM (Anthropic Claude) to orchestrate BESSER's deterministic
generators and customize their output based on natural language instructions.

Usage::

    from besser.generators.llm import LLMGenerator

    generator = LLMGenerator(
        model=domain_model,
        instructions="Build a FastAPI backend with JWT auth and PostgreSQL",
        api_key="sk-ant-...",
    )
    generator.generate()

The LLM has access to all BESSER generators as callable tools.  For parts
where a generator exists, it calls it (reliable, tested base code).  For
parts where no generator exists, it writes code from scratch using the
domain model as specification.
"""

from besser.generators.llm.llm_generator import LLMGenerator

__all__ = ["LLMGenerator"]
