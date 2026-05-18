"""
LLM-augmented code generation for BESSER.

Uses an LLM to orchestrate BESSER's deterministic generators and customize
their output based on natural language instructions.  Supports multiple
LLM providers (Anthropic Claude and OpenAI GPT/o3).

Usage::

    from besser.generators.llm import LLMGenerator

    # Anthropic (default)
    gen = LLMGenerator(
        model=domain_model,
        instructions="Build a FastAPI backend with JWT auth and PostgreSQL",
        api_key="sk-ant-...",
    )
    gen.generate()

    # OpenAI
    gen = LLMGenerator(
        model=domain_model,
        instructions="Build a FastAPI backend with JWT auth and PostgreSQL",
        api_key="sk-...",
        provider="openai",
        llm_model="gpt-4o",
    )
    gen.generate()

The LLM has access to all BESSER generators as callable tools.  For parts
where a generator exists, it calls it (reliable, tested base code).  For
parts where no generator exists, it writes code from scratch using the
domain model as specification.

Phase 1 method-body policy
--------------------------
Deterministic generators (python_classes, django, ...) emit minimal correct
bodies for class methods whenever the semantics can be inferred from the
class shape (name + attributes): trivial getters/setters, boolean predicates
over a matching attribute, and ``__str__``/``to_string`` over primitive
attributes.  When the semantics cannot be inferred (arbitrary business
logic, e.g. ``calculate_discount``, ``checkout``), they emit
``raise NotImplementedError("Implement <method> based on your business
rules")`` rather than ``pass`` or ``return True``.  This keeps the Phase 1
scaffold honest about which method bodies the user still owns, so the
Phase 2 customise loop neither treats placeholder stubs as authoritative
nor invents behaviour for them.  Body inference lives in each generator's
``templates/method_body.py.j2`` macro.
"""

from besser.generators.llm.llm_client import (
    LLMProvider,
    OpenAIProvider,
    create_llm_client,
)
from besser.generators.llm.llm_generator import LLMGenerator

__all__ = ["LLMGenerator", "LLMProvider", "OpenAIProvider", "create_llm_client"]
