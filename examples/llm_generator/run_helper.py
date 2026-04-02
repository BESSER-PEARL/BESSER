"""
Shared helper for running LLM generator examples with progress output.

Usage in examples:
    from run_helper import run_generator
    run_generator(model, instructions, output_dir)
"""

import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Silence noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def run_generator(model, instructions: str, output_dir: str,
                  gui_model=None, agent_model=None, agent_config=None,
                  llm_model=None, max_turns=None):
    """
    Run the LLM generator with visible progress output.

    Reads API credentials from environment variables:
    - ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
    - ANTHROPIC_BASE_URL (optional, for gateways)
    """
    from besser.generators.llm import LLMGenerator
    from besser.generators.llm.orchestrator import LLMOrchestrator

    # Clean previous output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    def on_progress(turn, tool, status):
        print(f"  [Turn {turn}] {tool}")

    generator = LLMGenerator(
        model=model,
        instructions=instructions,
        gui_model=gui_model,
        agent_model=agent_model,
        agent_config=agent_config,
        llm_model=llm_model,
        output_dir=output_dir,
        max_turns=max_turns,
    )

    print(f"Model: {model.name}")
    print(f"LLM:   {generator.llm_client.model}")
    print(f"Output: {output_dir}")
    print()
    print("Generating... (this may take 1-3 minutes)")
    print("-" * 50)

    output = generator.build_generation_dir()
    orchestrator = LLMOrchestrator(
        llm_client=generator.llm_client,
        domain_model=generator.model,
        gui_model=generator.gui_model,
        agent_model=generator.agent_model,
        agent_config=generator.agent_config,
        output_dir=output,
        max_turns=generator.max_turns,
        on_progress=on_progress,
    )
    result = orchestrator.run(generator.instructions)

    print("-" * 50)
    print(f"Done! {orchestrator.total_turns} turns, {len(orchestrator.tool_calls_log)} tool calls")
    print()
    print("Generated files:")
    for root, _, files in os.walk(result):
        for f in files:
            if not f.startswith(".besser_"):
                rel = os.path.relpath(os.path.join(root, f), result)
                size = os.path.getsize(os.path.join(root, f))
                print(f"  {rel} ({size:,} bytes)")
    print(f"\nOutput directory: {result}")
    return result
