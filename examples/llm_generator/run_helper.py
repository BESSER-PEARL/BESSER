"""
Shared helper for running LLM generator examples with progress + streaming output.

Usage in examples:
    from run_helper import run_generator
    run_generator(model, instructions, output_dir)
"""

import logging
import os
import shutil
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def run_generator(model, instructions: str, output_dir: str,
                  gui_model=None, agent_model=None, agent_config=None,
                  llm_model=None, max_turns=None, provider: str = "anthropic"):
    """
    Run the LLM generator with streaming text + progress output.
    """
    from besser.generators.llm import LLMGenerator
    from besser.generators.llm.orchestrator import LLMOrchestrator

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    def on_progress(turn, tool, status):
        print(f"\n  [{turn}] {tool}")

    def on_text(text):
        """Print LLM text as it streams — character by character."""
        sys.stdout.write(text)
        sys.stdout.flush()

    generator = LLMGenerator(
        model=model,
        instructions=instructions,
        gui_model=gui_model,
        agent_model=agent_model,
        agent_config=agent_config,
        llm_model=llm_model,
        output_dir=output_dir,
        max_turns=max_turns,
        provider=provider,
    )

    # Show which API key is being used (masked)
    key = getattr(generator.llm_client, '_client', None)
    print("-" * 50)
    print(key)
    if key and hasattr(key, 'api_key'):
        raw = str(key.api_key)
        masked = raw[:8] + "..." + raw[-4:] if len(raw) > 12 else "***"
    else:
        masked = "(unknown)"
    print(f"Model:    {model.name}")
    print(f"LLM:      {generator.llm_client.model}")
    print(f"Provider: {provider}")
    print(f"API key:  {masked}")
    print(f"Output:   {output_dir}")
    print()
    print("Generating... (streaming enabled)")
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
        on_text=on_text,
        use_streaming=True,
    )
    result = orchestrator.run(generator.instructions)

    print()
    print("-" * 50)
    stats = f"Done! {orchestrator.total_turns} turns, {len(orchestrator.tool_calls_log)} tool calls"
    if orchestrator._compaction_count > 0:
        stats += f", {orchestrator._compaction_count} compactions"
    print(stats)
    print(f"Cost: {generator.llm_client.usage}")
    if hasattr(orchestrator, '_validation_issues') and orchestrator._validation_issues:
        print(f"\nWarning: {len(orchestrator._validation_issues)} unfixed issues:")
        for issue in orchestrator._validation_issues:
            print(f"  - {issue}")
    print()
    print("Generated files:")
    total_size = 0
    for root, _, files in os.walk(result):
        if "node_modules" in root:
            continue
        for f in files:
            if not f.startswith(".besser_"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, result)
                size = os.path.getsize(full)
                total_size += size
                print(f"  {rel} ({size:,} bytes)")
    print(f"\nTotal: {total_size:,} bytes")
    print(f"Output: {result}")

    # Interactive fix loop — paste errors to get them fixed
    while True:
        print()
        error = input("Paste an error to fix (or Enter to skip): ").strip()
        if not error:
            break
        response = orchestrator.fix_error(error)
        print()
        print(f"Agent: {response}")
        print(f"Cost so far: {generator.llm_client.usage}")

    return result
