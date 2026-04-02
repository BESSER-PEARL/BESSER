"""
Example 6: Using a custom API gateway (enterprise setup).

Shows all the ways to configure the API key and base URL
for environments that proxy Anthropic through a gateway.

Usage:
    # Option A: Environment variables (recommended)
    $env:ANTHROPIC_BASE_URL = "https://gateway.pia.private.list.lu"
    $env:ANTHROPIC_AUTH_TOKEN = "your-token"
    python example_6_gateway_config.py

    # Option B: Direct parameters (see code below)
"""

from besser.BUML.metamodel.structural import (
    Class, Property, DomainModel, PrimitiveDataType,
)
from besser.generators.llm import LLMGenerator

# ── Minimal model ────────────────────────────────────────────────────

StringType = PrimitiveDataType("str")
IntegerType = PrimitiveDataType("int")

item = Class(name="Item")
item.attributes = {
    Property(name="id", type=IntegerType, is_id=True),
    Property(name="name", type=StringType),
}

model = DomainModel(name="Demo", types={item})

# ── Option A: Env vars (simplest — just set them before running) ─────
# The generator auto-reads:
#   ANTHROPIC_BASE_URL  →  gateway URL
#   ANTHROPIC_AUTH_TOKEN  →  auth token (or ANTHROPIC_API_KEY)

generator_a = LLMGenerator(
    model=model,
    instructions="Generate a simple Python dataclass for this model",
    output_dir="./output/demo_env",
)

# ── Option B: Explicit parameters ────────────────────────────────────

generator_b = LLMGenerator(
    model=model,
    instructions="Generate a simple Python dataclass for this model",
    api_key="your-gateway-token",
    base_url="https://gateway.pia.private.list.lu",
    output_dir="./output/demo_explicit",
)

# ── Option C: Different Claude model ─────────────────────────────────

generator_c = LLMGenerator(
    model=model,
    instructions="Generate a simple Python dataclass for this model",
    llm_model="claude-opus-4-20250514",  # Use Opus for complex tasks
    output_dir="./output/demo_opus",
)

# ── Run whichever config you want ────────────────────────────────────

print("Running with env var config...")
result = generator_a.generate()
print(f"Generated files in: {result}")
