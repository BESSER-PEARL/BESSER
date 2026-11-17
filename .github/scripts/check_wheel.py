"""Check that the installed ``besser`` wheel imports and carries its package data.

Run with the interpreter of an environment the wheel was installed into, from
outside the checkout, so ``besser`` resolves to the installed copy.

Usage: python .github/scripts/check_wheel.py
"""
import pkgutil
from importlib import import_module
from importlib.resources import files

import besser
import besser.generators

# Every generator imports on the base install; extras load lazily.
for module in pkgutil.iter_modules(besser.generators.__path__):
    import_module(f"besser.generators.{module.name}")

root = files("besser")
for path in (
    "llm/data/model_catalog.json",
    "llm/data/model_prices.json",
    "generators/spring/resources/mvnw",
    "utilities/data/gitleaks_rules.toml",
):
    assert root.joinpath(path).is_file(), f"missing from the wheel: {path}"
templates = list(root.joinpath("generators/django/templates").iterdir())
assert templates, "generator templates missing from the wheel"
print("wheel OK:", besser.__file__)
