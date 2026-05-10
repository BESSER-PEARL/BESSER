"""Runtime kernel package — copied verbatim into every generated platform.

This package is the home of the besser4DT execution engine. Phase 1.0 ships
with a tick-based ``Engine`` that runs ``step(self, dt)`` method bodies on
instances of the generated domain classes. Future phases plug a state-machine
dispatcher, a periodic scheduler, and a WebSocket delta channel into the
same Engine without touching the generated app's wiring.

Bootstrap is intentionally tiny: import the manager + class map exposed by
the generated ``services.instance_manager``, hand them to the Engine, and
expose a singleton named ``engine``. The router layer imports it as
``from runtime import engine``.
"""
from services.instance_manager import instance_manager

from .engine import Engine

# Module-level singleton used by the generated FastAPI router.
engine = Engine(instance_manager=instance_manager, class_map=instance_manager.class_map)

__all__ = ["Engine", "engine"]
