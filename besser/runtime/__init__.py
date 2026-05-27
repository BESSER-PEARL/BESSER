"""besser4DT runtime kernel.

Generic method dispatcher for instance state and Digital Twin simulation
utilities. Three consumers:

  1. **Generated platforms** — the platform generator copies this entire
     package into ``<backend>/runtime/`` and writes a separate
     ``__init__.py`` that wires the kernel singletons to the generated
     ``services.instance_manager``. See ``PlatformGenerator._copy_runtime_kernel``.
  2. **BESSER web modeling editor** — imports ``Engine`` and
     ``materialize_classes`` to run methods inline on the object-diagram
     view, without generating a platform first.
  3. **Tests / REPL** — imports any of the components directly.

This module intentionally does NOT create singletons or import any
generated-app modules. That keeps it usable outside a deployed platform.
"""
from .engine import Engine
from .materialize import materialize_classes
from .bindings import InputBindings
from .history import HistoryStore
from .scheduler import Scheduler

__all__ = ["Engine", "materialize_classes", "InputBindings", "HistoryStore", "Scheduler"]
