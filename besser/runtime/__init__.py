"""besser4DT runtime kernel.

Generic method dispatcher for instance state. Two consumers:

  1. **Generated platforms** — the platform generator copies the rest of
     this package (``engine.py``, ``materialize.py``) into ``<backend>/
     runtime/`` and writes a separate ``__init__.py`` over there that
     wires the kernel to the generated ``services.instance_manager``
     singleton. See ``PlatformGenerator._copy_runtime_kernel``.
  2. **BESSER web modeling editor** — imports ``Engine`` and
     ``materialize_classes`` directly from here to run methods inline on
     the object-diagram view, without generating a platform first.

This module intentionally does NOT create a singleton or import any
generated-app modules. That keeps it usable outside a deployed platform
(in the editor backend, in tests, in a Python REPL).
"""
from .engine import Engine
from .materialize import materialize_classes

__all__ = ["Engine", "materialize_classes"]
