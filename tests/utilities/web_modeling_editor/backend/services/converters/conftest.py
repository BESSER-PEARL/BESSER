"""Stub optional `bocl` dep when it isn't importable locally.

Importing any submodule under `backend.services.*` triggers
`backend/services/__init__.py`, which eagerly imports validators that
depend on the external `bocl` package. `bocl` isn't a declared runtime
dependency in `requirements.txt` (it's CI-installed separately). When it
isn't available, we stub just enough of it to let the rest of the import
chain load. If `bocl` *is* available and importable, the stub is a no-op.
"""

import sys
import types


def _install_bocl_stub() -> None:
    try:
        from bocl.OCLWrapper import OCLWrapper  # noqa: F401
        return  # real bocl is healthy; do nothing
    except Exception:  # ImportError or downstream errors inside bocl
        pass

    bocl = types.ModuleType("bocl")
    ocl_wrapper_mod = types.ModuleType("bocl.OCLWrapper")

    class _OCLWrapperStub:
        def __init__(self, *args, **kwargs):
            pass

    ocl_wrapper_mod.OCLWrapper = _OCLWrapperStub
    bocl.OCLWrapper = ocl_wrapper_mod  # type: ignore[attr-defined]
    sys.modules["bocl"] = bocl
    sys.modules["bocl.OCLWrapper"] = ocl_wrapper_mod


_install_bocl_stub()
