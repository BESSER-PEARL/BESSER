"""
Break the circular import for BAFGenerator.

Chain: baf_generator -> services.converters -> services.__init__ -> deployment
       -> github_deploy_api -> config.generators -> baf_generator  (CYCLE)

Strategy: temporarily stub the converters module during BAFGenerator import,
then restore the real modules so other tests can use them.
"""
import importlib
import sys
import types


def _stub_converters():
    """Temporarily stub converters to break import cycle, then restore."""
    _SERVICES = "besser.utilities.web_modeling_editor.backend.services"
    _CONVERTERS = _SERVICES + ".converters"

    # If already imported, nothing to do
    if _CONVERTERS in sys.modules and not isinstance(sys.modules[_CONVERTERS], types.ModuleType):
        return

    # Save any existing modules
    saved = {}
    for key in list(sys.modules.keys()):
        if key.startswith(_SERVICES):
            saved[key] = sys.modules.pop(key)

    # Create minimal stubs
    svc = types.ModuleType(_SERVICES)
    svc.__package__ = _SERVICES
    svc.__path__ = []
    sys.modules[_SERVICES] = svc

    conv = types.ModuleType(_CONVERTERS)
    conv.__package__ = _CONVERTERS
    conv.__path__ = []
    conv.agent_buml_to_json = lambda code: {}
    sys.modules[_CONVERTERS] = conv
    svc.converters = conv

    # Now import BAFGenerator (this triggers the cycle-breaking stub)
    try:
        importlib.import_module("besser.generators.agents.baf_generator")
    except Exception:
        pass

    # Restore real modules so other tests work
    for key in list(sys.modules.keys()):
        if key.startswith(_SERVICES) and key not in saved:
            # Keep only the baf_generator module, remove stubs
            if "baf_generator" not in key:
                del sys.modules[key]

    for key, mod in saved.items():
        sys.modules[key] = mod


_stub_converters()
