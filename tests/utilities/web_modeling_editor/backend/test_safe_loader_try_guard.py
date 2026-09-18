"""The loader must accept the one control-flow shape BESSER itself emits —
and nothing more.

``domain_model_builder`` wraps a method's ``state_machine`` / ``quantum_circuit``
assignment in ``try: ... except NameError: pass``, because ``sm`` / ``qc`` only
exist when the model was exported as part of a project. The loader rejected
``Try`` outright, so exporting a model with a state-machine method and
re-importing it failed on a file BESSER had just written.

Widening a security allowlist is the risky half of that fix, so these tests pin
the boundary: the guard shape is accepted, every neighbouring shape is refused,
and the classic escapes stay refused inside the guard as well as outside it.
"""
import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
    _safe_buml_loader as loader,
)

ALLOWED = {"holder": object()}


def _load(src):
    return loader.safe_load_buml(src, allowed_names=ALLOWED)


# --------------------------------------------------------------------------- #
# The shape we emit
# --------------------------------------------------------------------------- #
def test_the_builders_guard_is_accepted():
    _load("x = 1\ntry:\n    x.state_machine = sm\nexcept NameError:\n    pass\n")


def test_an_undefined_name_inside_the_guard_is_tolerated_at_runtime():
    """That is the entire purpose: a standalone export has no ``sm``."""
    ns = _load("x = 1\ntry:\n    x.quantum_circuit = qc\nexcept NameError:\n    pass\n")
    assert ns["x"] == 1  # the guarded statement was skipped, not fatal


def test_an_undefined_name_outside_a_guard_is_still_refused():
    with pytest.raises(loader.SafeBumlLoaderError, match="unknown name"):
        _load("x = sm\n")


# --------------------------------------------------------------------------- #
# Every neighbouring shape stays refused
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src, why", [
    ("try:\n    x = 1\nexcept:\n    pass\n", "bare except"),
    ("try:\n    x = 1\nexcept Exception:\n    pass\n", "a different exception"),
    ("try:\n    x = 1\nexcept NameError as e:\n    pass\n", "binding the exception"),
    ("try:\n    x = 1\nexcept NameError:\n    y = 2\n", "a handler with a real body"),
    ("try:\n    x = 1\nexcept NameError:\n    pass\nelse:\n    y = 2\n", "an else clause"),
    ("try:\n    x = 1\nexcept NameError:\n    pass\nfinally:\n    y = 2\n", "a finally clause"),
])
def test_only_the_exact_guard_shape_is_allowed(src, why):
    with pytest.raises(loader.SafeBumlLoaderError):
        _load(src)


def test_other_control_flow_is_still_refused():
    for src in (
        "if True:\n    x = 1\n",
        "for i in []:\n    x = 1\n",
        "while True:\n    x = 1\n",
        "def f():\n    pass\n",
        "class C:\n    pass\n",
        "import os\n",
    ):
        with pytest.raises(loader.SafeBumlLoaderError):
            _load(src)


# --------------------------------------------------------------------------- #
# The guard must not become a smuggling route
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    "try:\n    __import__('os').system('id')\nexcept NameError:\n    pass\n",
    "try:\n    x = holder.__class__.__mro__\nexcept NameError:\n    pass\n",
    "try:\n    x = holder.__class__\nexcept NameError:\n    pass\n",
    "try:\n    x = open('/etc/passwd')\nexcept NameError:\n    pass\n",
    "try:\n    x = eval('1+1')\nexcept NameError:\n    pass\n",
])
def test_the_guard_does_not_smuggle_an_escape(payload):
    """An unresolved NAME is tolerated inside the guard; dunder access,
    builtins and introspection are not."""
    with pytest.raises(loader.SafeBumlLoaderError):
        _load(payload)


def test_builtins_are_still_absent_apart_from_NameError():
    """NameError is an exception class, not a reachable helper."""
    with pytest.raises(loader.SafeBumlLoaderError):
        _load("try:\n    x = NameError\nexcept NameError:\n    pass\n")
