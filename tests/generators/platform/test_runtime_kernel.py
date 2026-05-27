"""End-to-end tests for the Phase 1.0 runtime kernel.

Generates a small platform whose domain model includes a class with an
executable ``step(self, dt)`` method, dynamic-imports the generated
services *and* the runtime engine, and asserts that ▶ Tick actually
mutates instance state. This is the riskiest assumption Phase 1 makes —
that we can compose `Method.code` → emitted `def` → live runtime call —
so it gets a behavioural test, not just structural assertions.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    IntegerType,
    FloatType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)
from besser.generators.platform.platform_generator import PlatformGenerator


# ---------------------------------------------------------------------------
# Domain model fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def counter_model() -> DomainModel:
    """One class with one attribute and one ``step(dt)`` method whose code
    increments the counter. Smallest model that proves the runtime can
    persist mutated attribute state across a tick."""
    counter = Class(
        name="Counter",
        attributes={Property(name="count", type=IntegerType)},
    )
    step = Method(
        name="step",
        parameters=[Parameter(name="dt", type=FloatType)],
        code="self.count = (self.count or 0) + 1",
    )
    counter.methods = {step}
    return DomainModel(name="CounterDomain", types={counter})


@pytest.fixture
def counter_model_full_def() -> DomainModel:
    """Same Counter, but the method's ``code`` field carries the FULL def
    block — the shape the BESSER web modeling editor saves when the user
    authors via the Python Implementation panel. The generator must
    detect this and avoid wrapping it in another ``def`` (which would
    yield a nested function that's only defined, never called)."""
    counter = Class(
        name="Counter",
        attributes={Property(name="count", type=IntegerType)},
    )
    step = Method(
        name="step",
        parameters=[Parameter(name="dt", type=FloatType)],
        code=(
            "def step(self, dt: float):\n"
            '    """User docstring."""\n'
            "    # User comment\n"
            "    self.count = (self.count or 0) + 1\n"
        ),
    )
    counter.methods = {step}
    return DomainModel(name="CounterDomainFullDef", types={counter})


@pytest.fixture
def echo_model() -> DomainModel:
    """A class with a ``step`` whose body uses ``dt`` directly and returns
    a value — exercises the ``result`` channel of TickResponse."""
    echo = Class(
        name="Echo",
        attributes={Property(name="last_dt", type=FloatType)},
    )
    step = Method(
        name="step",
        parameters=[Parameter(name="dt", type=FloatType)],
        code="self.last_dt = dt\nreturn dt * 2",
    )
    echo.methods = {step}
    return DomainModel(name="EchoDomain", types={echo})


# ---------------------------------------------------------------------------
# Dynamic-import helpers (mirror those in test_export_import_roundtrip.py
# but also clear ``runtime`` modules so the kernel rebinds to a fresh
# instance_manager singleton per test).
# ---------------------------------------------------------------------------
_PLATFORM_PACKAGES = ("services", "metamodel", "runtime")


def _drop_platform_modules() -> None:
    for mod_name in list(sys.modules):
        if mod_name in _PLATFORM_PACKAGES or mod_name.startswith(
            tuple(f"{p}." for p in _PLATFORM_PACKAGES)
        ):
            del sys.modules[mod_name]


def _import_platform_runtime(backend_dir: Path):
    """Import the manager + the runtime engine from a generated platform."""
    backend_str = str(backend_dir)
    _drop_platform_modules()
    sys.path.insert(0, backend_str)
    try:
        im_mod = importlib.import_module("services.instance_manager")
        runtime_mod = importlib.import_module("runtime")
        return im_mod, runtime_mod
    except Exception:
        sys.path.remove(backend_str)
        _drop_platform_modules()
        raise


def _cleanup_imports(backend_dir: Path) -> None:
    backend_str = str(backend_dir)
    if backend_str in sys.path:
        sys.path.remove(backend_str)
    _drop_platform_modules()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRuntimeKernel:
    def test_tick_increments_counter_attribute(self, tmp_path, counter_model):
        """The full chain: Method.code → emitted def → engine.tick → store
        mutated. Two ticks in a row should produce count == 2."""
        out = tmp_path / "platform_counter"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            inst = im.create_instance("Counter", "c1", {"count": 0})

            res1 = engine.tick("Counter", inst["id"])
            assert res1["instance"]["attributes"]["count"] == 1
            assert res1["method_name"] == "step"
            assert res1["args"] == {"dt": 1.0}

            res2 = engine.tick("Counter", inst["id"])
            assert res2["instance"]["attributes"]["count"] == 2

            # The store reflects the mutation independently of what the tick
            # call returned — no double-counting via stale snapshots.
            stored = im.get_instance("Counter", inst["id"])
            assert stored["attributes"]["count"] == 2

        finally:
            _cleanup_imports(backend_dir)

    def test_tick_works_when_code_carries_full_def(self, tmp_path, counter_model_full_def):
        """Regression: the BESSER web modeling editor's Python Implementation
        editor saves the entire ``def step(...): body`` block. Without
        special handling, the generator wraps it in another ``def`` and
        the inner one — the one that mutates state — never runs. Make
        sure tick still increments."""
        out = tmp_path / "platform_full_def"
        PlatformGenerator(counter_model_full_def, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            # Confirm we don't have a nested-def in the generated source —
            # the cheapest assertion that catches this regression even
            # before the runtime call below runs. Match real def lines
            # (line starts with optional whitespace then ``def step``);
            # explanatory comments containing the words must not count.
            domain_src = (backend_dir / "metamodel" / "domain_classes.py").read_text(encoding="utf-8")
            real_def_lines = [
                ln for ln in domain_src.splitlines() if ln.lstrip().startswith("def step")
            ]
            assert len(real_def_lines) == 1, (
                "Found a nested def step — generator wrapped a full-def body in "
                f"another def, defeating execution.\n\nMatched: {real_def_lines}\n\n{domain_src}"
            )

            inst = im.create_instance("Counter", "c1", {"count": 0})
            res = engine.tick("Counter", inst["id"])
            assert res["instance"]["attributes"]["count"] == 1
        finally:
            _cleanup_imports(backend_dir)

    def test_tick_passes_dt_and_returns_method_result(self, tmp_path, echo_model):
        """``dt`` should reach the method body and the method's return
        value should land in the response under ``result``. The
        ``engine.tick`` convenience must forward to ``invoke``."""
        out = tmp_path / "platform_echo"
        PlatformGenerator(echo_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            inst = im.create_instance("Echo", "e1", {"last_dt": None})
            res = engine.tick("Echo", inst["id"], dt=2.5)

            # ``args`` is the new contract — ``dt`` is no longer a top-level
            # field on the response but lives inside ``args``.
            assert res["args"] == {"dt": 2.5}
            assert res["result"] == 5.0
            assert res["instance"]["attributes"]["last_dt"] == 2.5

        finally:
            _cleanup_imports(backend_dir)

    def test_invoke_arbitrary_method_with_kwargs(self, tmp_path):
        """The general ``invoke`` path: a class with two methods that take
        named parameters. The runtime should dispatch the right one and
        forward the args by keyword."""
        tank = Class(
            name="Tank",
            attributes={Property(name="level", type=FloatType)},
        )
        fill = Method(
            name="fill",
            parameters=[Parameter(name="amount", type=FloatType)],
            code="self.level = (self.level or 0.0) + amount",
        )
        drain = Method(
            name="drain",
            parameters=[Parameter(name="amount", type=FloatType)],
            code="self.level = max((self.level or 0.0) - amount, 0.0)",
        )
        tank.methods = {fill, drain}
        model = DomainModel(name="TankDomain", types={tank})

        out = tmp_path / "platform_invoke"
        PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            inst = im.create_instance("Tank", "t1", {"level": 0.0})

            r1 = engine.invoke("Tank", inst["id"], "fill", {"amount": 30.0})
            assert r1["instance"]["attributes"]["level"] == 30.0
            assert r1["method_name"] == "fill"
            assert r1["args"] == {"amount": 30.0}

            r2 = engine.invoke("Tank", inst["id"], "fill", {"amount": 10.0})
            assert r2["instance"]["attributes"]["level"] == 40.0

            r3 = engine.invoke("Tank", inst["id"], "drain", {"amount": 25.0})
            assert r3["instance"]["attributes"]["level"] == 15.0
            assert r3["method_name"] == "drain"

        finally:
            _cleanup_imports(backend_dir)

    def test_invoke_with_no_args(self, tmp_path):
        """A parameter-less method must invoke cleanly with ``args=None``
        (or omitted) — that's the path the UI ▶ button takes for methods
        whose signature is ``(self)``."""
        counter = Class(
            name="Counter",
            attributes={Property(name="count", type=IntegerType)},
        )
        reset = Method(
            name="reset",
            parameters=[],
            code="self.count = 0",
        )
        counter.methods = {reset}
        model = DomainModel(name="CounterReset", types={counter})

        out = tmp_path / "platform_noargs"
        PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            inst = im.create_instance("Counter", "c", {"count": 17})
            res = engine.invoke("Counter", inst["id"], "reset")
            assert res["instance"]["attributes"]["count"] == 0
            assert res["args"] == {}

        finally:
            _cleanup_imports(backend_dir)

    def test_invoke_unknown_instance_raises_keyerror(self, tmp_path, counter_model):
        """The router translates KeyError → 404; here we just confirm the
        engine raises rather than silently no-opping (silent simulations
        are the worst kind of bug for a DT)."""
        out = tmp_path / "platform_unknown"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            _, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine

            with pytest.raises(KeyError, match="No instance"):
                engine.invoke("Counter", "nope", "step")
        finally:
            _cleanup_imports(backend_dir)

    def test_invoke_missing_method_raises_attributeerror(self, tmp_path, counter_model):
        """Asking for a method the class doesn't declare must surface as
        AttributeError, not as a None call. The router maps this to 422."""
        out = tmp_path / "platform_missing_method"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            inst = im.create_instance("Counter", "c1", {"count": 0})
            with pytest.raises(AttributeError, match="no callable method"):
                engine.invoke("Counter", inst["id"], "not_a_method")
        finally:
            _cleanup_imports(backend_dir)

    def test_generated_platform_resolves_links_for_tank_step(self, tmp_path):
        """Phase 1.0.8 — the generated platform's Engine must traverse the
        link graph just like the editor backend does. Builds a tiny H2-
        plant model (Tank/Port/MaterialStream with the same role names
        the user is using on H2PlantDSL), generates the platform,
        dynamic-imports the runtime + instance_manager, creates linked
        instances via the real CRUD methods, and asserts ``step(dt)``
        integrates inflow − outflow from connected streams."""
        # Domain model — mirrors the user's hydrogen plant DSL.
        tank = Class(
            name="Tank",
            attributes={Property(name="level", type=FloatType)},
        )
        port = Class(
            name="Port",
            attributes={Property(name="direction", type=StringType)},
        )
        stream = Class(
            name="MaterialStream",
            attributes={Property(name="massFlow", type=FloatType)},
        )

        equi_ports = BinaryAssociation(
            name="equi_ports",
            ends={
                Property(
                    name="equipment",
                    type=tank,
                    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
                ),
                Property(
                    name="ports",
                    type=port,
                    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
                ),
            },
        )
        source_assoc = BinaryAssociation(
            name="source",
            ends={
                Property(
                    name="outgoingStreams",
                    type=stream,
                    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
                ),
                Property(
                    name="sourcePort",
                    type=port,
                    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
                ),
            },
        )
        target_assoc = BinaryAssociation(
            name="target",
            ends={
                Property(
                    name="incomingStreams",
                    type=stream,
                    multiplicity=Multiplicity(min_multiplicity=0, max_multiplicity=9999),
                ),
                Property(
                    name="targetPort",
                    type=port,
                    multiplicity=Multiplicity(min_multiplicity=1, max_multiplicity=1),
                ),
            },
        )

        step = Method(
            name="step",
            parameters=[Parameter(name="dt", type=FloatType)],
            code=(
                "inflow = 0.0\n"
                "outflow = 0.0\n"
                "for p in (self.ports or []):\n"
                "    direction = str(getattr(p, 'direction', '')).strip().lower()\n"
                "    if direction == 'inlet':\n"
                "        for s in (p.incomingStreams or []):\n"
                "            inflow += float(s.massFlow or 0.0)\n"
                "    elif direction == 'outlet':\n"
                "        for s in (p.outgoingStreams or []):\n"
                "            outflow += float(s.massFlow or 0.0)\n"
                "self.level = max(0.0, min(100.0, (self.level or 0.0) + (inflow - outflow) * dt))\n"
            ),
        )
        tank.methods = {step}

        model = DomainModel(
            name="TinyH2Platform",
            types={tank, port, stream},
            associations={equi_ports, source_assoc, target_assoc},
        )

        out = tmp_path / "platform_h2"
        PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
        backend_dir = out / "backend"
        try:
            im_mod, runtime_mod = _import_platform_runtime(backend_dir)
            engine = runtime_mod.engine
            im = im_mod.instance_manager

            # Build the topology via the real CRUD surface so the
            # instance_manager's ``links`` arrays are populated exactly
            # the way an operator's clicks would populate them.
            tank_a = im.create_instance("Tank", "Tank_A", {"level": 10.0})
            port_in = im.create_instance("Port", "Port_In", {"direction": "Inlet"})
            port_out = im.create_instance("Port", "Port_Out", {"direction": "Outlet"})
            stream_in = im.create_instance("MaterialStream", "Stream_In", {"massFlow": 5.0})
            stream_out = im.create_instance("MaterialStream", "Stream_Out", {"massFlow": 2.0})

            im.create_link("Tank", tank_a["id"], "Port", port_in["id"], "equi_ports")
            im.create_link("Tank", tank_a["id"], "Port", port_out["id"], "equi_ports")
            # Stream_In flows INTO Port_In  (target association)
            im.create_link("MaterialStream", stream_in["id"], "Port", port_in["id"], "target")
            # Stream_Out flows OUT OF Port_Out  (source association)
            im.create_link("MaterialStream", stream_out["id"], "Port", port_out["id"], "source")

            res = engine.invoke("Tank", tank_a["id"], "step", {"dt": 2.0})

            # inflow=5, outflow=2, dt=2 → delta=+6; level was 10 → 16
            assert res["instance"]["attributes"]["level"] == 16.0
            assert im.get_instance("Tank", tank_a["id"])["attributes"]["level"] == 16.0

        finally:
            _cleanup_imports(backend_dir)


class TestMethodBodyValidation:
    """The generator must catch broken Python in user-authored method
    bodies BEFORE emitting domain_classes.py, otherwise the operator
    only finds out at backend startup time with an obscure
    IndentationError pointing to generated code they didn't write."""

    def test_clean_body_validates_and_generates(self, tmp_path, counter_model):
        """Sanity: a syntactically correct body still generates fine."""
        out = tmp_path / "platform_clean_body"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        # No exception = pass.
        assert (out / "backend" / "metamodel" / "domain_classes.py").exists()

    def test_broken_indentation_raises_targeted_error(self, tmp_path):
        """A body with mixed-indent steps (the real failure mode from the
        user's H2 plant) should fail loudly with class + method + line
        context, NOT silently emit broken Python."""
        broken_body = (
            "def update_tank_level(self):\n"
            "      inflow = 0.0\n"
            "      for p in (self.ports or []):\n"
            "        direction = ''\n"
            "          if direction == 'inlet':\n"        # ← 2 extra spaces — Python rejects
            "              inflow += 1.0\n"
        )
        tank = Class(
            name="StorageTank",
            attributes={Property(name="level", type=FloatType)},
        )
        bad = Method(
            name="update_tank_level",
            parameters=[],
            code=broken_body,
        )
        tank.methods = {bad}
        model = DomainModel(name="BrokenBody", types={tank})

        out = tmp_path / "platform_broken"
        with pytest.raises(ValueError, match=r"Python syntax error in method 'update_tank_level' on class 'StorageTank'"):
            PlatformGenerator(model, customization=None, output_dir=str(out)).generate()

    def test_bare_body_with_uniform_indent_passes(self, tmp_path):
        """Body-only authoring (no ``def`` line) with a single consistent
        indent level should parse cleanly via the synthetic-def wrap path."""
        tank = Class(
            name="Counter",
            attributes={Property(name="count", type=IntegerType)},
        )
        ok = Method(
            name="increment",
            parameters=[],
            code="self.count = (self.count or 0) + 1",
        )
        tank.methods = {ok}
        model = DomainModel(name="OkBare", types={tank})

        out = tmp_path / "platform_bare_ok"
        # Should not raise.
        PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
        assert (out / "backend" / "metamodel" / "domain_classes.py").exists()


class TestPhase1Wiring:
    """Structural checks: the generator emits all the pieces that compose
    into the runtime end-to-end. If any of these break, the behavioural
    tests above will too — but these isolate which piece regressed."""

    def test_runtime_kernel_copied_into_backend(self, tmp_path, counter_model):
        out = tmp_path / "platform_copy"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        runtime_dir = out / "backend" / "runtime"
        assert (runtime_dir / "__init__.py").exists()
        assert (runtime_dir / "engine.py").exists()
        # __pycache__ filtered out by the copy, in case it existed in source.
        assert not (runtime_dir / "__pycache__").exists()

    def test_domain_classes_emit_step_method(self, tmp_path, counter_model):
        out = tmp_path / "platform_emit"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        domain_src = (out / "backend" / "metamodel" / "domain_classes.py").read_text(encoding="utf-8")
        assert "def step(self, dt):" in domain_src
        assert "self.count = (self.count or 0) + 1" in domain_src

    def test_runtime_router_present_and_wired(self, tmp_path, counter_model):
        out = tmp_path / "platform_router"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        router_src = (out / "backend" / "routers" / "runtime.py").read_text(encoding="utf-8")
        # Both endpoints must be present — /invoke is the general entry,
        # /tick is the scheduler convenience.
        assert '@router.post("/invoke"' in router_src
        assert '@router.post("/tick"' in router_src
        main_src = (out / "backend" / "main.py").read_text(encoding="utf-8")
        assert "from routers import instances, export, runtime" in main_src
        assert "/api/runtime" in main_src

    def test_frontend_emits_method_metadata(self, tmp_path, counter_model):
        """Per-class methods registry lands as a typed array in models.ts —
        drives the per-method ▶ buttons in the PropertyEditor."""
        out = tmp_path / "platform_methods_meta"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        models_src = (out / "frontend" / "src" / "types" / "models.ts").read_text(encoding="utf-8")
        assert "export interface MethodMetadata" in models_src
        assert "export interface ParameterMetadata" in models_src
        # Counter has a single ``step(dt: float)`` method
        assert "methods: [" in models_src
        assert "name: 'step'" in models_src
        assert "name: 'dt'" in models_src

    def test_frontend_property_editor_renders_per_method_buttons(self, tmp_path, counter_model):
        out = tmp_path / "platform_per_method_buttons"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        editor_src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        assert "classMetadata.methods" in editor_src
        assert "invokeMethod" in editor_src
        # Parameter prompt component must be rendered for methods with args
        assert "ParameterPromptDialog" in editor_src

    def test_frontend_api_exports_invoke_helper(self, tmp_path, counter_model):
        out = tmp_path / "platform_api"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        api_src = (out / "frontend" / "src" / "api" / "instances.ts").read_text(encoding="utf-8")
        assert "export const invokeMethod" in api_src
        assert "/runtime/invoke" in api_src
        # tickInstance survives as the scheduler convenience
        assert "export const tickInstance" in api_src
        assert "/runtime/tick" in api_src


class TestInspectorLayout:
    """Right-side inspector replaces the old centred Edit Instance modal —
    locks the new layout so future regressions don't accidentally bring
    the Dialog wrapper back."""

    def test_property_editor_renders_as_aside(self, tmp_path, counter_model):
        out = tmp_path / "platform_aside"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        editor_src = (out / "frontend" / "src" / "components" / "PropertyEditor.tsx").read_text(encoding="utf-8")
        # Side panel, not a Dialog
        assert "<aside" in editor_src
        # Dialog/DialogContent wrapping the form is gone (parameter
        # prompt is still a Dialog, so we check for the previous
        # full-modal markers specifically).
        assert "<DialogContent" not in editor_src or 'sm:max-w-xl' not in editor_src

    def test_instance_node_has_no_hover_popup(self, tmp_path, counter_model):
        out = tmp_path / "platform_no_hover_popup"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        node_src = (out / "frontend" / "src" / "components" / "InstanceNode.tsx").read_text(encoding="utf-8")
        # The old popup labelled its block with "Attributes" + an inline
        # Edit button. Both are gone.
        assert "uppercase tracking-wider text-primary" not in node_src or "Attributes" not in node_src
        assert "isHovered &&" in node_src  # state still around for scale/shadow feedback
        # The popup-specific JSX is gone
        assert "min-w-[220px] max-w-[300px]" not in node_src

    def test_canvas_wires_node_click_for_inspection(self, tmp_path, counter_model):
        out = tmp_path / "platform_click_inspect"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        canvas_src = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        # Clicking a node opens the inspector
        assert "onNodeClick={onNodeClick}" in canvas_src
        # Clicking empty pane closes it
        assert "onPaneClick={onPaneClick}" in canvas_src
        # Inspector is conditionally mounted — canvas expands when nothing is selected
        assert "selectedInstance &&" in canvas_src

    def test_property_editor_remounts_per_selection(self, tmp_path, counter_model):
        """Regression: PropertyEditor must reset its form state when the
        user switches selected instance. Achieved by keying it on the
        selected id so React remounts the component. Without this, the
        first selection's instance name + attribute values leak into
        the second selection's form."""
        out = tmp_path / "platform_propeditor_remount"
        PlatformGenerator(counter_model, customization=None, output_dir=str(out)).generate()
        canvas_src = (out / "frontend" / "src" / "components" / "InstanceCanvas.tsx").read_text(encoding="utf-8")
        assert "key={selectedInstance.id}" in canvas_src
