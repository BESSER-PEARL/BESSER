"""Tests for the Run/Pause/Step/Reset toolbar and associated backend routes.

Verifies that:
  - Generated App.tsx contains the Run/Pause/Step/Reset buttons when
    there are steppable classes.
  - App.tsx does NOT include simulation controls when no steppable class exists.
  - runtime_control.py router is generated and mounted in main.py.
  - bindings.py router is generated and mounted in main.py.
  - requirements.txt includes websockets.
  - package.json includes recharts.
  - HistoryChart.tsx is generated.
  - The runtime/__init__.py bootstrap wires Scheduler, InputBindings, HistoryStore.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import (
    Class,
    DomainModel,
    FloatType,
    IntegerType,
    Metadata,
    Method,
    MethodImplementationType,
    Parameter,
    Property,
)
from besser.generators.platform.platform_generator import PlatformGenerator


@pytest.fixture
def steppable_model() -> DomainModel:
    """A minimal domain model with one steppable class."""
    counter = Class(
        name="Counter",
        attributes={Property(name="count", type=IntegerType)},
    )
    step = Method(
        name="update_counter",
        parameters=[Parameter(name="dt", type=FloatType)],
        code="self.count = (self.count or 0) + 1",
        implementation_type=MethodImplementationType.CODE,
    )
    step.metadata = Metadata(is_step=True)
    counter.methods = {step}
    return DomainModel(name="RunDomain", types={counter})


@pytest.fixture
def no_step_model() -> DomainModel:
    """A model with no CODE step methods."""
    widget = Class(
        name="Widget",
        attributes={Property(name="label", type=FloatType)},
    )
    return DomainModel(name="NoStepDomain", types={widget})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(model, tmp_path, name="platform"):
    out = tmp_path / name
    PlatformGenerator(model, customization=None, output_dir=str(out)).generate()
    return out


# ---------------------------------------------------------------------------
# App.tsx — toolbar visibility
# ---------------------------------------------------------------------------

class TestRunToolbarInApp:
    def test_run_button_present_when_steppable(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "handleRun" in src
        assert "handlePause" in src
        assert "handleStep" in src
        assert "handleReset" in src

    def test_tick_counter_displayed(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "tickCount" in src

    def test_dt_input_present(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "handleDtChange" in src

    def test_ws_subscription_present(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        assert "subscribeWs" in src or "runtimeApi" in src

    def test_no_run_button_without_steppable_class(self, no_step_model, tmp_path):
        out = _generate(no_step_model, tmp_path, "norun")
        src = (out / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        # Run/Pause controls are guarded by {% if steppable_classes %}
        assert "handleRun" not in src
        assert "handlePause" not in src


# ---------------------------------------------------------------------------
# Backend routers
# ---------------------------------------------------------------------------

class TestBackendRouters:
    def test_runtime_control_router_generated(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        router_src = (out / "backend" / "routers" / "runtime_control.py").read_text(encoding="utf-8")
        assert "/run" in router_src or "scheduler.start" in router_src
        assert "/pause" in router_src or "scheduler.pause" in router_src
        assert "/step" in router_src or "scheduler.step_once" in router_src
        assert "/reset" in router_src or "scheduler.reset" in router_src
        assert "/ws" in router_src

    def test_bindings_router_generated(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        router_src = (out / "backend" / "routers" / "bindings.py").read_text(encoding="utf-8")
        assert "set_binding" in router_src or "/bindings" in router_src

    def test_main_mounts_runtime_control_router(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        main_src = (out / "backend" / "main.py").read_text(encoding="utf-8")
        assert "runtime_control" in main_src
        assert "/api/control" in main_src

    def test_main_mounts_bindings_router(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        main_src = (out / "backend" / "main.py").read_text(encoding="utf-8")
        assert "bindings" in main_src
        assert "/api/bindings" in main_src

    def test_main_uses_lifespan(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        main_src = (out / "backend" / "main.py").read_text(encoding="utf-8")
        assert "lifespan" in main_src


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_requirements_includes_websockets(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        req = (out / "backend" / "requirements.txt").read_text(encoding="utf-8")
        assert "websockets" in req

    def test_package_json_includes_recharts(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        pkg = (out / "frontend" / "package.json").read_text(encoding="utf-8")
        assert "recharts" in pkg


# ---------------------------------------------------------------------------
# Runtime bootstrap
# ---------------------------------------------------------------------------

class TestRuntimeBootstrap:
    def test_bootstrap_wires_scheduler(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        init_src = (out / "backend" / "runtime" / "__init__.py").read_text(encoding="utf-8")
        assert "Scheduler" in init_src
        assert "scheduler" in init_src

    def test_bootstrap_wires_history_and_bindings(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        init_src = (out / "backend" / "runtime" / "__init__.py").read_text(encoding="utf-8")
        assert "HistoryStore" in init_src
        assert "InputBindings" in init_src

    def test_bootstrap_restores_bindings(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        init_src = (out / "backend" / "runtime" / "__init__.py").read_text(encoding="utf-8")
        assert "bindings.load" in init_src

    def test_steppable_classes_in_bootstrap(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        init_src = (out / "backend" / "runtime" / "__init__.py").read_text(encoding="utf-8")
        assert "Counter" in init_src  # The steppable class name


# ---------------------------------------------------------------------------
# Frontend API helpers
# ---------------------------------------------------------------------------

class TestFrontendRuntimeApi:
    def test_runtime_ts_generated(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        api_src = (out / "frontend" / "src" / "api" / "runtime.ts").read_text(encoding="utf-8")
        assert "run()" in api_src or "export async function run" in api_src
        assert "pause()" in api_src or "export async function pause" in api_src
        assert "subscribeWs" in api_src

    def test_bindings_ts_generated(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        api_src = (out / "frontend" / "src" / "api" / "bindings.ts").read_text(encoding="utf-8")
        assert "setBinding" in api_src
        assert "clearBinding" in api_src

    def test_history_chart_generated(self, steppable_model, tmp_path):
        out = _generate(steppable_model, tmp_path)
        chart_src = (out / "frontend" / "src" / "components" / "HistoryChart.tsx").read_text(encoding="utf-8")
        assert "LineChart" in chart_src or "recharts" in chart_src
