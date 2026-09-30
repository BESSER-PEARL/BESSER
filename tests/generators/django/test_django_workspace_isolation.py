"""Workspace-isolation tests for the Django generator.

DjangoGenerator.generate() shells out to ``django-admin startproject`` /
``manage.py startapp`` and writes many template files. These tests pin down
that every one of those filesystem effects lands under the generator's own
output_dir — never in the calling process's current working directory — and
that regenerating into the same output_dir is idempotent (a leftover project
directory from a previous run must not hard-fail the generation).
"""

import os
import shutil
import subprocess
import sys

import pytest

from besser.BUML.metamodel.structural import DomainModel, Class, Property
from besser.BUML.metamodel.gui import (
    GUIModel,
    Module,
    DataList,
    DataSourceElement,
    Screen,
)
from besser.generators.django import DjangoGenerator

PROJECT_NAME = "iso_project"
APP_NAME = "iso_app"

requires_django_admin = pytest.mark.skipif(
    shutil.which("django-admin") is None,
    reason="django-admin is not available in this environment",
)


@pytest.fixture
def domain_model():
    """Minimal domain model, mirroring the existing Django generator tests."""
    name_prop = Property(name="name", type="int")
    item = Class(name="Item", attributes=[name_prop])
    return DomainModel(name="model", types={item}, associations=set())


@pytest.fixture
def gui_model(domain_model):
    """Small GUI model with a main page and one list screen."""
    item = next(iter(domain_model.get_classes()))
    name_prop = next(iter(item.attributes))
    datasource = DataSourceElement(
        name="DataSource", dataSourceClass=item, fields=[name_prop]
    )
    data_list = DataList(
        name="MyList", description="items", list_sources={datasource}
    )
    list_screen = Screen(
        name="ItemScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={data_list},
    )
    home_screen = Screen(
        name="HomeScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements=set(),
        is_main_page=True,
    )
    module = Module(name="module_name", screens={home_screen, list_screen})
    return GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="Test app",
        screenCompatibility=True,
        modules={module},
    )


def _fake_subprocess_run(recorded_calls):
    """Stand-in for subprocess.run that scaffolds a minimal Django skeleton.

    Asserting on (and honoring) the ``cwd`` argument is the point: the
    generator must confine both subprocesses to its own output directory.
    """

    def fake_run(cmd, *args, check=False, cwd=None, **kwargs):
        recorded_calls.append({"cmd": list(cmd), "cwd": cwd})
        assert cwd is not None, "generator subprocess must receive an explicit cwd"
        if "startproject" in cmd:
            project_name = cmd[cmd.index("startproject") + 1]
            project_dir = os.path.join(cwd, project_name)
            pkg_dir = os.path.join(project_dir, project_name)
            # Like django-admin, fail if the target already exists.
            os.makedirs(pkg_dir)
            with open(os.path.join(project_dir, "manage.py"), "w", encoding="utf-8") as f:
                f.write("# manage.py stub\n")
            with open(os.path.join(pkg_dir, "settings.py"), "w", encoding="utf-8") as f:
                f.write("INSTALLED_APPS = [\n    'django.contrib.admin',\n]\n")
            with open(os.path.join(pkg_dir, "urls.py"), "w", encoding="utf-8") as f:
                f.write("urlpatterns = []\n")
        elif "startapp" in cmd:
            app_name = cmd[cmd.index("startapp") + 1]
            os.makedirs(os.path.join(cwd, app_name))
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run


def _make_isolated_dirs(tmp_path, monkeypatch):
    """Chdir into a scratch dir foreign to the generation target."""
    foreign_cwd = tmp_path / "foreign_cwd"
    output_dir = tmp_path / "output"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    return foreign_cwd, output_dir


def test_generate_confined_to_output_dir_with_foreign_cwd(
    domain_model, tmp_path, monkeypatch
):
    foreign_cwd, output_dir = _make_isolated_dirs(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(calls))

    gen = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen.generate()

    # Both subprocesses ran with an explicit cwd inside output_dir.
    assert len(calls) == 2
    assert "startproject" in calls[0]["cmd"]
    assert os.path.realpath(calls[0]["cwd"]) == os.path.realpath(str(output_dir))
    assert "startapp" in calls[1]["cmd"]
    assert os.path.realpath(calls[1]["cwd"]) == os.path.realpath(
        str(output_dir / PROJECT_NAME)
    )

    # All generated artifacts landed under output_dir.
    project_dir = output_dir / PROJECT_NAME
    assert (project_dir / "manage.py").exists()
    assert (project_dir / "requirements.txt").exists()
    assert (project_dir / APP_NAME / "models.py").exists()
    assert (project_dir / APP_NAME / "admin.py").exists()
    settings = (project_dir / PROJECT_NAME / "settings.py").read_text(encoding="utf-8")
    assert f"'{APP_NAME}'," in settings

    # Nothing leaked into the caller's working directory.
    assert os.listdir(foreign_cwd) == []
    # And the caller-supplied output_dir was not mutated by generate().
    assert gen.output_dir == str(output_dir)


def test_generate_with_gui_model_confined_to_output_dir(
    domain_model, gui_model, tmp_path, monkeypatch
):
    foreign_cwd, output_dir = _make_isolated_dirs(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(calls))

    gen = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        gui_model=gui_model,
        output_dir=str(output_dir),
    )
    gen.generate()

    app_dir = output_dir / PROJECT_NAME / APP_NAME
    templates_dir = app_dir / "templates"
    assert (app_dir / "models.py").exists()
    assert (app_dir / "forms.py").exists()
    assert (app_dir / "views.py").exists()
    assert (app_dir / "urls.py").exists()
    assert (templates_dir / "home.html").exists()
    assert (templates_dir / "item.html").exists()
    assert (templates_dir / "item_list.html").exists()
    assert (templates_dir / "item_form.html").exists()
    assert (output_dir / PROJECT_NAME / PROJECT_NAME / "urls.py").exists()

    # The GUI/template path used to re-anchor self.output_dir on os.getcwd():
    # nothing may land in the caller's cwd, and output_dir must stay intact.
    assert os.listdir(foreign_cwd) == []
    assert gen.output_dir == str(output_dir)


def test_second_generate_into_same_output_dir_succeeds(
    domain_model, tmp_path, monkeypatch
):
    foreign_cwd, output_dir = _make_isolated_dirs(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(calls))

    gen = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen.generate()

    project_dir = output_dir / PROJECT_NAME
    leftover = project_dir / "leftover_from_crashed_run.txt"
    leftover.write_text("stale", encoding="utf-8")

    # A fresh generator into the same output_dir (the live failure mode:
    # "CommandError: '<name>' already exists") must regenerate cleanly.
    gen2 = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen2.generate()

    assert not leftover.exists(), "stale project dir was not cleaned before regeneration"
    assert (project_dir / "manage.py").exists()
    assert (project_dir / APP_NAME / "models.py").exists()
    # Two generate() calls -> four subprocess invocations, all with cwd set.
    assert len(calls) == 4
    assert all(c["cwd"] is not None for c in calls)
    assert os.listdir(foreign_cwd) == []


def test_leftover_dir_outside_output_dir_is_never_deleted(
    domain_model, tmp_path, monkeypatch
):
    """The clean-regeneration rmtree must stay inside output_dir."""
    foreign_cwd, output_dir = _make_isolated_dirs(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run(calls))

    # A same-named directory in the caller's cwd simulates the shared
    # fixed-location leftover from the original bug.
    stray = foreign_cwd / PROJECT_NAME
    stray.mkdir()
    (stray / "keep.txt").write_text("keep me", encoding="utf-8")

    gen = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen.generate()

    assert (stray / "keep.txt").exists(), "generator deleted files outside output_dir"
    assert (output_dir / PROJECT_NAME / "manage.py").exists()


@requires_django_admin
def test_real_generate_twice_with_foreign_cwd(domain_model, tmp_path, monkeypatch):
    """End-to-end run with the real django-admin: isolated and idempotent."""
    foreign_cwd, output_dir = _make_isolated_dirs(tmp_path, monkeypatch)

    gen = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen.generate()

    project_dir = output_dir / PROJECT_NAME
    assert (project_dir / "manage.py").exists()
    assert (project_dir / PROJECT_NAME / "settings.py").exists()
    assert (project_dir / APP_NAME / "models.py").exists()
    assert os.listdir(foreign_cwd) == []

    # Second generation into the same output_dir must succeed (this exact
    # sequence used to crash with "CommandError: '<name>' already exists").
    sentinel = project_dir / "stale_marker.txt"
    sentinel.write_text("stale", encoding="utf-8")

    gen2 = DjangoGenerator(
        model=domain_model,
        project_name=PROJECT_NAME,
        app_name=APP_NAME,
        output_dir=str(output_dir),
    )
    gen2.generate()

    assert not sentinel.exists()
    assert (project_dir / "manage.py").exists()
    assert (project_dir / APP_NAME / "models.py").exists()
    settings = (project_dir / PROJECT_NAME / "settings.py").read_text(encoding="utf-8")
    assert f"'{APP_NAME}'," in settings
    assert os.listdir(foreign_cwd) == []
