"""Generators whose dependency is an optional extra fail with the install hint.

Django, docker and the database drivers left the base install (B5): importing a
generator must not need its extra, and using it without the extra must name
the ``pip install "besser[...]"`` that fixes it, not raise a bare traceback.
"""
import builtins
import subprocess
import sys

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property


def _model():
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True),
        Property(name="title", type=PrimitiveDataType("str")),
    }
    return DomainModel(name="M", types={book})


def _without_docker(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docker" or name.startswith("docker."):
            raise ImportError("No module named 'docker'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "docker", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_backend_generator_imports_without_docker():
    # A fresh interpreter: the module is imported once per process.
    code = ("import sys; sys.modules['docker'] = None; "
            "from besser.generators.backend import BackendGenerator")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_docker_image_build_without_docker_names_the_extra(tmp_path, monkeypatch):
    from besser.generators.backend import BackendGenerator

    generator = BackendGenerator(model=_model(), output_dir=str(tmp_path))
    generator.config = {"docker_image_name": "i", "docker_repository": "r", "docker_tag": "t",
                        "docker_port": 8000}
    generator.has_nn_methods = False
    _without_docker(monkeypatch)
    with pytest.raises(ImportError, match=r'besser\[docker\]'):
        generator.build_and_push_docker_image(str(tmp_path))


def test_django_generator_without_django_admin_names_the_extra(tmp_path, monkeypatch):
    from besser.generators.django import DjangoGenerator

    def missing_command(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", args[0][0])

    monkeypatch.setattr(subprocess, "run", missing_command)
    with pytest.raises(RuntimeError, match=r'besser\[django\]'):
        DjangoGenerator(model=_model(), project_name="myproject", app_name="app",
                        output_dir=str(tmp_path)).generate()
