"""POST /besser_api/deploy-app: off by default, and a docker-compose failure is an error, not a 200."""

import os
import subprocess
import types

import pytest

from besser.generators.django.django_generator import DjangoGenerator
from besser.utilities.web_modeling_editor.backend.services.deployment import docker_deployment
from besser.utilities.web_modeling_editor.backend.services.exceptions import UpstreamError

URL = "/besser_api/deploy-app"

PAYLOAD = {
    "title": "Shop",
    "generator": "django",
    "config": {"project_name": "shop", "app_name": "store"},
    "model": {
        "type": "ClassDiagram",
        "elements": {
            "class-1": {"type": "Class", "name": "Book", "attributes": ["attr-1"], "methods": []},
            "attr-1": {"type": "Attribute", "name": "+ title: str"},
        },
        "relationships": {},
    },
}


class FakeDocker:
    """Stands in for ``subprocess.run`` in the docker deployment service; records each command."""

    def __init__(self, fail_on=None, error=None):
        self.commands = []
        self.fail_on = fail_on
        self.error = error

    def __call__(self, args, **kwargs):
        self.commands.append((args, kwargs))
        if self.fail_on and args[:len(self.fail_on)] == self.fail_on:
            raise self.error
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


@pytest.fixture
def fake_docker(monkeypatch):
    def install(**kwargs):
        fake = FakeDocker(**kwargs)
        # Only this module's view of subprocess: the Django generator runs startproject for real.
        fake_subprocess = types.SimpleNamespace(**{
            name: getattr(subprocess, name)
            for name in ("PIPE", "CalledProcessError", "TimeoutExpired", "CompletedProcess")})
        fake_subprocess.run = fake
        monkeypatch.setattr(docker_deployment, "subprocess", fake_subprocess)
        return fake
    return install


@pytest.fixture
def stub_django(monkeypatch):
    """The generator is not under test (and needs django-admin): write a bare compose project instead."""
    def generate(self):
        project_dir = os.path.join(self.output_dir, self.project_name)
        os.makedirs(project_dir)
        with open(os.path.join(project_dir, "docker-compose.yml"), "w", encoding="utf-8") as f:
            f.write("services:\n  web:\n    volumes: [x]\n")
    monkeypatch.setattr(DjangoGenerator, "generate", generate)


def test_disabled_by_default_answers_403_and_runs_nothing(client, route_settings, fake_docker):
    docker = fake_docker()
    assert route_settings.docker_deploy_enabled is False

    response = client.post(URL, json=PAYLOAD)

    assert response.status_code == 403
    assert "BESSER_ENABLE_DOCKER_DEPLOY" in response.json()["detail"]
    assert docker.commands == []


def test_enabled_deploys_with_timeouts(client, route_settings, fake_docker, stub_django):
    route_settings.docker_deploy_enabled = True
    docker = fake_docker()

    response = client.post(URL, json=PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"status": "deployed", "project_name": "shop"}
    assert [args for args, _ in docker.commands][-3:] == [
        ["docker-compose", "down"], ["docker-compose", "build"], ["docker-compose", "up", "-d"]]
    assert all(kwargs.get("timeout") for _, kwargs in docker.commands)


def test_compose_failure_answers_502_without_internals(client, route_settings, fake_docker, stub_django, caplog):
    route_settings.docker_deploy_enabled = True
    error = subprocess.CalledProcessError(1, ["docker-compose", "build"], stderr=b"/srv/secret/path: boom")
    fake_docker(fail_on=["docker-compose", "build"], error=error)

    response = client.post(URL, json=PAYLOAD)

    assert response.status_code == 502
    assert "/srv/secret/path" not in response.text
    assert "/srv/secret/path: boom" in caplog.text


def test_invalid_app_name_answers_400(client, route_settings, fake_docker):
    route_settings.docker_deploy_enabled = True
    docker = fake_docker()
    payload = {**PAYLOAD, "config": {"project_name": "shop", "app_name": "../x"}}

    response = client.post(URL, json=payload)

    assert response.status_code == 400
    assert docker.commands == []


@pytest.mark.parametrize("error, status", [
    (subprocess.TimeoutExpired(["docker-compose", "up", "-d"], 600), 504),
    (subprocess.CalledProcessError(1, ["docker-compose", "up", "-d"], stderr=b"no daemon"), 502),
    (FileNotFoundError("docker-compose"), 500),
], ids=["timeout", "non-zero exit", "not installed"])
def test_service_raises_instead_of_returning(tmp_path, fake_docker, error, status):
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n    volumes: [x]\n")
    fake_docker(fail_on=["docker-compose", "up", "-d"], error=error)

    with pytest.raises(UpstreamError) as raised:
        docker_deployment.run_docker_compose(str(tmp_path), "shop")

    assert raised.value.status_code == status
    assert str(tmp_path) not in str(raised.value)
