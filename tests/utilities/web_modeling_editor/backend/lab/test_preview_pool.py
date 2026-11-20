"""The lab preview's pool of slots, with fake supervisors, fake environments and a fake clock (no process runs)."""

from __future__ import annotations

import os
import threading
import time

import pytest

from besser.spec_driven_agent.execution.sandbox import SandboxMounts
from besser.spec_driven_agent.lab.contracts import (
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    LabChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.environments import LabEnvironmentError
from besser.utilities.web_modeling_editor.backend.models.lab import LabPreviewService
from besser.utilities.web_modeling_editor.backend.services.lab import preview
from besser.utilities.web_modeling_editor.backend.services.lab.preview import (
    SLOT_IDLE_TAKEOVER_SECONDS,
    PreviewFullError,
    PreviewPool,
    PreviewSandbox,
    PreviewUnavailableError,
)

WAIT_SECONDS = 10.0
PLANTED_KEY = "sk-ant-api03-" + "Ab1Cd2Ef3Gh4" * 7 + "Xy1234567" + "AA"


def _wait_for(condition, what: str):
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


class FakeSupervisor:
    """Records what the pool asks of a slot's services."""

    made: list["FakeSupervisor"] = []

    def __init__(self, services, *, log_dir, sandbox):
        self.services, self.log_dir, self.sandbox = list(services), log_dir, sandbox
        self.started = self.stopped = self.restarts = 0
        self.restarted: list[str] = []
        self.confined = True
        FakeSupervisor.made.append(self)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def restart(self, name):
        if name not in {service.name for service in self.services}:
            raise KeyError(name)
        self.restarted.append(name)

    def restart_changed(self):
        self.restarts += 1
        return ["backend"]

    def status(self):
        return [LabPreviewService(name=service.name, state="running", detail="running", startedAt=1.0, logTail=[])
                for service in self.services]


class FakeEnvironments:
    """Records each ``ensure``; ``gate``, when set, holds every call until it is set."""

    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []
        self.gate: threading.Event | None = None
        self.error: str = ""

    def ensure(self, repo, *, runner=None):
        if self.gate is not None:
            assert self.gate.wait(WAIT_SECONDS)
        if self.error:
            raise LabEnvironmentError(self.error)
        self.calls.append((repo.key, repo.path, runner))


class FakeWorkspace:
    def __init__(self, root):
        self.root = str(root)

    def change_repos(self, change):
        trees = os.path.join(self.root, "changes", change.id)
        besser = os.path.join(trees, "besser")
        return {
            REPO_FRONTEND: RepoSpec(REPO_FRONTEND, os.path.join(besser, "frontend"), "develop", "o", "f",
                                    git_dir=os.path.join(self.root, "git", "f")),
            REPO_BESSER: RepoSpec(REPO_BESSER, besser, "development", "o", "b", git_dir=os.path.join(self.root, "g")),
            REPO_MODELING_AGENT: RepoSpec(REPO_MODELING_AGENT, os.path.join(trees, "modeling-agent"), "develop", "o",
                                          "m", git_dir=os.path.join(self.root, "git", "m")),
        }


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def fake_supervisors(monkeypatch):
    FakeSupervisor.made = []
    monkeypatch.setattr(preview, "PreviewSupervisor", FakeSupervisor)


@pytest.fixture
def made(tmp_path):
    """Builds pools and stops them at the end."""
    pools = []

    def make(urls=(), *, environments=None, deployment_url="", bind="127.0.0.1", proxy_port=None):
        environments = environments or FakeEnvironments()
        clock = Clock()
        pool = PreviewPool(list(urls), FakeWorkspace(tmp_path), environments,
                           sandbox_for=lambda change: PreviewSandbox(
                               root=os.path.join(str(tmp_path), "changes", change.id), mounts=SandboxMounts()),
                           log_dir=str(tmp_path / "preview-logs"), bind=bind, proxy_port=proxy_port,
                           deployment_url=deployment_url, clock=clock)
        pools.append(pool)
        return pool, environments, clock

    yield make
    for pool in pools:
        pool.stop()


def _change(change_id: str, title: str, owner: str, status: str = "open") -> LabChange:
    return LabChange(id=change_id, slug=title.lower().replace(" ", "-"), title=title, owner=owner, status=status)


A = _change("aaaaaaaaaaaa", "Note element", "ana")
B = _change("bbbbbbbbbbbb", "Dark mode", "ben")
C = _change("cccccccccccc", "Faster export", "cy")
TWO_SLOTS = ["https://preview.lab.example.org", "https://preview-2.lab.example.org"]


def _supervisor_of(change):
    """The supervisor made for ``change`` (slots prepare in parallel threads, in no fixed order)."""
    (supervisor,) = [s for s in FakeSupervisor.made if s.sandbox.root.endswith(change.id)]
    return supervisor


def _ready(pool, index):
    return lambda: pool.view().slots[index].detail == "" and pool.view().slots[index].services[0].state == "running"


def test_one_local_slot_when_no_url_is_set_and_take_prepares_it_in_the_background(made, tmp_path):
    pool, environments, _ = made(proxy_port=9300)

    view = pool.view()
    assert view.enabled is True and view.detail == ""
    (slot,) = view.slots
    assert (slot.slot, slot.url, slot.changeId) == (0, "http://localhost:8180", None)
    assert [service.state for service in slot.services] == ["stopped"] * 3

    taken = pool.take(A, "ana")
    assert (taken.slot, taken.changeId, taken.title, taken.owner, taken.usedBy) == (0, A.id, A.title, "ana", "ana")
    _wait_for(_ready(pool, 0), "the slot to start")

    # Each repository's environment, made with the change's sandboxed install runner.
    assert [key for key, _, _ in environments.calls] == [REPO_FRONTEND, REPO_BESSER, REPO_MODELING_AGENT]
    assert all(callable(runner) for _, _, runner in environments.calls)
    (supervisor,) = FakeSupervisor.made
    assert supervisor.started == 1
    assert supervisor.log_dir == os.path.join(str(tmp_path / "preview-logs"), "slot-0")
    assert supervisor.sandbox.root == os.path.join(str(tmp_path), "changes", A.id)
    ports = {service.name: service.port for service in supervisor.services}
    assert ports == {"backend": 9100, "frontend": 8180, "modeling_agent": 8865}
    agent = supervisor.services[2]
    assert agent.tcp_out == (9300,)


def test_take_returns_before_preparation_ends_and_the_view_names_the_step(made):
    environments = FakeEnvironments()
    environments.gate = threading.Event()
    pool, _, _ = made(environments=environments)

    taken = pool.take(A, "ana")

    assert taken.changeId == A.id
    step = _wait_for(lambda: "editor" in pool.view().slots[0].detail and pool.view().slots[0].detail,
                     "the first step")
    assert step == "Preparing the editor environment"
    slot = pool.view().slots[0]
    assert [service.state for service in slot.services] == ["starting"] * 3
    assert all(service.detail == step for service in slot.services)
    assert FakeSupervisor.made == []

    environments.gate.set()
    _wait_for(_ready(pool, 0), "the slot to start")
    assert FakeSupervisor.made[0].started == 1


def test_take_reuses_the_slot_already_showing_the_change(made):
    pool, environments, clock = made(TWO_SLOTS)
    first = pool.take(A, "ana")
    _wait_for(_ready(pool, first.slot), "slot A")

    clock.now += 30
    again = pool.take(A, "ben")

    assert again.slot == first.slot
    assert again.usedBy == "ben" and again.lastUsedAt == clock.now and again.startedAt == first.startedAt
    assert len(FakeSupervisor.made) == 1 and len(environments.calls) == 3


def test_a_third_change_with_both_slots_busy_is_refused_naming_each_slot(made):
    pool, _, clock = made(TWO_SLOTS)
    pool.take(A, "ana")
    pool.take(B, "ben")
    _wait_for(lambda: _ready(pool, 0)() and _ready(pool, 1)(), "both slots")
    clock.now += 60
    pool.touch(0, "cy")

    with pytest.raises(PreviewFullError) as refused:
        pool.take(C, "cy")

    message = str(refused.value)
    assert 'slot 1 shows "Note element" (@ana), last used by @cy 0 min ago' in message
    assert 'slot 2 shows "Dark mode" (@ben), last used by @ben 1 min ago' in message
    assert "10 minutes" in message


def test_the_slot_idle_longest_is_taken_after_the_timeout(made):
    pool, _, clock = made(TWO_SLOTS)
    pool.take(A, "ana")
    clock.now += 100
    pool.take(B, "ben")
    _wait_for(lambda: _ready(pool, 0)() and _ready(pool, 1)(), "both slots")

    clock.now += SLOT_IDLE_TAKEOVER_SECONDS - 100 - 1
    with pytest.raises(PreviewFullError):
        pool.take(C, "cy")
    clock.now += 1
    taken = pool.take(C, "cy")

    assert taken.slot == 0 and taken.changeId == C.id and taken.owner == "cy"
    _wait_for(_ready(pool, 0), "slot 0 to show C")
    supervisor_a, supervisor_b, supervisor_c = (_supervisor_of(change) for change in (A, B, C))
    assert supervisor_a.stopped == 1 and supervisor_b.stopped == 0 and supervisor_c.started == 1
    assert pool.view().slots[1].changeId == B.id


def test_after_build_restarts_only_the_slots_showing_that_change(made):
    pool, environments, _ = made(TWO_SLOTS)
    pool.take(A, "ana")
    pool.take(B, "ben")
    _wait_for(lambda: _ready(pool, 0)() and _ready(pool, 1)(), "both slots")
    supervisor_a, supervisor_b = _supervisor_of(A), _supervisor_of(B)
    calls_before = len(environments.calls)

    pool.after_build(B.id)

    _wait_for(lambda: supervisor_b.restarts == 1, "B's restart")
    assert supervisor_a.restarts == 0
    # Lock files or requirements a build changed are installed before the restart.
    assert [key for key, path, _ in environments.calls[calls_before:]] == [
        REPO_FRONTEND, REPO_BESSER, REPO_MODELING_AGENT]
    assert all(B.id in path for _, path, _ in environments.calls[calls_before:])
    pool.after_build("dddddddddddd")
    time.sleep(0.1)
    assert (supervisor_a.restarts, supervisor_b.restarts) == (0, 1)


def test_release_frees_the_slots_showing_the_change(made):
    pool, _, _ = made(TWO_SLOTS)
    pool.take(A, "ana")
    pool.take(B, "ben")
    _wait_for(lambda: _ready(pool, 0)() and _ready(pool, 1)(), "both slots")
    supervisor_a, supervisor_b = _supervisor_of(A), _supervisor_of(B)

    pool.release(A.id)

    slot = pool.view().slots[0]
    assert (slot.changeId, slot.title, slot.owner, slot.usedBy, slot.lastUsedAt) == (None, None, None, None, None)
    assert [service.state for service in slot.services] == ["stopped"] * 3
    assert supervisor_a.stopped == 1 and supervisor_b.stopped == 0
    assert pool.take(C, "cy").slot == 0


def test_a_slot_on_the_labs_own_origin_is_off_with_a_reason(made):
    pool, _, _ = made(["https://lab.example.org/", "https://preview-2.lab.example.org"],
                      deployment_url="https://LAB.example.org")

    view = pool.view()
    off, on = view.slots
    assert view.enabled is True
    assert off.enabled is False and off.url is None
    assert "lab's own origin" in off.detail and "BESSER_LAB_PREVIEW_URLS" in off.detail
    assert on.enabled is True and on.url == "https://preview-2.lab.example.org"
    assert pool.take(A, "ana").slot == 1
    with pytest.raises(PreviewFullError):
        pool.take(B, "ben")


def test_the_preview_is_off_when_its_only_slot_is_on_the_labs_origin(made):
    pool, _, _ = made(["https://lab.example.org"], deployment_url="https://lab.example.org/")

    assert pool.view().enabled is False and pool.view().detail
    with pytest.raises(PreviewUnavailableError):
        pool.take(A, "ana")


def test_a_bind_address_that_is_not_an_ip_turns_the_preview_off(made):
    pool, _, _ = made(bind="localhost")

    view = pool.view()
    assert view.enabled is False
    assert "BESSER_LAB_PREVIEW_BIND='localhost' is not an IP address" in view.detail
    assert view.slots[0].enabled is False and view.slots[0].url is None
    with pytest.raises(PreviewUnavailableError, match="BESSER_LAB_PREVIEW_BIND"):
        pool.take(A, "ana")


def test_a_change_without_working_trees_is_refused(made):
    pool, _, _ = made()

    with pytest.raises(PreviewUnavailableError, match="draft"):
        pool.take(_change("dddddddddddd", "", "ana", status="draft"), "ana")


def test_a_failed_preparation_is_shown_and_restart_prepares_again(made):
    environments = FakeEnvironments()
    environments.error = "npm ci failed in /tree (exit 1): ERR!"
    pool, _, _ = made(environments=environments)
    pool.take(A, "ana")

    slot = _wait_for(lambda: pool.view().slots[0].detail.startswith("Could not") and pool.view().slots[0],
                     "the failure")
    assert slot.detail == "Could not prepare the environments: npm ci failed in /tree (exit 1): ERR!"
    assert [service.state for service in slot.services] == ["failed"] * 3
    assert FakeSupervisor.made == []

    environments.error = ""
    pool.restart(0)
    _wait_for(_ready(pool, 0), "the slot to start")


def test_restart_and_unknown_slots_or_services(made):
    pool, _, _ = made()
    pool.take(A, "ana")
    _wait_for(_ready(pool, 0), "the slot")
    (supervisor,) = FakeSupervisor.made

    pool.restart(0, "frontend")
    assert supervisor.restarted == ["frontend"]
    pool.restart(0)
    assert (supervisor.stopped, supervisor.started) == (1, 2)
    with pytest.raises(KeyError):
        pool.restart(1)
    with pytest.raises(KeyError):
        pool.restart(0, "database")
    with pytest.raises(KeyError):
        pool.touch(5, "ana")
    with pytest.raises(KeyError):
        pool.log_tail(0, "database", 10)


def test_log_tail_reads_the_slots_log_redacted(made, tmp_path):
    pool, _, _ = made(TWO_SLOTS)
    log = tmp_path / "preview-logs" / "slot-1" / "backend.log"
    log.parent.mkdir(parents=True)
    log.write_text("started\nboom " + PLANTED_KEY + "\n", encoding="utf-8")

    tail = pool.log_tail(1, "backend", 1)

    assert (tail.slot, tail.service, tail.lines, tail.truncated) == (1, "backend", ["boom [REDACTED]"], True)
    assert pool.log_tail(0, "frontend", 200).lines == []
