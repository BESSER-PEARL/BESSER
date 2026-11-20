"""``network_wait_argv``: a bwrap plan with a private network that starts its command once pasta's interface is up.

The lab's install sandbox (``lab/checks.py``) and its preview services (``services/lab/preview.py``) both need it.
"""

import sys

from besser.spec_driven_agent.execution import sandbox
from besser.spec_driven_agent.execution.sandbox import NETWORK_WAIT_SECONDS, network_wait_argv


def test_the_plan_reports_its_pid_waits_for_its_network_then_runs_the_command():
    command = ["/venv/bin/uv", "pip", "install", "--", "x"]
    plan_argv = ["/usr/bin/bwrap", "--unshare-net", "--chdir", "/w", "--", *command]

    argv = network_wait_argv(plan_argv, command, info_fd=7)

    assert argv[:3] == ["/usr/bin/bwrap", "--info-fd", "7"]
    assert argv[3:7] == ["--unshare-net", "--chdir", "/w", "--"]
    wrapper = argv[7:-len(command)]
    assert wrapper[:4] == [sys.executable, "-I", "-S", "-c"]
    assert wrapper[5] == str(NETWORK_WAIT_SECONDS)
    assert argv[-len(command):] == command


def test_the_wrapper_waits_with_the_shared_probe_then_becomes_the_command():
    argv = network_wait_argv(["bwrap", "--", "cmd"], ["cmd"], info_fd=3)
    source = argv[argv.index("-c") + 1]

    assert sandbox.NETWORK_WAIT_SOURCE in source
    assert "os.execvp(sys.argv[2], sys.argv[2:])" in source
    compile(source, "<network wait>", "exec")
