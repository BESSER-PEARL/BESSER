"""pasta forwarding exactly the named ports, for the lab's preview services.

A preview service runs in its own sandbox with a private network: the host reaches its one port
(``-t <bind>/<port>``) and it reaches exactly the host loopback ports it needs (``-T <port>``, the
LLM proxy). UDP stays closed and the gateway is never mapped. With no ports named, pasta's argv is
the pre-change one (golden, recorded from it).
"""

import subprocess

import pytest

from besser.spec_driven_agent.execution import sandbox as sandbox_mod
from besser.spec_driven_agent.execution import shell_session
from besser.spec_driven_agent.execution.sandbox import (
    SandboxUnavailable,
    pasta_argv,
    spawn_long_lived,
    start_pasta,
)

_GOLDEN = ["/usr/bin/pasta", "-f", "-q", "--config-net", "--runas", "0:0",
           "-t", "none", "-u", "none", "-T", "none", "-U", "none", "--no-map-gw", "4242"]
_GOLDEN_DNS = [*_GOLDEN[:-1], "--dns-forward", "169.254.1.53", "--dns-host", "127.0.0.11", "4242"]


def _values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_default_arguments_give_the_pre_change_argv():
    assert pasta_argv("/usr/bin/pasta", 4242) == _GOLDEN
    assert pasta_argv("/usr/bin/pasta", 4242, "127.0.0.11") == _GOLDEN_DNS


def test_forwards_only_named_ports():
    """D1 test 2."""
    argv = pasta_argv("/usr/bin/pasta", 4242, tcp_in=(("127.0.0.1", 9100),), tcp_out=(9300,))
    assert _values(argv, "-t") == ["127.0.0.1/9100"]
    assert _values(argv, "-T") == ["9300"]
    assert _values(argv, "-u") == ["none"] and _values(argv, "-U") == ["none"]
    assert "--no-map-gw" in argv
    assert argv[0] == "/usr/bin/pasta" and argv[-1] == "4242"


def test_several_ports_each_get_their_own_option():
    argv = pasta_argv("pasta", 1, tcp_in=(("0.0.0.0", 9100), ("::1", 8180)), tcp_out=(9300, 9301))
    assert _values(argv, "-t") == ["0.0.0.0/9100", "::1/8180"]
    assert _values(argv, "-T") == ["9300", "9301"]


@pytest.mark.parametrize("tcp_in, tcp_out", [
    ((("127.0.0.1/0", 9100),), ()),
    ((("localhost", 9100),), ()),
    ((("127.0.0.1", "1-65535"),), ()),
    ((), ("1-65535",)),
    ((), (0,)),
    ((), (70000,)),
])
def test_a_malformed_forward_is_refused(tcp_in, tcp_out):
    """A port range or a spec in the address would forward far more than one port."""
    with pytest.raises(ValueError):
        pasta_argv("pasta", 1, tcp_in=tcp_in, tcp_out=tcp_out)


def test_start_pasta_forwards_the_ports(monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _n: "/usr/bin/pasta")
    monkeypatch.setattr(sandbox_mod, "_loopback_nameserver", lambda: None)
    monkeypatch.setattr(sandbox_mod.subprocess, "Popen", lambda argv, **_k: seen.append(argv) or "proc")
    assert start_pasta(4242, subprocess.DEVNULL, tcp_in=(("127.0.0.1", 9100),),
                       tcp_out=(9300,)) == "proc"
    assert seen == [pasta_argv("/usr/bin/pasta", 4242, tcp_in=(("127.0.0.1", 9100),), tcp_out=(9300,))]


def test_start_pasta_without_pasta_refuses(monkeypatch):
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _n: None)
    with pytest.raises(SandboxUnavailable):
        start_pasta(4242, subprocess.DEVNULL, tcp_in=(("127.0.0.1", 9100),))


def test_spawn_long_lived_is_the_shell_sessions_spawner():
    """Moved from shell_session._spawn; sessions and the preview share the one thread."""
    assert shell_session._spawn is spawn_long_lived
    assert spawn_long_lived(lambda: 7) == 7
