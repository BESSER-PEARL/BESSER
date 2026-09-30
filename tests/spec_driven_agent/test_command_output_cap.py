"""A flooding command must not fill the worker's disk or memory.

``run_bounded`` captures into temp files OUTSIDE the sandbox and read them
back whole, so ``yes`` (or a server logging in a loop) wrote until the 120 s
timeout and the result was then loaded into worker memory in full.
"""
import subprocess
import sys
import time

from besser.spec_driven_agent.execution import process

CAP = 200_000

# Floods well past the cap, then idles so only the cap can end it early.
_FLOOD = (
    "import sys, time\n"
    "for _ in range(40):\n"
    "    sys.stdout.write('x' * 65536)\n"
    "    sys.stdout.flush()\n"
    "sys.stderr.write('TAIL-MARKER')\n"
    "sys.stdout.write('END-OF-OUTPUT')\n"
    "sys.stdout.flush()\n"
    "time.sleep(60)\n"
)


def test_a_flood_is_killed_at_the_cap_not_the_timeout(monkeypatch):
    monkeypatch.setattr(process, "MAX_CAPTURE_BYTES", CAP, raising=False)
    started = time.monotonic()

    result = process.run_bounded([sys.executable, "-c", _FLOOD], timeout=20)

    assert time.monotonic() - started < 15
    assert result.returncode != 0
    assert len(result.stdout) < CAP + 200, len(result.stdout)
    assert "bytes of output dropped" in result.stdout
    assert "the command was killed" in result.stderr


def test_output_under_the_cap_is_returned_whole():
    result = process.run_bounded(
        [sys.executable, "-c", "print('a' * 1000); import sys; sys.stderr.write('e')"],
        timeout=20,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "a" * 1000
    assert result.stderr == "e"


def test_the_timeout_still_raises(monkeypatch):
    try:
        process.run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    except subprocess.TimeoutExpired:
        return
    raise AssertionError("timeout did not raise")
