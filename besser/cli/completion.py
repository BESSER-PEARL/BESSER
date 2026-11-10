"""``besser completion``: shell completion scripts generated from the argument parser."""

from __future__ import annotations

import argparse

from besser.cli.output import Result, failure

COMPLETION_SHELLS = ("bash", "zsh", "fish")


def completion_script(shell: str, parser: argparse.ArgumentParser) -> Result:
    """The completion script for ``shell``, built from ``parser``."""
    return failure("completion", "not implemented yet")  # TODO(completion agent)
