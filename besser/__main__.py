"""``python -m besser``: the ``besser`` command-line interface."""

import sys

from besser.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
