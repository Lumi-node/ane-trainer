"""Entry point for python -m ane_trainer."""

import sys

from ane_trainer.cli import main_cli


if __name__ == "__main__":
    sys.exit(main_cli())
