#!/usr/bin/env python3
"""Entry point: `python run.py case case-011` or `python run.py eval`."""

import sys

from pv.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
