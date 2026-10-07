#!/usr/bin/env python3
"""Run the installed package with an external mechanical figure configuration."""

import sys

from neuroimaging_maxent.cli import main

if __name__ == "__main__":
    main(["mechanical", *sys.argv[1:]])
