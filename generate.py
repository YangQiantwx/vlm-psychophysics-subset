#!/usr/bin/env python3
"""Compatibility entry for ``python run.py generate``."""
from vlm_psychophysics.generation import main
if __name__ == "__main__":
    raise SystemExit(main())
