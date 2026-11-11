#!/usr/bin/env python3
"""Render and verify the released psychophysics stimuli."""

import argparse
import json
import os
from pathlib import Path
import re

# Raster and codec workers are parallelized by trial.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

from . import stimuli, runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=(*stimuli.SCOPES, "all"))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("generated"))
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Local Hub snapshot. Missing files are fetched at the pinned revision.",
    )
    parser.add_argument(
        "--revision", default=stimuli.REVISION, help="Full Hub commit SHA."
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args(argv)
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full immutable Hugging Face commit SHA")
    try:
        args.data_root = runner.data_directory(args.data_root)
        runner.check_storage(args.out, minimum_free=1024**3, writable=True)
        for record in stimuli.generate(
            args.task,
            args.sample_id,
            args.out,
            args.data_root,
            args.revision,
            args.workers,
        ):
            print(json.dumps(record, sort_keys=True), flush=True)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
