"""API-only command entry; accept both `run.py api ...` and `run.py ...`."""
import sys
from .api.cli import main as api_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["api"]:
        argv = argv[1:]
    return api_main(argv)
