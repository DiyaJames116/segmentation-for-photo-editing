"""
Optional SCHP human parsing adapter.

SCHP is deliberately kept optional because its repository/checkpoints have
their own installation and weight-download flow. This script explains the
expected integration point and fails clearly if the external checkout is absent.
"""
from pathlib import Path
import sys
from .common import ensure_dir

def run(image_path, out_dir):
    ensure_dir(out_dir)
    raise RuntimeError(
        "SCHP is an optional external dependency. Clone the SCHP/Human Parsing "
        "repository and download its pretrained checkpoint, then wire its "
        "simple_extractor.py into this adapter. See README.md."
    )

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
