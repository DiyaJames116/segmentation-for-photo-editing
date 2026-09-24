from pathlib import Path
import argparse
from src.common import make_comparison


def comparison_candidate(directory):
    """Return the best visual result available for one model output folder."""
    for name in ("studio.jpg", "studio.jpeg", "cutout.png"):
        candidate = directory / name
        if candidate.exists():
            return candidate

    # Mask R-CNN writes one file per detected person before it writes a merged
    # result, so retain it in a partial/previous run as well.
    person_cutouts = sorted(directory.glob("person_*_cutout.png"))
    if person_cutouts:
        return person_cutouts[0]

    # These are useful fallbacks for SAM2 auto and partially completed runs.
    for name in ("combined_large_objects.png", "alpha.png", "mask.png"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    args = p.parse_args(argv)
    root = Path(args.root)
    entries = []
    for d in sorted(root.iterdir() if root.exists() else []):
        if not d.is_dir() or d.name == "comparison":
            continue
        candidate = comparison_candidate(d)
        if candidate:
            entries.append((d.name, candidate))
    out = root / "comparison" / "comparison.jpg"
    make_comparison(entries, out)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
