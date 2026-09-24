import argparse
import importlib
from pathlib import Path
import traceback

MODELS = {
    1: ("01_binary_u2net", "01_binary"),
    2: ("02_deeplab", "02_deeplab"),
    3: ("03_segformer", "03_segformer"),
    4: ("04_maskrcnn", "04_maskrcnn"),
    5: ("05_mask2former", "05_mask2former"),
    6: ("06_birefnet", "06_birefnet"),
    7: ("07_human_parser", "07_human"),
    8: ("08_modnet", "08_modnet"),
    9: ("09_sam2_prompt", "09_sam2_prompt"),
    10: ("10_sam2_auto", "10_sam2_auto"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="input/photo.jpg")
    ap.add_argument("--only", default=None, help="Comma-separated model numbers, e.g. 1,2,4,6")
    ap.add_argument("--sam-point", default="0.5,0.45",
                    help="Normalized x,y point for SAM2 prompt")
    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
        raise SystemExit(f"Input image not found: {image}")

    selected = list(MODELS)
    if args.only:
        selected = [int(x.strip()) for x in args.only.split(",")]

    sam_point = tuple(float(x) for x in args.sam_point.split(","))

    for n in selected:
        if n not in MODELS:
            print(f"[skip] Unknown model number {n}")
            continue
        module_name, out_name = MODELS[n]
        print(f"\n{'='*70}\n[{n}] {module_name}\n{'='*70}")
        try:
            module = importlib.import_module(f"src.{module_name}")
            if n == 9:
                module.run(str(image), f"outputs/{out_name}", point=sam_point)
            else:
                module.run(str(image), f"outputs/{out_name}")
            print("[ok]")
        except Exception as e:
            print(f"[failed] {e}")
            traceback.print_exc(limit=2)

    try:
        import compare
        # ``run_all.py`` has its own CLI flags.  Passing an empty argument list
        # prevents compare.py from trying to parse flags such as ``--only``.
        compare.main([])
    except Exception as e:
        print(f"[comparison failed] {e}")

if __name__ == "__main__":
    main()
