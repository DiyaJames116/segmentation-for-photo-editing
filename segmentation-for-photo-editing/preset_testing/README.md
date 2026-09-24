# Experimental XMP preset renderer

This standalone photo-editing experiment reads Lightroom / Camera Raw XMP settings and approximates their effects with Pillow, NumPy, and OpenCV. It is separate from the segmentation comparison.

The Lightroom presets used for the saved example were created by the project author and are kept private. Their XMP files are not distributed or covered by the repository's MIT license. The renderer and example output remain available.

Export your own Lightroom preset as an XMP file, then run from inside the `segmentation-for-photo-editing/` project folder after installing `requirements.txt`. Replace `path/to/your-preset.xmp` with your file's path:

```bash
python preset_testing/test_preset.py --image preset_testing/input/photo.jpg --preset path/to/your-preset.xmp
```

The default output naming convention is `outputs/<image>__<preset>.jpg` inside this directory. Use `--output path/to/result.jpg` to choose another destination. Despite its name, `test_preset.py` is the renderer's command-line entry point, not a unit-test suite.

![Saved studio preset example](outputs/photo__studio.jpg)

The implementation includes operations for crop, white balance, exposure, tonal adjustments, detail, curves, HSL, calibration, colour grading, grain, and vignette. These are approximations on decoded images, not Lightroom parity or a RAW processing pipeline. Parsing an XMP setting does not guarantee that its original Adobe behavior is reproduced.

The saved example is illustrative; no quantitative colour-matching evaluation is included. Reproducing its exact appearance requires the private original preset.
