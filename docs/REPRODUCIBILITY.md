# Reproducibility

## Scope of the saved results

The repository includes example photographs and saved outputs for nine approaches. No SAM 2 automatic result is included. The experiments use pretrained inference without fine-tuning.

The saved outputs do not include an environment lockfile, hardware log, or exact checkpoint revisions. Dependencies use minimum-version constraints, and model weights are downloaded from upstream sources. Changes to dependencies, default weights, or model code may affect subsequent results.

## Running an experiment

Follow the installation instructions in the [README](../README.md). Run model and editing commands from the `segmentation-for-photo-editing/` project folder; paths below are relative to that folder.

1. Archive the existing `outputs/` directory before changing the input image. The runner reuses output paths, and the comparison builder includes all available model folders, including results from earlier runs.
2. Record the input filename and checksum, selected model IDs, and SAM point coordinates where applicable.
3. Record Python and package versions (`python -m pip freeze`), operating system, hardware, and inference device for each run.
4. Record checkpoint revisions, preprocessing, failures, elapsed time, and manual changes.
5. Inspect full-resolution masks alongside composites. Quantitative evaluation requires labelled reference masks and a documented evaluation protocol.

## Tests

The effect tests cover compositing, option validation, image handling, and mocked integration. They do not measure pretrained model accuracy.

With the project environment activated, run from the repository root:

```bash
cd segmentation-for-photo-editing/color_subject_streak
python -m unittest discover -s tests -v
```

## Preset experiment

The example Lightroom presets are not distributed. The XMP renderer accepts a user-supplied preset; reproducing the exact saved preset result requires the original preset. See the [preset guide](../segmentation-for-photo-editing/preset_testing/README.md).
