# Colour subject + monochrome streak background

An experimental feature using the existing BiRefNet adapter to locate the
subject. The background stays in the photograph, becomes black and white, and
receives directional motion blur. The subject retains its original colours and
detail. No XMP preset is applied.

## Run in Terminal

```bash
# From the repository root
source .venv/bin/activate
python color_subject_streak/run_effect.py --image input/photo.jpg
```

The finished image is saved to `color_subject_streak/output/photo_color_streak.jpg`.
HEIC, HEIF, JPG, JPEG, PNG and TIFF inputs are supported, including camera
orientation. Transparent inputs are flattened over white. Install the optional image-format dependencies with
`python -m pip install -r color_subject_streak/requirements.txt`. First use requires the BiRefNet download;
cached weights can be used offline.

## Change the look

```bash
# Longer, horizontal streaks
python color_subject_streak/run_effect.py --image input/photo.jpg --blur 0.15 --angle 0

# Diagonal streaks with a separate output
python color_subject_streak/run_effect.py --image input/photo.jpg --blur 0.10 --angle -35 --output color_subject_streak/output/diagonal.jpg

# Black-and-white background without blur
python color_subject_streak/run_effect.py --image input/photo.jpg --blur 0
```

- `--blur`: fraction of the shorter image dimension, from 0 to 0.5. Default 0.08;
  maximum effective streak length is 501 pixels.
- `--angle`: 0 is horizontal, 90 is vertical. Positive angles slope down-right;
  negative angles slope up-right. Default -25.
- `--output`: choose a JPEG or PNG destination. PNG avoids JPEG colour changes.
- `--mask`: optional existing grayscale mask, white subject / black background,
  matching the oriented photo dimensions. This skips inference for faster tuning.

Alternatively, put a photo in `color_subject_streak/input/` and run the script
without `--image`; it selects the newest supported file by modification time.
Existing output files with the same name are replaced only after encoding
succeeds. Intermediate model files are temporary. Use distinct output paths for
concurrent jobs. Failures return a nonzero exit code; `--debug` adds a traceback.

## Python integration

```python
from color_subject_streak.run_effect import run_effect

finished = run_effect('input/photo.jpg', blur=0.12, angle=-30)
```

The blur samples background pixels only, limiting subject streaks near the
outline. Soft mask edges blend the colour subject with the monochrome background.
Quality depends on segmentation, especially fine hair, glass and group photos.
Areas without usable background samples retain their original grayscale value.
The effect uses an 8-bit RGB workflow, without HDR or ICC-profile conversion.

## Tests

From `Model_Choosing`:

```bash
cd color_subject_streak
../.venv/bin/python -m unittest discover -s tests -v
```
