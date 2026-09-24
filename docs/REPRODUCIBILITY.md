# Reproducibility and repository notes

## What is preserved

The repository includes the existing example photographs and saved results. The main contact sheet contains nine approaches; there is no saved SAM 2 automatic result. Result files alone do not establish which dependency versions, hardware, or exact checkpoint revisions produced them.

The code uses pretrained inference without fine-tuning. Requirements specify minimum versions. A future run can differ if dependencies, default weights, or hub model code change.

## Recording a new experiment

Run experiment commands from the `segmentation-for-photo-editing/` project folder inside the repository. Paths such as `outputs/` below are relative to that folder.

1. Move existing `outputs/` elsewhere before using a different input. The runner and contact-sheet builder reuse paths and do not remove stale results.
2. Record the input filename and checksum, selected model IDs, and any SAM point coordinates.
3. Record Python, installed packages (`python -m pip freeze`), OS, hardware, and device. Keep environment records with that experiment rather than claiming they describe the saved examples.
4. Record checkpoint revisions, preprocessing, failures, elapsed time, and any manual changes.
5. Inspect masks at full resolution alongside composites. Use labelled reference masks and a documented evaluation protocol before reporting accuracy.

The existing effect tests exercise compositing, options, image handling, and mocked integration. They do not evaluate pretrained model accuracy.

```bash
# From the repository root
cd segmentation-for-photo-editing/color_subject_streak
../.venv/bin/python -m unittest discover -s tests -v
```

## Push to GitHub

From the repository root, after creating an empty GitHub repository:

```bash
git init -b main
git add .
git status --short
git commit -m "Add segmentation research prototype and qualitative results"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your repository URL. If Git is already initialized, skip `git init`; if `origin` already exists, use its configured URL instead of adding it again.

The ignore rules keep virtual environments, caches, local secrets, external checkouts, and model weights out of the upload. Existing result folders and README images are explicitly included; other generated output directories are ignored. Original files remain in place.

Original code and documentation are licensed under MIT. The author-created Lightroom XMP presets are private and excluded from uploads; supply your own XMP file to run the preset renderer.
