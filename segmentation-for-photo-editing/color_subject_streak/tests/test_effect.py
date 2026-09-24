import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from run_effect import apply_effect, motion_kernel, run_effect, load_image


class EffectTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.rgb = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.rgb)

    def test_solid_subject_is_unchanged(self):
        result = apply_effect(self.image, Image.new('L', self.image.size, 255))
        np.testing.assert_array_equal(np.asarray(result), self.rgb)

    def test_background_is_grayscale_and_blurred(self):
        mask = Image.new('L', self.image.size)
        result = np.asarray(apply_effect(self.image, mask, 0.4, 0))
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 1], result[..., 2])
        unblurred = np.asarray(apply_effect(self.image, mask, 0, 0))
        self.assertLess(result[..., 0].std(), unblurred[..., 0].std())

    def test_subject_does_not_streak_into_background(self):
        rgb = np.full((40, 60, 3), 50, np.uint8)
        rgb[:, 25:35] = (255, 0, 0)
        mask = np.zeros((40, 60), np.uint8)
        mask[:, 25:35] = 255
        result = np.asarray(apply_effect(Image.fromarray(rgb), Image.fromarray(mask), 0.5, 0))
        np.testing.assert_array_equal(result, rgb)

    def test_angle_controls_direction(self):
        horizontal = motion_kernel(21, 0)
        vertical = motion_kernel(21, 90)
        self.assertAlmostEqual(float(horizontal.sum()), 1, places=6)
        np.testing.assert_allclose(horizontal.T, vertical, atol=0.005)
        self.assertGreater(horizontal[10].sum(), horizontal[:, 10].sum())

    def test_invalid_options_and_mask(self):
        for blur in (-1, 1, float('nan')):
            with self.assertRaises(ValueError):
                apply_effect(self.image, Image.new('L', self.image.size), blur)
        with self.assertRaises(ValueError):
            apply_effect(self.image, Image.new('L', (1, 1)))
        with self.assertRaises(ValueError):
            apply_effect(self.image, Image.new('L', self.image.size), angle=float('inf'))

    def test_file_pipeline_and_failure_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, mask, output = root/'photo.jpg', root/'mask.png', root/'result.png'
            self.image.save(source)
            Image.new('L', self.image.size, 255).save(mask)
            with patch('run_effect.extract_mask') as model:
                self.assertEqual(run_effect(source, output, mask=mask), output.resolve())
                model.assert_not_called()
            before = output.read_bytes()
            with patch('run_effect.apply_effect', side_effect=RuntimeError('render failed')):
                with self.assertRaises(RuntimeError):
                    run_effect(source, output, mask=mask)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual({p.name for p in root.iterdir()}, {'photo.jpg', 'mask.png', 'result.png'})
            with self.assertRaises(ValueError):
                run_effect(source, source, mask=mask)

    def test_heic_and_jpeg_decode(self):
        from pillow_heif import register_heif_opener
        register_heif_opener()
        with tempfile.TemporaryDirectory() as temp:
            for suffix, fmt in (('.HEIC', 'HEIF'), ('.jpg', 'JPEG'), ('.jpeg', 'JPEG')):
                path = Path(temp) / ('photo' + suffix)
                self.image.save(path, format=fmt)
                self.assertEqual(load_image(path).size, self.image.size)


if __name__ == '__main__':
    unittest.main()
