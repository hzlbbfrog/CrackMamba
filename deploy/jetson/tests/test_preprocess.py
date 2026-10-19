"""CPU checks for normalization, geometry and lossless prediction export."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import collect_images, load_image, save_prediction, validate_output_directory


class PreprocessTests(unittest.TestCase):
    def test_rgb_channel_order_normalization_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pixel.png"
            Image.fromarray(np.array([[[0, 127, 255]]], dtype=np.uint8)).save(path)
            array = load_image(path, size=(1, 1))
            self.assertEqual(array.shape, (1, 3, 1, 1))
            self.assertEqual(array.dtype, np.float32)
            np.testing.assert_allclose(array[0, :, 0, 0], [-1, -1 / 255, 1], atol=1e-7)
            with self.assertRaisesRegex(ValueError, "--resize"):
                load_image(path)
            self.assertEqual(load_image(path, resize=True).shape, (1, 3, 480, 480))

    def test_sorted_recursive_images_and_collision_free_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs"
            root.mkdir()
            (root / "sub").mkdir()
            for name in ("z.png", "a.png", "a.jpg", "sub/a.png"):
                Image.new("RGB", (1, 1)).save(root / name)
            images = collect_images(root)
            self.assertEqual([p.relative_to(root).as_posix() for p in images],
                             ["a.jpg", "a.png", "sub/a.png", "z.png"])
            probability = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)
            paths = [save_prediction(probability, image, root, Path(temporary) / "outputs", 0.5)
                     for image in images]
            self.assertEqual(len({item["probability"] for item in paths}), 4)
            np.testing.assert_array_equal(np.load(paths[0]["probability"]), probability[0, 0])
            np.testing.assert_array_equal(np.asarray(Image.open(paths[0]["preview"])), [[0, 128, 255]])
            np.testing.assert_array_equal(np.asarray(Image.open(paths[0]["binary"])), [[0, 255, 255]])

    def test_reject_logits_and_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.png"
            path.touch()
            for probability in (np.array([[2.0]]), np.array([[np.nan]])):
                with self.assertRaises(ValueError):
                    save_prediction(probability, path, path, Path(temporary) / "out")

    def test_outputs_cannot_pollute_recursive_input_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "images"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_output_directory(source, source / "predictions")
            validate_output_directory(source, Path(temporary) / "predictions")


if __name__ == "__main__":
    unittest.main()
