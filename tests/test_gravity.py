"""Gravity configuration and preprocessing regression tests."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from omc.gravity import correct_gravity, load_gravity_config, validate_gravity_subjects


def _config():
    return {
        "linear_acceleration_subjects": ["A"],
        "raw_acceleration_subjects": ["F"],
        "raw_gravity_removal": {"method": "lowpass", "cutoff_hz": 0.5, "order": 2},
    }


class GravityTest(unittest.TestCase):
    def test_config_validation_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gravity.json"
            path.write_text(json.dumps(_config()), encoding="utf-8")
            cfg = load_gravity_config(path)
            self.assertEqual(len(cfg["_digest"]), 64)
            validate_gravity_subjects(cfg, ["A", "F"])
            with self.assertRaises(ValueError):
                validate_gravity_subjects(cfg, ["A", "F", "new"])
            path.write_text(json.dumps({**_config(), "linear_acceleration_subjects": ["A", "F"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gravity_config(path)

    def test_linear_acceleration_unchanged(self):
        cfg = _config()
        t = np.arange(300, dtype=float) / 100
        x = np.random.default_rng(1).normal(size=(300, 6)).astype("float32")
        corrected, estimate = correct_gravity(x, t, "A", cfg)
        np.testing.assert_array_equal(corrected, x)
        np.testing.assert_array_equal(estimate, np.zeros((300, 3)))

    def test_raw_static_gravity_removed(self):
        cfg = _config()
        t = np.arange(1000, dtype=float) / 100
        x = np.zeros((1000, 6), dtype="float32")
        x[:, :3] = np.array([0.0, 0.0, 9.81], dtype="float32")
        corrected, estimated = correct_gravity(x, t, "F", cfg)
        self.assertLess(float(np.max(np.abs(corrected[:, :3]))), 1e-3)
        self.assertAlmostEqual(float(np.median(estimated[:, 2])), 9.81, places=3)
        np.testing.assert_array_equal(corrected[:, 3:], x[:, 3:])

    def test_unknown_subject_rejected(self):
        cfg = _config()
        t = np.arange(100, dtype=float) / 100
        with self.assertRaises(ValueError):
            correct_gravity(np.zeros((100, 6), dtype="float32"), t, "new", cfg)

    def test_mean_method_constant_offset(self):
        cfg = _config()
        cfg["raw_gravity_removal"] = {"method": "mean"}
        t = np.arange(100, dtype=float) / 100
        x = np.ones((100, 6), dtype="float32")
        x[:, :3] = [1.0, 2.0, 9.0]
        corrected, _ = correct_gravity(x, t, "F", cfg)
        self.assertAlmostEqual(float(np.linalg.norm(corrected[:, :3])), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
