"""Planner-field representation/provenance regressions (synthetic inputs only)."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from safe_length_field import build_field, build_field_file, file_hash


def cloud(n=31):
    x, y = np.meshgrid((np.arange(n)+.5)*.1, (np.arange(n)+.5)*.1)
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(n*n)))


class SafeLengthFieldTests(unittest.TestCase):
    def test_flat_support_and_frozen_buffers(self):
        field = build_field(cloud(), [0, 0, 3.1, 3.1])
        self.assertTrue(field.support.all())
        self.assertTrue(np.all(field.risk == 0))
        with self.assertRaises(ValueError):
            field.risk[0, 0] = 100

    def test_hole_propagates_all_three_stencils(self):
        points = cloud()
        points = np.delete(points, 15*31+15, axis=0)
        field = build_field(points, [0, 0, 3.1, 3.1])
        self.assertFalse(field.support[4:27, 4:27].any())
        self.assertTrue(field.support[3, 15])
        self.assertEqual(int((~field.support).sum()), 23*23)
        self.assertTrue(np.isnan(field.risk[~field.support]).all())

    def test_invalid_inputs_and_upper_boundary(self):
        for points, bounds in (([], [0, 0, 1, 1]), ([[0, 0, float("nan")]], [0, 0, 1, 1]),
                               ([[2, 0, 0]], [0, 0, 1, 1]), ([[0, 0, 0]], [0, 0, 0, 1])):
            with self.assertRaises(ValueError):
                build_field(points, bounds)
        field = build_field([[1, 1, 0]], [0, 0, 1, 1])
        self.assertEqual(field.count[-1, -1], 1)
        self.assertEqual(field.support.shape, (10, 10))

    def test_serialized_missing_values_and_hashes(self):
        with tempfile.TemporaryDirectory(prefix="safe_field_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            directory = Path(tmp)
            cp, mp, fp = directory/"synthetic.cloud.npy", directory/"synthetic.map", directory/"field.txt"
            np.save(cp, [[.05, .05, 0.]])
            mp.write_text("synthetic provenance fixture, not planner input")
            metadata = build_field_file(cp, mp, [0, 0, 2, 2], fp)
            self.assertEqual(metadata["cloud_sha256"], file_hash(cp))
            self.assertEqual(metadata["map_sha256"], file_hash(mp))
            self.assertEqual(metadata["field_sha256"], hashlib.sha256(fp.read_bytes()).hexdigest())
            self.assertEqual(json.loads(fp.with_suffix(".txt.meta.json").read_text()), metadata)
            lines = fp.read_text().splitlines()
            self.assertEqual(lines[0], "SAFE_SEM_FIELD_V1")
            self.assertEqual(len(lines), 402)
            self.assertEqual(set(lines[2:]), {"0 0"})
            before = fp.read_bytes()
            with self.assertRaises(FileExistsError):
                build_field_file(cp, mp, [0, 0, 2, 2], fp)
            self.assertEqual(fp.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
