"""Registration and baseline parity tests; no planner campaign is launched."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sem_exposure_protocol as protocol
import corridor_v4_protocol as corridor


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.default_config()
        self.entries = protocol.inventory(self.config)["entries"]
        self.info = dict(map_path=str(ROOT / "tests/fixtures/map.t25d"), cloud_path=str(ROOT / "tests/fixtures/cloud.npy"),
            cloud_sha256="cloud", map_sha256="map", start=[2., 10.], goal=[18., 10.])
        self.field = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="builder")
        self.field_path = ROOT / "tests/fixtures/planner_field.txt"
        self.bridge = ROOT / "tests/fixtures/libbridge.dll"

    def test_frozen_inventory_and_config(self):
        self.assertEqual(protocol.load_config(), self.config)
        self.assertEqual(len(self.entries), 42)
        self.assertEqual(len({row["trial_id"] for row in self.entries}), 42)
        for method in protocol.METHODS:
            rows = [row for row in self.entries if row["method"] == method]
            self.assertEqual(len(rows), 7)
            self.assertEqual({row["planner_seed"] for row in rows}, {11})
        self.assertEqual(self.config["analysis"]["nominal_method"], "a3t_sem_exposure_l1")
        self.assertFalse(self.config["analysis"]["smoke_selects_weight"])

    def test_exposure_requests_full_safe(self):
        for entry in self.entries[:4]:
            request = protocol.request_for(entry, self.info, self.config, field_path=self.field_path,
                                            field_metadata=self.field, bridge=self.bridge)
            self.assertEqual(request["objective_type"], "surface_length_sem_exposure")
            self.assertEqual(request["sem_exposure"], {"lambda": entry["sem_lambda"], "risk_scale": 40.})
            self.assertEqual(request["safe_length"]["policy_id"], "a3t_safe_length_v1")
            self.assertNotIn("corridor_policy", request["safe_length"])
            self.assertEqual(request["safe_length"]["risk_cap"], 40.)
            self.assertTrue(request["safe_length"]["atlas_sampling"])
            self.assertEqual((request["samples_per_batch"], request["batch_after_first_solution"]), (100, 200))

    def test_baselines_are_byte_equivalent_requests(self):
        for method in (protocol.CORRIDOR, protocol.NATIVE):
            entry = next(row for row in self.entries if row["method"] == method)
            actual = protocol.request_for(entry, self.info, self.config, field_path=self.field_path,
                                          field_metadata=self.field, bridge=self.bridge)
            baseline = corridor.request_for(entry, self.info,
                corridor.default_config(beam_scheduler="length_first_stable"), "smoke",
                self.field_path, self.field, self.bridge)
            self.assertEqual(actual, baseline)
            self.assertNotIn("sem_exposure", actual)

    def test_no_weight_seed_or_profile_drift(self):
        for key, value in (("sem_lambda", 2.), ("planner_seed", 12)):
            entry = dict(self.entries[0], **{key: value})
            with self.assertRaisesRegex(ValueError, "unregistered"):
                protocol.request_for(entry, self.info, self.config)
        config = copy.deepcopy(self.config)
        config["sem_exposure"]["risk_scale"] = 20.
        with self.assertRaises(ValueError):
            protocol.validate_config(config)
        for profile in ("pilot", "full"):
            with self.assertRaisesRegex(ValueError, "only smoke"):
                protocol.inventory(self.config, profile)


if __name__ == "__main__":
    unittest.main()
