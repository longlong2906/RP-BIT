"""Scientific registration and request boundaries; no planner is launched."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pri_portfolio_protocol as protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config()

    def test_exact_inventory_and_disjoint_held_out_maps(self):
        plans = {phase: protocol.inventory(self.config, phase, 1.025 if phase == "confirm" else None)
                 for phase in protocol.PHASES}
        self.assertEqual([plans[phase]["planned_trials"] for phase in protocol.PHASES], [35, 35, 375, 1000])
        for phase, plan in plans.items():
            self.assertEqual(len({entry["trial_id"] for entry in plan["entries"]}), plan["planned_trials"])
            self.assertTrue(all(entry["status"] == "NOT_RUN" for entry in plan["entries"]))
            self.assertTrue(all(entry["query_id"] == "q0" for entry in plan["entries"]))
        pilot = {entry["map_id"] for entry in plans["pilot"]["entries"]}
        confirm = {entry["map_id"] for entry in plans["confirm"]["entries"]}
        self.assertTrue(pilot.isdisjoint(confirm))
        self.assertEqual(len(pilot), 15)
        self.assertEqual(len(confirm), 50)
        for family in protocol.FAMILIES:
            for method in (protocol.method_for(1.025), protocol.NATIVE):
                self.assertEqual(sum(entry["family"] == family and entry["method"] == method
                                     for entry in plans["confirm"]["entries"]), 100)

    def test_smoke_serial_parallel_have_identical_scientific_slots(self):
        a = protocol.inventory(self.config, "smoke_serial")["entries"]
        b = protocol.inventory(self.config, "smoke_parallel")["entries"]
        self.assertEqual([{k: v for k, v in row.items() if k != "phase"} for row in a],
                         [{k: v for k, v in row.items() if k != "phase"} for row in b])

    def test_unregistered_config_and_selection_fail_closed(self):
        changed = deepcopy(self.config)
        changed["profiles"]["pilot"]["planner_seeds"].append(999)
        with self.assertRaises(ValueError):
            protocol.validate_config(changed)
        with self.assertRaises(ValueError):
            protocol.inventory(self.config, "confirm")
        with self.assertRaises(ValueError):
            protocol.inventory(self.config, "pilot", 1.)
        for alpha in (True, float("nan"), float("inf"), .99, 1.01, 1.2):
            with self.assertRaises(ValueError):
                protocol.method_for(alpha)

    def _request(self, entry):
        info = dict(map_id=entry["map_id"], map_path="synthetic.t25d", map_sha256="map",
            cloud_path="synthetic.npy", cloud_sha256="cloud", query_id="q0", start=[1., 1.], goal=[2., 2.])
        metadata = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="source")
        return protocol.request_for(entry, info, self.config, entry["phase"],
            ROOT / "tests/fixtures/synthetic_field.txt", metadata, ROOT / "synthetic_bridge.dll")

    def test_portfolio_request_keeps_length_objective_and_native_sem_is_opt_in(self):
        entries = protocol.inventory(self.config, "pilot")["entries"]
        for method in protocol.METHODS:
            entry = next(row for row in entries if row["method"] == method)
            request = self._request(entry)
            self.assertEqual(request["planner_seed"], entry["planner_seed"])
            self.assertEqual(request["planning_budget_s"], 20.)
            if method == protocol.NATIVE:
                self.assertNotIn("pri_portfolio", request)
                self.assertEqual(request["algorithm_budget_policy"], "pri_portfolio_v1")
            else:
                self.assertEqual(request["planner_id"], "bitstar_a3")
                self.assertEqual(request["objective_type"], "surface_length")
                self.assertEqual(request["pri_portfolio"], {"length_cap_ratio": entry["alpha"]})
                self.assertEqual(request["safe_length"]["risk_cap"], 40.)
                self.assertEqual(request["safe_length"]["slope_cap_deg"], 35.)
                self.assertNotIn("sem_exposure", request)

    def test_changed_method_seed_or_alpha_is_rejected_before_launch(self):
        entry = next(row for row in protocol.inventory(self.config, "pilot")["entries"]
                     if row["method"] != protocol.NATIVE)
        for key, value in (("alpha", 1.2), ("planner_seed", 999), ("query_id", "q1"), ("phase", "confirm")):
            changed = dict(entry, **{key: value})
            with self.assertRaises(ValueError):
                self._request(changed)
        one = next(row for row in protocol.inventory(self.config, "pilot")["entries"] if row["alpha"] == 1.)
        with self.assertRaises(ValueError):
            self._request(dict(one, alpha=True))


if __name__ == "__main__":
    unittest.main()
