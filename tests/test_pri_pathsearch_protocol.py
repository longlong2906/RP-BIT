"""Registration boundaries; no experiment or planner is launched."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pri_pathsearch_protocol as protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config()

    def test_five_physical_backends_not_eleven_independent_methods(self):
        for phase, count in (("smoke_serial", 35), ("smoke_parallel", 35), ("pilot", 375)):
            plan = protocol.inventory(self.config, phase)
            self.assertEqual(plan["planned_trials"], count)
            self.assertEqual(len({r["trial_id"] for r in plan["entries"]}), count)
            self.assertEqual({r["method"] for r in plan["entries"]}, set(protocol.METHODS))
            self.assertEqual(plan["planner_execution_count"], 0)
            self.assertTrue(all(r["status"] == "NOT_RUN" for r in plan["entries"]))
        pilot = protocol.inventory(self.config, "pilot")
        self.assertEqual({r["map_seed"] for r in pilot["entries"]}, {41001, 41002, 41003})
        self.assertEqual({r["planner_seed"] for r in pilot["entries"]}, set(range(416001, 416006)))
        self.assertTrue(self.config["analysis"]["pilot_is_development"])

    def test_smoke_schedules_share_scientific_slots(self):
        left, right = (protocol.inventory(self.config, p)["entries"] for p in protocol.PHASES[:2])
        self.assertEqual([{k:v for k,v in r.items() if k != "phase"} for r in left],
                         [{k:v for k,v in r.items() if k != "phase"} for r in right])

    def test_full_and_confirmation_are_not_registered_even_after_selection(self):
        for phase in ("confirm", "full"):
            with self.assertRaises(ValueError):
                protocol.inventory(self.config, phase)
        with self.assertRaises(ValueError):
            protocol.inventory(self.config, "pilot", 1.1)
        changed = deepcopy(self.config)
        changed["analysis"]["stop_after_pilot"] = False
        with self.assertRaises(ValueError):
            protocol.validate_config(changed)

    def request(self, entry):
        info = dict(map_id=entry["map_id"], map_path="synthetic.t25d", map_sha256="map",
            cloud_path="synthetic.npy", cloud_sha256="cloud", query_id="q0", start=[1.,1.], goal=[2.,2.])
        metadata = dict(field_sha256="field", cloud_sha256="cloud", builder_source_sha256="source")
        return protocol.request_for(entry, info, self.config, entry["phase"],
            ROOT/"tests/fixtures/synthetic_field.txt", metadata, ROOT/"synthetic_bridge.dll")

    def test_actual_backend_modes_and_unchanged_sem_legacy(self):
        entries = protocol.inventory(self.config, "pilot")["entries"]
        for method in protocol.METHODS:
            entry = next(r for r in entries if r["method"] == method)
            request = self.request(entry)
            self.assertEqual(request["planning_budget_s"], 20.)
            self.assertEqual(request["planner_seed"], entry["planner_seed"])
            self.assertEqual(request["variant_id"], method)
            if method == protocol.NATIVE:
                self.assertEqual(request["algorithm_budget_policy"], "pri_portfolio_v1")
                self.assertNotIn("pri_pathsearch", request)
            else:
                self.assertEqual(request["objective_type"], "surface_length")
                self.assertEqual(request["planner_id"], "bitstar_a3")
                self.assertEqual(request["safe_length"]["risk_cap"], 40.)
                self.assertNotIn("sem_exposure", request)
                if method == protocol.LEGACY:
                    self.assertEqual(request["pri_portfolio"], {"length_cap_ratio":1.1})
                    self.assertNotIn("pri_pathsearch", request)
                else:
                    self.assertEqual(request["pri_pathsearch"], {"mode":protocol.MODES[method]})
                    self.assertNotIn("pri_portfolio", request)

    def test_unregistered_slot_or_selector_is_rejected(self):
        entry = next(r for r in protocol.inventory(self.config, "pilot")["entries"] if r["method"] == protocol.FULL)
        for key, value in (("mode", "shortcut"), ("planner_seed", 999), ("query_id", "q1"), ("alpha", 1.1)):
            with self.assertRaises(ValueError):
                self.request(dict(entry, **{key:value}))
        for alpha in (True, float("nan"), float("inf"), 1.01):
            with self.assertRaises(ValueError):
                protocol.selector_method(protocol.FULL, alpha)
        with self.assertRaises(ValueError):
            protocol.selector_method(protocol.LEGACY, 1.1)

    def test_source_hash_covers_worker_include_and_build_registration(self):
        hashes = protocol.source_hashes()
        self.assertIn("apps/PriPathSearchWorker.inc", hashes)
        self.assertIn("configs/a3t_pri_pathsearch_v1.json", hashes)
        self.assertIn("tools/run_pri_pathsearch.py", hashes)
        self.assertIn("tests/test_pri_pathsearch_protocol.py", hashes)


if __name__ == "__main__":
    unittest.main()
