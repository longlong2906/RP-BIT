"""Synthetic protocol tests; no planner observations or experiment results."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from safe_length_v2_protocol import load_config, default_config, inventory, request_for, dump, check_selection, digest
from safe_length_field import file_hash


class ProtocolTests(unittest.TestCase):
    def test_registered_cardinalities_and_disjoint_maps(self):
        config = load_config()
        maps = {}
        for profile,count in (("smoke",28),("pilot",210),("full",1500)):
            value = inventory(config,profile)
            self.assertEqual(value["planned_trials"],count)
            self.assertEqual(len({e["trial_id"] for e in value["entries"]}),count)
            self.assertEqual(value["planner_execution_count"],0)
            maps[profile] = {e["map_id"] for e in value["entries"]}
        self.assertFalse(maps["pilot"] & maps["full"])
        self.assertFalse(maps["pilot"] & maps["smoke"])

    def test_unknown_or_changed_frozen_config_rejected(self):
        with tempfile.TemporaryDirectory(prefix="v2_protocol_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path = Path(tmp)/"config.json"
            for change in (lambda c:c.update(unregistered=True),
                           lambda c:c["calibration"].update(grid_spacing_m=.2),
                           lambda c:c["profiles"]["pilot"].update(map_seeds=[1,2])):
                config = default_config()
                change(config)
                dump(path,config)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_request_policies_and_unchanged_native_objective(self):
        config = load_config()
        info = dict(start=[1.,1.],goal=[3.,3.],map_path="synthetic.t25d",map_sha256="a"*64,
                    cloud_path="synthetic.npy",cloud_sha256="b"*64)
        meta = dict(field_sha256="c"*64,cloud_sha256="b"*64,builder_source_sha256="d"*64)
        entries = inventory(config)["entries"][:4]
        requests = {e["method"]:request_for(e,info,config,"smoke","synthetic.txt",meta,"synthetic.dll") for e in entries}
        calibrated = requests["a3t_v2"]
        self.assertEqual(calibrated["safe_length"]["policy_id"],"a3t_safe_length_v2")
        self.assertEqual(calibrated["safe_length"]["risk_slack"],.5)
        self.assertNotIn("risk_cap",calibrated["safe_length"])
        self.assertEqual((calibrated["samples_per_batch"],calibrated["batch_after_first_solution"]),(100,200))
        self.assertEqual(requests["a3t_cap40"]["safe_length"]["risk_cap"],40.)
        self.assertFalse(requests["sem_common_v2"]["safe_length"]["atlas_sampling"])
        self.assertEqual(requests["sem_common_v2"]["planner_id"],"sem_rrtstar_common_v1")
        self.assertEqual(requests["sem_native"]["native_objective_id"],"sem_length_turn_risk_paper_v2")
        self.assertNotIn("safe_length",requests["sem_native"])
        bad = copy.deepcopy(entries[0]); bad["risk_slack"]=4.
        with self.assertRaises(ValueError):
            request_for(bad,info,config,"smoke","synthetic.txt",meta)

    def test_failed_or_synthetic_selection_cannot_authorize_full(self):
        with tempfile.TemporaryDirectory(prefix="v2_selection_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);selection_file=directory/"selection.json"
            config=load_config()
            dump(selection_file,dict(status="NO_CANDIDATE",protocol_id=config["protocol_id"],config_sha256=digest(config)))
            with self.assertRaisesRegex(ValueError,"SELECTED"):
                check_selection(selection_file,config)
            manifest_file,summary_file=directory/"manifest.json",directory/"summary.json"
            dump(manifest_file,dict(profile="pilot",status="COMPLETE"))
            dump(summary_file,dict(synthetic_fixture=True,rows=[]))
            dump(selection_file,dict(status="SELECTED",protocol_id=config["protocol_id"],config_sha256=digest(config),
                pilot_manifest_path=str(manifest_file),pilot_manifest_sha256=file_hash(manifest_file),
                pilot_summary_path=str(summary_file),pilot_summary_sha256=file_hash(summary_file)))
            with self.assertRaisesRegex(ValueError,"complete pilot"):
                check_selection(selection_file,config)


if __name__ == "__main__":
    unittest.main()
