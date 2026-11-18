"""Synthetic registration tests, never measured planner evidence."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from corridor_v3_protocol import default_config,inventory,load_config,request_for,method_specs


class ProtocolTests(unittest.TestCase):
    def test_registered_counts_and_disjoint_holdout(self):
        config=load_config()
        self.assertEqual(config,default_config())
        for profile,count in (("smoke",28),("pilot",270),("full",2000)):
            manifest=inventory(config,profile)
            self.assertEqual(manifest["planned_trials"],count)
            self.assertEqual(manifest["planner_execution_count"],0)
            self.assertEqual(len({e["trial_id"] for e in manifest["entries"]}),count)
        self.assertFalse(set(config["profiles"]["full"]["map_seeds"]) & set(range(31001,32011)))

    def test_every_method_is_fully_paired_without_seed_replacement(self):
        manifest=inventory(default_config(),"pilot")
        slots={}
        for entry in manifest["entries"]:
            slots.setdefault(entry["method"],set()).add((entry["map_id"],entry["planner_seed"]))
        self.assertEqual(len(slots),9)
        self.assertTrue(all(slot==next(iter(slots.values())) for slot in slots.values()))
        config=default_config()
        config["corridor_policy"]["alpha"]=2
        with self.assertRaisesRegex(ValueError,"unregistered"):
            inventory(config)

    def test_requests_keep_surface_objective_and_matched_masks(self):
        config=default_config()
        info=dict(map_path="synthetic.t25d",start=[1.,1.],goal=[5.,5.],cloud_path="synthetic.npy",
                  cloud_sha256="cloud",map_sha256="map")
        metadata=dict(field_sha256="field",cloud_sha256="cloud",builder_source_sha256="builder")
        entries=inventory(config)["entries"][:4]
        requests={entry["method"]:request_for(entry,info,config,"smoke","field.txt",metadata,"bridge.dll")
                  for entry in entries}
        a3=requests["a3t_corridor_v3"]
        common=requests["sem_common_corridor_v3"]
        self.assertEqual(a3["objective_type"],"surface_length")
        self.assertEqual(a3["safe_length"]["corridor_policy"],common["safe_length"]["corridor_policy"])
        self.assertEqual(a3["safe_length"]["policy_id"],"a3t_corridor_v3")
        self.assertTrue(a3["safe_length"]["atlas_sampling"])
        self.assertFalse(common["safe_length"]["atlas_sampling"])
        self.assertEqual((a3["samples_per_batch"],a3["batch_after_first_solution"]),(100,200))
        self.assertEqual(requests["a3t_cap40"]["safe_length"]["risk_cap"],40.)

    def test_unknown_alpha_and_unselected_full_are_rejected(self):
        with self.assertRaises(ValueError):
            method_specs("full",1.15)
        config=default_config()
        entry=inventory(config,"full")["entries"][0]
        with self.assertRaisesRegex(ValueError,"alpha"):
            request_for(entry,{},config,"full")


if __name__=="__main__":
    unittest.main()
