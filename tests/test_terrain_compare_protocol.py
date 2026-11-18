"""Schema-10 tests use explicitly synthetic data only under tests/fixtures."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from experiment_common import digest, dump
from terrain_io import write_map
from independent_verifier import Oracle
import terrain_compare_protocol as protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_compare_protocol_",dir=ROOT/"tests/fixtures")
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def fixture_dataset(self,profile):
        config=protocol.default_config(); maps=[]
        for family in config["profiles"][profile]["families"]:
            for seed in config["profiles"][profile]["map_seeds"]:
                path=self.root/f"{family}_{seed}.t25d"
                write_map(path,[[0.]*5 for _ in range(5)],metadata={"synthetic_test_fixture":True})
                maps.append(dict(map_id=path.stem,family=family,map_seed=seed,path=path.name,
                    sha256=protocol.sha256(path),metadata_sha256=protocol.sha256(path.with_suffix(".json")),
                    queries=[dict(query_id="q0",start=[1,1],goal=[3,3])]))
        return config,dict(profile=profile,config_hash=digest(config),maps=maps,synthetic_test_fixture=True)
    def test_slot_counts_rotation_and_blocked_own_sem(self):
        for profile,slots,executed in (("smoke",48,42),("main",720,630)):
            config,dataset=self.fixture_dataset(profile)
            plan=protocol.plan_trials(config,profile,dataset,self.root)
            self.assertEqual((plan["planned_slots"],plan["planned_executable"]),(slots,executed))
            self.assertEqual(len({e["trial_id"] for e in plan["entries"]}),slots)
            for e in plan["entries"]:
                self.assertEqual(e["execution_status"]=="BLOCKED_SOURCE",e["backend"]=="sem" and e["track"]=="own")
                self.assertEqual(e["request"]["traversability"]["lambda"],1.)
            for i in range(0,len(plan["entries"]),4):
                block=plan["entries"][i:i+4]
                self.assertEqual({e["backend"] for e in block},set(protocol.BACKENDS))
                self.assertEqual(len({(e["map_id"],e["planner_seed"],e["track"]) for e in block}),1)
            replay=protocol.plan_trials(config,profile,dataset,self.root)
            self.assertEqual(plan,replay)
    def test_frozen_default_has_no_mutable_aliases(self):
        baseline=copy.deepcopy(protocol.default_config()); changed=protocol.default_config()
        changed["model"]["robot_radius_m"]+=1
        changed["traversability"]["lambda"]+=1
        changed["profiles"]["main"]["families"].append("unregistered")
        self.assertEqual(protocol.default_config(),baseline)
    def test_frozen_load_and_raw_metadata_hash_tampering(self):
        config,dataset=self.fixture_dataset("smoke")
        path=self.root/"config.json";dump(path,config)
        self.assertEqual(protocol.load_config(path),config)
        broken=copy.deepcopy(config);broken["execution"]["max_workers"]=5;dump(path,broken)
        with self.assertRaises(ValueError):protocol.load_config(path)
        file=self.root/dataset["maps"][0]["path"]
        original=file.read_bytes();file.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError,"dataset changed"):
            protocol.plan_trials(config,"smoke",dataset,self.root)
        file.write_bytes(original);file.with_suffix(".json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"dataset changed"):
            protocol.plan_trials(config,"smoke",dataset,self.root)
    def test_request_policies_and_raw_oracle(self):
        file=self.root/"analytic.t25d"
        write_map(file,[[0.]*11 for _ in range(11)],rectangles=[(4.99,0,5.01,10)],metadata={"synthetic_test_fixture":True})
        oracle=Oracle(file);q=dict(start=[1,5],goal=[9,5])
        direct=oracle.validate_path([[1,5],[9,5]],q["start"],q["goal"],traversability=protocol.DEFAULT_TRAVERSABILITY)
        self.assertFalse(direct["valid"])
        for backend in protocol.BACKENDS:
            r=protocol.request_for(backend,"common",q,file,11,2.)
            self.assertEqual(r["samples_per_batch"],200)
            self.assertEqual(r["sampler_query_cache"],backend=="a3")
            self.assertEqual(r["exact_flat_certificate"],backend=="a3")
            self.assertEqual(r["traversability_build_policy"],"skip_dominated")
            self.assertEqual(r["planner_seed"],11)
        sem=protocol.request_for("sem","common",q,file,11,2.)
        self.assertEqual(sem["planner_id"],"sem_rrtstar_common_t_v1")


if __name__=="__main__":unittest.main()
