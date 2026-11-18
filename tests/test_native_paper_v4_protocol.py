"""Frozen v4 planning tests; tiny synthetic parser fixtures are not results."""
import collections
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import native_paper_v4_protocol as p
from experiment_common import digest, dump


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixtures = ROOT/"tests/fixtures/native_paper_v4"
        self.fixtures.mkdir(parents=True, exist_ok=True)

    def synthetic_dataset(self, root, profile="main"):
        config, maps = p.default_config(), []
        for family in config["profiles"][profile]["families"]:
            for seed in config["profiles"][profile]["map_seeds"]:
                mid = f"{family}_{seed}"
                path = root/(mid+".t25d")
                path.write_text("SYNTHETIC_PARSER_FIXTURE_ONLY "+mid)
                meta = path.with_suffix(".json")
                dump(meta, dict(synthetic_parser_fixture=True))
                cloud = path.with_suffix(".cloud.npy")
                cloud.write_bytes(b"SYNTHETIC_PARSER_FIXTURE_ONLY")
                field = path.with_suffix(".safe_sem.txt")
                field.write_bytes(b"SYNTHETIC_PARSER_FIXTURE_ONLY")
                field_meta = dict(version="synthetic", format="SAFE_SEM_FIELD_V1", bounds=[0, 0, 40, 40],
                    resolution_m=.1, shape_yx=[400,400], cloud_sha256=p.sha256(cloud),
                    map_sha256=p.sha256(path), builder_source_sha256=p.sha256(ROOT/"tools/safe_length_field.py"),
                    field_sha256=p.sha256(field))
                field_meta_path = field.with_suffix(".txt.meta.json")
                dump(field_meta_path, field_meta)
                maps.append(dict(map_id=mid, family=family, map_seed=seed, path=path.name,
                    sha256=p.sha256(path), metadata_sha256=p.sha256(meta), cloud_path=cloud.name,
                    cloud_sha256=p.sha256(cloud), safe_field_path=field.name,
                    safe_field_sha256=p.sha256(field), safe_field_metadata_sha256=p.sha256(field_meta_path),
                    safety_field_builder_key=p.field_builder_key(field_meta),
                    queries=[dict(query_id="q0", start=[2.,2.], goal=[38.,38.])]))
        return config, dict(config_hash=digest(config), profile=profile, maps=maps)

    def test_registered_config_and_unchanged_native_parameters(self):
        from native_paper_v3_protocol import default_config as previous
        config = p.load_config(ROOT/"configs/native_paper_kaggle_v4.yaml")
        self.assertEqual(config["schema_version"], 13)
        self.assertEqual(config["execution"]["required_cpu_capacity"], 4)
        self.assertEqual(config["execution"]["threads_per_trial"], 1)
        self.assertEqual(config["native_parameters"]["a3t_old"], previous()["native_parameters"]["a3"])
        self.assertEqual(config["native_parameters"]["sem_native"], previous()["native_parameters"]["sem"])
        self.assertEqual(config["profiles"]["main"]["map_seeds"], list(range(28001,28011)))
        self.assertEqual(config["profiles"]["main"]["planner_seeds"], list(range(296001,296011)))
        self.assertEqual(config["native_parameters"]["a3t_safe"]["threshold_policy"], "immutable_no_relaxation")
        self.assertNotIn("putn_voxel_m", config["representation"])

    def test_1500_slots_are_paired_balanced_frozen_and_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_plan_", dir=self.fixtures) as tmp:
            root = Path(tmp)
            config, dataset = self.synthetic_dataset(root)
            plan = p.plan_trials(config, "main", dataset, root)
            self.assertEqual(plan["planned_slots"], 1500)
            self.assertEqual(plan["phase_timeouts"], dict(setup_s=600., serialize_s=60., solve_grace_s=5.))
            self.assertEqual(collections.Counter(e["backend"] for e in plan["entries"]), dict.fromkeys(p.BACKENDS, 500))
            self.assertEqual(collections.Counter(e["family"] for e in plan["entries"]), dict.fromkeys(p.FAMILIES, 300))
            blocks = collections.defaultdict(set)
            for entry in plan["entries"]:
                blocks[entry["map_id"], entry["planner_seed"]].add(entry["backend"])
                req = entry["request"]
                self.assertEqual(req["planning_budget_s"], 20.)
                self.assertEqual(req["native_objective_id"], p.OBJECTIVES[entry["backend"]])
                self.assertEqual(req["work_iterations"], 0)
                self.assertEqual(entry["execution_status"], "PLANNED")
                if entry["backend"] == "a3t_old":
                    self.assertEqual(req["objective_type"], "surface_length_traversability")
                    self.assertEqual(req["samples_per_batch"], 200)
                    self.assertNotIn("safe_length", req)
                elif entry["backend"] == "a3t_safe":
                    self.assertEqual(req["objective_type"], "surface_length")
                    self.assertEqual((req["samples_per_batch"], req["batch_after_first_solution"]), (100,200))
                    self.assertEqual(req["safe_length"]["risk_cap"], 40.)
                    self.assertEqual(req["safe_length"]["slope_cap_deg"], 35.)
                    self.assertTrue(req["safe_length"]["atlas_sampling"])
                    self.assertNotIn("traversability", req)
                else:
                    self.assertEqual(req["planner_id"], "sem_native_paper_v2")
                    self.assertNotIn("safe_length", req)
            self.assertEqual(len(blocks), 500)
            self.assertTrue(all(value == set(p.BACKENDS) for value in blocks.values()))
            self.assertEqual(plan, p.plan_trials(config,"main",dataset,root))
            dataset["maps"][0]["cloud_sha256"] = "changed"
            with self.assertRaises(ValueError): p.plan_trials(config,"main",dataset,root)

    def test_smoke_is_separate_21_slots_and_small_budget(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_smoke_plan_", dir=self.fixtures) as tmp:
            config, dataset = self.synthetic_dataset(Path(tmp), "smoke")
            plan = p.plan_trials(config,"smoke",dataset,tmp)
            self.assertEqual(plan["planned_slots"], 21)
            self.assertEqual({e["planner_seed"] for e in plan["entries"]}, {11})
            self.assertEqual({e["request"]["planning_budget_s"] for e in plan["entries"]}, {2.})
            self.assertEqual(len(dataset["maps"]),7)

    def test_changed_config_and_field_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_rejection_", dir=self.fixtures) as tmp:
            root = Path(tmp)
            config, dataset = self.synthetic_dataset(root,"smoke")
            configpath = root/"changed.json"
            changed = copy.deepcopy(config)
            changed["native_parameters"]["a3t_safe"]["risk_cap"] = 41.
            dump(configpath, changed)
            with self.assertRaises(ValueError): p.load_config(configpath)
            with self.assertRaises(ValueError): p.plan_trials(changed,"smoke",dataset,root)
            field = root/dataset["maps"][0]["safe_field_path"]
            field.write_bytes(b"TAMPERED_SYNTHETIC_FIELD")
            with self.assertRaises(ValueError): p.plan_trials(config,"smoke",dataset,root)


if __name__ == "__main__": unittest.main()
