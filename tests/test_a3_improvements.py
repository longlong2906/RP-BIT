"""Configuration and provenance regression tests; no experimental runs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from experiment_common import digest,dump,load_config,read_json,resolved_planner_options
from run_experiments import plan,request_for,verify_variant_result


CONFIG=ROOT/"configs/a3_improvements_smoke.yaml"


class A3ImprovementConfigTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(CONFIG)
        self.entry={"map_path":"fixture.t25d","planner_id":"bitstar_a3","planner_seed":11,
                    "start":[1.,1.],"goal":[9.,9.]}

    def load_mutation(self,config,profile="smoke"):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.yaml"
            path.write_text(yaml.safe_dump(config),encoding="utf-8")
            return load_config(path,profile)

    def test_smoke_is_bounded_and_preserves_original_control(self):
        profile=self.config["profiles"]["smoke"]
        self.assertEqual(set(self.config["profiles"]),{"smoke"})
        self.assertEqual(profile["planning_budget_s"],2.)
        self.assertEqual(profile["planners"][0],"bitstar_a3")
        self.assertEqual(len(profile["families"])*len(profile["map_seeds"])*
                         profile["queries_per_map"]*len(profile["planner_seeds"])*
                         len(profile["planners"]),40)
        original=resolved_planner_options(self.config,"bitstar_a3")
        self.assertEqual(original,{"planner_id":"bitstar_a3","state_cache":False,
                                  "scalar_cost":False,"collision_index":False,
                                  "samples_per_batch":200,"batch_after_first_solution":0})

    def test_old_configs_and_control_keep_exact_worker_request(self):
        original=load_config(ROOT/"configs/experiments.yaml")
        expected={"map_path":"fixture.t25d","planner_id":"bitstar_a3","planner_seed":11,
                  "start":[1.,1.],"goal":[9.,9.],"planning_budget_s":2.,
                  "samples_per_batch":200,"rewire_factor":1.1,"sample_attempts":100,
                  "progress_interval_s":.05}
        self.assertEqual(request_for(self.entry,original,"smoke"),expected)
        self.assertEqual(request_for(self.entry,self.config,"smoke"),expected)
        for filename,profile in (("a3_vs_advanced_v1.yaml","advanced"),
                                  ("a3_vs_sem_rrtstar_common_v1.yaml","sem_comparison"),
                                  ("a3_vs_all_v1.yaml","combined")):
            cfg=load_config(ROOT/"configs"/filename,profile)
            self.assertNotIn("planner_variants",cfg)
            req=request_for(self.entry,cfg,profile)
            self.assertNotIn("variant_id",req)
            self.assertNotIn("state_cache",req)

    def test_variants_resolve_backend_and_independent_flags(self):
        for label in self.config["planner_variants"]:
            with self.subTest(label=label):
                request=request_for(dict(self.entry,planner_id=label),self.config,"smoke")
                self.assertEqual(request["planner_id"],"bitstar_a3")
                self.assertEqual(request["variant_id"],label)
                for key,value in resolved_planner_options(self.config,label).items():
                    self.assertEqual(request[key],value)
        cache=request_for(dict(self.entry,planner_id="a3_cache"),self.config,"smoke")
        self.assertTrue(cache["state_cache"])
        self.assertFalse(cache["scalar_cost"])
        self.assertFalse(cache["collision_index"])
        two_phase=request_for(dict(self.entry,planner_id="a3_two_phase50to200"),self.config,"smoke")
        self.assertEqual((two_phase["samples_per_batch"],two_phase["batch_after_first_solution"]),(50,200))

    def test_planned_labels_and_resolved_options_are_hashed(self):
        # Synthetic input metadata is in memory; no result or data files created.
        profile=self.config["profiles"]["smoke"]
        manifest={"maps":[{"map_id":family+"_101","map_path":"fixture.t25d",
                            "path":"fixture.t25d","sha256":"synthetic_test_fixture",
                            "family":family,"map_seed":101,
                            "queries":[{"query_id":"q0","start":[1.,1.],"goal":[9.,9.]}]}
                           for family in profile["families"]]}
        entries=plan(self.config,"smoke",manifest)
        self.assertEqual(len(entries),40)
        self.assertEqual(len({e["trial_id"] for e in entries}),40)
        self.assertEqual(entries,plan(self.config,"smoke",manifest))
        for entry in entries:
            self.assertEqual(entry["backend_planner_id"],"bitstar_a3")
            self.assertEqual(entry["resolved_optimizations_sha256"],digest(entry["resolved_optimizations"]))
            req=request_for(entry,self.config,"smoke")
            self.assertEqual(req.get("variant_id",req["planner_id"]),entry["planner_id"])
        entry=next(e for e in entries if e["planner_id"]=="a3_cache")
        changed=copy.deepcopy(self.config)
        changed["planner_variants"]["a3_cache"]["scalar_cost"]=True
        self.assertNotEqual(digest(changed),digest(self.config))
        with self.assertRaisesRegex(ValueError,"changed after trial planning"):
            request_for(entry,changed,"smoke")

    def test_unknown_fields_and_bad_types_are_rejected(self):
        mutations=[("state_cache","true"),("state_cache",1),("scalar_cost",None),
                   ("collision_index",0),("samples_per_batch",True),
                   ("samples_per_batch",0),("samples_per_batch",50.5),
                   ("samples_per_batch",2**32),("batch_after_first_solution",-1),
                   ("batch_after_first_solution",False),("typo_option",True),
                   ("planner_id","a3_cache")]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                config=copy.deepcopy(self.config)
                config["planner_variants"]["a3_cache"][key]=value
                with self.assertRaises(ValueError):self.load_mutation(config)

    def test_original_backend_cannot_be_overridden_or_dropped(self):
        config=copy.deepcopy(self.config)
        config["planner_variants"]["bitstar_a3"]={"planner_id":"bitstar_a3","state_cache":True}
        with self.assertRaises(ValueError):self.load_mutation(config)
        config=copy.deepcopy(self.config)
        config["profiles"]["smoke"]["planners"].remove("bitstar_a3")
        with self.assertRaisesRegex(ValueError,"retain the original"):
            self.load_mutation(config)

    def test_unsafe_labels_duplicates_and_unused_variants_are_rejected(self):
        for label in ("../outside","nested/label","UPPER","a"*81):
            config=copy.deepcopy(self.config)
            config["planner_variants"][label]=config["planner_variants"].pop("a3_cache")
            with self.assertRaises(ValueError):self.load_mutation(config)
        config=copy.deepcopy(self.config)
        config["profiles"]["smoke"]["planners"].append("a3_cache")
        with self.assertRaises(ValueError):self.load_mutation(config)
        config=copy.deepcopy(self.config)
        config["profiles"]["smoke"]["planners"].remove("a3_cache")
        with self.assertRaises(ValueError):self.load_mutation(config)

    def test_noop_variants_and_unsupported_batch_schedules_are_rejected(self):
        for definition in ({"planner_id":"bitstar_a3"},
                           {"planner_id":"bitstar_a3","state_cache":False},
                           {"planner_id":"rrtstar","state_cache":True,"samples_per_batch":50},
                           {"planner_id":"abitstar","batch_after_first_solution":200},
                           {"planner_id":"bitstar_a3","samples_per_batch":50,"batch_after_first_solution":50}):
            with self.subTest(definition=definition):
                config=copy.deepcopy(self.config)
                config["planner_variants"]["a3_cache"]=definition
                with self.assertRaises(ValueError):self.load_mutation(config)

    def test_shared_infrastructure_is_available_to_comparator_backends(self):
        for backend in ("bitstar_a0","rrtstar","informed_rrtstar","abitstar","aitstar","lazyprmstar_hauser2015"):
            with self.subTest(backend=backend):
                config=copy.deepcopy(self.config)
                config["planner_variants"]["comparator_optimized"]={"planner_id":backend,
                    "state_cache":True,"scalar_cost":True,"collision_index":True}
                config["profiles"]["smoke"]["planners"].append("comparator_optimized")
                loaded=self.load_mutation(config)
                request=request_for(dict(self.entry,planner_id="comparator_optimized"),loaded,"smoke")
                self.assertEqual(request["planner_id"],backend)
                self.assertTrue(all(request[key] for key in ("state_cache","scalar_cost","collision_index")))

    def test_worker_must_confirm_variant_identity_and_effective_options(self):
        entry=dict(self.entry,planner_id="a3_optimized_two_phase50to200")
        record={"planner_id":entry["planner_id"],"variant_id":entry["planner_id"],
                "backend_planner_id":"bitstar_a3","effective_policy":{
                    "state_cache":True,"scalar_cost":True,"collision_index":True,
                    "samples_per_batch":50,"batch_after_first_solution":200,
                    "batch_policy":"after_first_solution"}}
        verify_variant_result(entry,self.config,record)
        for key,value in (("state_cache",False),("scalar_cost",False),("collision_index",False),
                          ("samples_per_batch",200),("batch_after_first_solution",0),
                          ("batch_policy","fixed")):
            changed=copy.deepcopy(record);changed["effective_policy"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_variant_result(entry,self.config,changed)
        for key in ("planner_id","variant_id","backend_planner_id"):
            changed=copy.deepcopy(record);changed[key]="wrong_label"
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_variant_result(entry,self.config,changed)
        changed=copy.deepcopy(record);changed.pop("effective_policy")
        changed.update(termination_reason="INVALID_QUERY",planner_status="INVALID_START")
        verify_variant_result(entry,self.config,changed)
        # Legacy records have no optimization metadata and remain supported.
        verify_variant_result(self.entry,self.config,{"planner_id":"bitstar_a3"})

    def test_analysis_keeps_variant_labels_and_underlying_sampler_semantics(self):
        from test_analysis import fixture
        from analyze_results import analyze
        with tempfile.TemporaryDirectory(prefix="a3_variant_parser_",dir=ROOT/"tests/fixtures") as directory:
            source=fixture(Path(directory))
            config=read_json(source/"resolved_config.json")
            config["planner_variants"]={"rrtstar_optimized":{"planner_id":"rrtstar"}}
            dump(source/"resolved_config.json",config)
            manifest=read_json(source/"manifest.json")
            for entry in manifest["trials"]:
                if entry["planner_id"]!="rrtstar":continue
                entry["planner_id"]="rrtstar_optimized"
                trial_path=source/entry["trial_id"]/"trial.json"
                record=read_json(trial_path);record["planner_id"]="rrtstar_optimized"
                dump(trial_path,record)
            dump(source/"manifest.json",manifest)
            output=Path(directory)/"report"
            summary=analyze(source,output,bootstrap_replicates=10,make_plots=False)
            self.assertIn("rrtstar_optimized",summary["planner_ids"])
            self.assertNotIn("rrtstar",summary["planner_ids"])
            records=read_json(output/"trial_metrics.json")
            aliases=[row for row in records if row["planner_id"]=="rrtstar_optimized"]
            self.assertEqual(len(aliases),2)
            for row in aliases:
                self.assertIsNone(row["custom_sampler_acceptance_all_phases"]["value"])
                self.assertEqual(row["informed_acceptance"]["reason"],"not_applicable_non_informed_planner")

    def test_smoke_schema_rejects_pilot_full_and_legacy_schema_extension(self):
        for profile in ("pilot","full"):
            with self.assertRaises(ValueError):load_config(CONFIG,profile)
        config=copy.deepcopy(self.config)
        config["schema_version"]=1
        with self.assertRaises(ValueError):self.load_mutation(config)
        config=copy.deepcopy(self.config)
        config["profiles"]["pilot"]=copy.deepcopy(config["profiles"]["smoke"])
        with self.assertRaises(ValueError):self.load_mutation(config)


if __name__=="__main__":unittest.main()
