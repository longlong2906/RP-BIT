"""Schema/transport and independent certificate correctness; no planner runs."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from a3_traversability_perf_protocol import FAMILIES, PLANNERS, PROTOCOL_ID, generate_maps, validate_config
from experiment_common import digest, load_config, resolved_planner_options
from flat_direct_certificate import reconstruct, verify_claim, POLICY, TERMINATION
from independent_verifier import Oracle
from run_experiments import plan, request_for, verify_variant_result, verify_resume_identity
from terrain_io import write_map


class PerformanceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config(ROOT/"configs/a3_traversability_perf_smoke.yaml")
        cls.maps={"config_hash":digest(cls.config),"profile":"smoke","protocol_id":PROTOCOL_ID,
            "maps":[dict(family=f,map_id=f+"_101",map_seed=101,path="synthetic_unopened.t25d",sha256="fixture",
                         queries=[dict(query_id="q0",start=[2.,10.],goal=[18.,10.])]) for f in FAMILIES]}
        cls.entries=plan(cls.config,"smoke",cls.maps)

    def test_exact_36_arm_design_and_transport(self):
        self.assertEqual(len(self.entries),36)
        self.assertEqual({e["planner_id"] for e in self.entries},set(PLANNERS))
        for entry in self.entries:
            request=request_for(entry,self.config,"smoke")
            expected=resolved_planner_options(self.config,entry["planner_id"])
            for key,value in expected.items():self.assertEqual(request[key],value)
            self.assertEqual(request["traversability"]["lambda"],1.)
            self.assertTrue(all(request[k] for k in ("state_cache","scalar_cost","collision_index")))
            self.assertEqual(entry["resolved_optimizations_sha256"],digest(expected))

    def test_new_generator_passes_version_without_changing_geometry(self):
        with patch("a3_traversability_perf_protocol._generate_maps",return_value={"stub":True}) as generate:
            self.assertEqual(generate_maps(self.config,"smoke","unused"),{"stub":True})
            self.assertEqual(generate.call_args.kwargs,{"schema_version":9,"protocol_id":PROTOCOL_ID,"generator_version":"9.0.0"})

    def test_defaults_and_policy_types_are_strict(self):
        for key,value in (("sampler_query_cache",1),("sampler_query_cache",True),
                          ("traversability_build_policy","square"),("exact_flat_certificate",True)):
            changed=copy.deepcopy(self.config);changed["algorithm"][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_config(changed)
        changed=copy.deepcopy(self.config);changed["planner_variants"][PLANNERS[0]]["sampler_query_cache"]=0
        with self.assertRaises(ValueError):validate_config(changed)
        for profile in ("pilot","full"):
            with self.assertRaisesRegex(ValueError,"smoke only"):validate_config(self.config,profile)

    def test_policy_changes_invalidate_transport_and_resume(self):
        entry=self.entries[0]
        for key,value in (("sampler_query_cache",True),("traversability_build_policy","skip_dominated"),
                          ("exact_flat_certificate",True)):
            changed=copy.deepcopy(self.config)
            changed["planner_variants"][entry["planner_id"]][key]=not entry["resolved_optimizations"][key] if type(value) is bool else (
                "legacy" if entry["resolved_optimizations"][key]==value else value)
            with self.assertRaisesRegex(ValueError,"changed after trial planning"):
                request_for(entry,changed,"smoke")
            with self.assertRaisesRegex(ValueError,"resume identity mismatch"):
                verify_resume_identity({"config_hash":digest(self.config)},{"config_hash":digest(changed)})

    def test_effective_policy_and_certificate_provenance_required(self):
        entry=self.entries[0];request=request_for(entry,self.config,"smoke")
        options=entry["resolved_optimizations"]
        result=dict(planner_id=entry["planner_id"],variant_id=entry["planner_id"],backend_planner_id="bitstar_a3",
            effective_policy=dict(options,batch_policy="fixed"),objective_type=request["objective_type"],
            requested_traversability=request["traversability"],effective_traversability=request["traversability"],
            exact_certificate=dict(policy=POLICY,enabled=options["exact_flat_certificate"]))
        verify_variant_result(entry,self.config,result)
        for key in ("sampler_query_cache","traversability_build_policy","exact_flat_certificate"):
            changed=copy.deepcopy(result);del changed["effective_policy"][key]
            with self.assertRaisesRegex(ValueError,"worker did not apply"):verify_variant_result(entry,self.config,changed)
        del result["exact_certificate"]
        with self.assertRaisesRegex(ValueError,"certificate policy/provenance"):verify_variant_result(entry,self.config,result)

    def test_v8_defaults_and_request_shape_unchanged(self):
        old=load_config(ROOT/"configs/a3_traversability_smoke.yaml")
        opts=resolved_planner_options(old,"a3_t_lambda1_optimized")
        for key in ("sampler_query_cache","traversability_build_policy","exact_flat_certificate"):
            self.assertNotIn(key,opts)
        self.assertEqual(old["schema_version"],8)

    def test_microbenchmark_plan_is_exactly_54_cold_builds(self):
        from benchmark_a3_preprocessing import build_plan
        with patch("benchmark_a3_preprocessing.write_map",return_value={"data_sha256":"synthetic"}):
            entries=build_plan(ROOT/"tests/fixtures/synthetic_micro_plan_not_written")
        self.assertEqual(len(entries),54)
        self.assertEqual(len({entry["run_id"] for entry in entries}),54)
        self.assertEqual({entry["spacing_m"] for entry in entries},{.4,.2,.1})
        self.assertEqual({entry["repetition"] for entry in entries},{0,1,2})

    def test_replay_rejects_changed_trace_instead_of_reporting_speed(self):
        from benchmark_a3_sampler import verify_replay
        document=dict(kind="sampler_fixed_work_replay",seed=11,calls_per_block=10000,
                      clock_policy="steady_clock_per_block",rows=[])
        for name in ("lower_bound_guard","near_empty_flat","near_empty_hill","ellipse","full_domain"):
            for cache in (False,True):
                document["rows"].append(dict(case_id=name,sampler_query_cache=cache,calls=10000,elapsed_s=.01,
                    trace_hash="synthetic",sample_proposals=0,samples_returned=0,rejected_bounds=0,
                    rejected_nodata=0,rejected_heuristic=0,exhausted_calls=10000,
                    sampler_invalid_interval_calls=0,sampler_lower_bound_rejected_calls=10000,
                    sampler_proposal_limit_calls=0))
        self.assertEqual(len(verify_replay(document)),5)
        document["rows"][1]["trace_hash"]="different"
        with self.assertRaisesRegex(ValueError,"replay changed"):verify_replay(document)


class IndependentCertificateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="certificate_oracle_",dir=ROOT/"tests/fixtures")
        self.path=Path(self.temp.name)/"map.t25d"
        self.start=[1.,4.];self.goal=[7.,4.];self.xy=[self.start,self.goal]
        self.config=dict(feature_version="cell_envelope_v1",slope_weight=.5,clearance_weight=.5,
                         clearance_comfort_m=.4,**{"lambda":1.})
    def tearDown(self):self.temp.cleanup()
    def oracle(self,heights=None,rectangles=(),mask=None):
        write_map(self.path,heights or [[2.]*9 for _ in range(9)],rectangles=rectangles,mask=mask)
        return Oracle(self.path)
    def check(self,oracle,xy=None,config=None):
        return reconstruct(oracle,xy or self.xy,self.start,self.goal,config or self.config)

    def test_accepts_flat_direct_lambda_zero_one_four_and_reverse(self):
        oracle=self.oracle()
        for weight in (0.,1.,4.):
            self.assertTrue(self.check(oracle,config=dict(self.config,**{"lambda":weight}))["certified"])
        self.assertTrue(self.check(oracle,list(reversed(self.xy)))["certified"])

    def test_rejects_ulp_height_missing_data_multiple_waypoints_and_contact(self):
        for value in (math.nextafter(2.,math.inf),None):
            heights=[[2.]*9 for _ in range(9)];heights[0][0]=value
            self.assertFalse(self.check(self.oracle(heights))["certified"])
        self.assertFalse(self.check(self.oracle(),[self.start,[4.,4.],self.goal])["certified"])
        # Hard invalid remains invalid even at lambda=0.
        oracle=self.oracle(rectangles=[(3.99,3.8,4.01,4.2)])
        self.assertFalse(self.check(oracle,config=dict(self.config,**{"lambda":0.}))["certified"])
        mask=[[0]*8 for _ in range(8)];mask[4][4]=1
        self.assertFalse(self.check(self.oracle(mask=mask))["certified"])

    def test_active_clearance_rejected_even_if_far_or_product_underflows(self):
        oracle=self.oracle(rectangles=[(.1,.1,.2,.2)])
        self.assertFalse(self.check(oracle)["certified"])
        tiny=dict(self.config,clearance_weight=1e-320,**{"lambda":1e-320})
        self.assertEqual(tiny["lambda"]*tiny["clearance_weight"],0.)
        self.assertFalse(self.check(oracle,config=tiny)["certified"])
        self.assertTrue(self.check(oracle,config=dict(self.config,clearance_weight=0.))["certified"])

    def test_provenance_and_raw_reconstruction_gate_claim(self):
        oracle=self.oracle();proof=self.check(oracle)
        request=dict(exact_flat_certificate=True,start=self.start,goal=self.goal,traversability=self.config)
        certificate=dict(policy=POLICY,enabled=True,certified=True,snapshot_id="snapshot_0",check_wall_s=.001,
                         checks=1,conditions=proof["conditions"],triggered_termination=True)
        trial=dict(exact_certificate=certificate,termination_reason=TERMINATION)
        snapshots={"snapshot_0":dict(valid=True,path_xy=self.xy)}
        self.assertTrue(verify_claim(oracle,request,trial,snapshots)["valid"])
        for key,value in (("snapshot_id","missing"),("snapshot_id",[]),("checks",0),("check_wall_s",float("nan")),("enabled",False),
                          ("triggered_termination",False),("triggered_termination",None)):
            changed=copy.deepcopy(trial);changed["exact_certificate"][key]=value
            self.assertFalse(verify_claim(oracle,request,changed,snapshots)["valid"])
        wrong=copy.deepcopy(snapshots);wrong["snapshot_0"]["path_xy"]=[self.start,[4.,4.],self.goal]
        self.assertFalse(verify_claim(oracle,request,trial,wrong)["valid"])
        trial["termination_reason"]="TIME_BUDGET"
        self.assertFalse(verify_claim(oracle,request,trial,snapshots)["valid"])

    def test_missing_snapshot_retains_invalid_trial_with_certificate_option(self):
        from experiment_common import dump, file_hash
        from validate_results import validate_trial
        oracle=self.oracle();folder=Path(self.temp.name)/"missing_snapshot"
        request=dict(map_path=str(self.path),start=self.start,goal=self.goal,traversability=self.config,
                     exact_flat_certificate=True)
        trial=dict(map_hash=file_hash(self.path),path_available=False,execution_status="COMPLETED",
            planner_status="TIMEOUT",time_budget_s=2.,objective_type="surface_length_traversability",
            requested_traversability=self.config,effective_traversability=self.config,
            termination_reason="TIME_BUDGET",exact_certificate=dict(policy=POLICY,enabled=True,
                certified=False,checks=1,check_wall_s=.001,triggered_termination=False))
        dump(folder/"request.json",request);dump(folder/"trial.json",trial)
        (folder/"progress.jsonl").write_text('{"event_type":"improvement","snapshot_id":"missing","best_cost_terrain_m":6,"objective_value":6,"elapsed_s":0.01}\n')
        result=validate_trial(folder,oracle)
        self.assertFalse(result["success"])
        self.assertFalse(result["snapshots_valid"])


if __name__=="__main__":unittest.main()
