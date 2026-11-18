"""Runner protocol and independent spatial-index regression; no experimental trials."""
import copy
import datetime as dt
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from experiment_common import ROOT,digest,dump,load_config,read_json
from generate_scenarios import generate_advanced,generate_sem_comparison,pilot_geometry
from independent_verifier import Oracle
from run_advanced_campaign import (_report_artifacts,CampaignClock,campaign_counts,collection_block_allowed,
                                   enforce_final_deadline,final_campaign_status,freeze_raw_artifacts,
                                   load_resume,planner_blocks,raw_artifact_entries,recovered_elapsed,
                                   require_disjoint_output_trees,require_explicit_allow,
                                   retain_interrupted_attempt,run_analysis_with_deadline,
                                   run_campaign,stage_elapsed,validate_until_checkpoint,verify_complete_campaign,verify_dataset,
                                   verify_raw_artifacts)
from run_experiments import (ADVANCED_PLANNERS,TRIAL_ORDER_LATIN,TRIAL_ORDER_PAIRED,
                             TRIAL_ORDER_SEM_COMPARISON,audit_latin_plan,
                             audit_paired_plan,audit_sem_comparison_plan,
                             bitstar_log_counters,plan,request_for,summarize,
                             verify_resume_identity)
from run_sem_comparison_campaign import require_explicit_allow as require_sem_explicit_allow
from terrain_io import write_map


class RunnerTests(unittest.TestCase):
    def setUp(self):self.config=load_config(ROOT/"configs/experiments.yaml","pilot")

    def manifest_fixture(self):
        # In-memory schema-only fixture, never written as observed results.
        maps=[]
        for family in self.config["profiles"]["pilot"]["families"]:
            for seed in (201,202):
                _,_,queries,_=pilot_geometry(family,seed,40,40)
                maps.append({"map_id":f"{family}_{seed}","family":family,"map_seed":seed,"path":"fixture_only", "sha256":"fixture_only",
                    "queries":[{"query_id":qid,"start":start,"goal":goal,"feasibility":"feasible_witness_validated",
                        "reference":{"validation":{"valid":True}}} for qid,start,goal,_,_ in queries]})
        return {"maps":maps}

    def write_advanced_report_fixture(self,report,raw,status="PARTIAL"):
        report=Path(report);report.mkdir(parents=True,exist_ok=True)
        inventory=read_json(Path(raw)/"raw_artifact_inventory.json")
        from experiment_common import file_hash
        dump(report/"summary.json",{"raw_manifest_sha256":file_hash(Path(raw)/"manifest.json"),
             "raw_artifact_inventory_sha256":inventory["inventory_sha256"],"status":status,"figures":[]})
        for name in ("paired_bootstrap.json","anytime_curves.json","trial_metrics.json","reference_targets.json"):
            dump(report/name,[])
        (report/"REPORT.md").write_text("fixture\n",encoding="utf-8")
        (report/"trials.csv").write_text("trial_id\n",encoding="utf-8")
        return _report_artifacts(report,file_hash(Path(raw)/"manifest.json"),inventory["inventory_sha256"])

    def test_locked_plan_and_budget(self):
        source=self.manifest_fixture();first=plan(self.config,"pilot",source);second=plan(self.config,"pilot",source)
        self.assertEqual(first,second);self.assertEqual(len(first),400)
        self.assertEqual(len({t["trial_id"] for t in first}),400)
        for trial in first:self.assertEqual(request_for(trial,self.config,"pilot")["planning_budget_s"],5.)
        by_problem={}
        for trial in first:by_problem.setdefault((trial["map_id"],trial["query_id"],trial["planner_seed"]),[]).append(trial)
        self.assertEqual(len(by_problem),100)
        for group in by_problem.values():
            self.assertEqual(len(group),4);self.assertEqual(len({g["query_hash"] for g in group}),1)
        bad=copy.deepcopy(source);bad["maps"][0]["queries"].pop()
        with self.assertRaises(ValueError):plan(self.config,"pilot",bad)
        with self.assertRaises(ValueError):load_config(ROOT/"configs/experiments.yaml","full")

    def test_resume_requires_every_identity_component(self):
        identity={"config_hash":"a","code_sha256":"b","worker_sha256":"c","map_hashes":{"m":"d"},
            "map_manifest_sha256":"e","dependency_lock_sha256":"f","dependency_archives":{"a":"g"},
            "patch_sha256":{"p":"h"},"ompl_library_sha256":"i"}
        verify_resume_identity(identity,copy.deepcopy(identity))
        for key in identity:
            changed=copy.deepcopy(identity);changed[key]="changed"
            with self.assertRaises(ValueError,msg=key):verify_resume_identity(identity,changed)

    def test_limited_paired_plan_keeps_a0_a3_on_the_same_problem(self):
        source=self.manifest_fixture()
        first=plan(self.config,"pilot",source,order_policy=TRIAL_ORDER_PAIRED,max_problem_slots=50)
        second=plan(self.config,"pilot",source,order_policy=TRIAL_ORDER_PAIRED,max_problem_slots=50)
        self.assertEqual(first,second)
        self.assertEqual(len(first),200)
        audit=audit_paired_plan(first,160)
        self.assertEqual(audit["problem_slots"],50)
        self.assertEqual(audit["problem_slots_per_block"],[40,10])
        self.assertEqual((audit["a0_first"],audit["a3_first"]),(25,25))
        family_slots={family:set() for family in self.config["profiles"]["pilot"]["families"]}
        for offset in range(0,len(first),4):
            group=first[offset:offset+4]
            pair=[entry for entry in group if entry["planner_id"] in ("bitstar_a0","bitstar_a3")]
            self.assertEqual(len(pair),2)
            self.assertEqual(abs(group.index(pair[0])-group.index(pair[1])),1)
            keys={(entry["map_id"],entry["query_id"],entry["planner_seed"],entry["query_hash"],
                  tuple(entry["start"]),tuple(entry["goal"])) for entry in pair}
            self.assertEqual(len(keys),1)
            family_slots[group[0]["family"]].add(group[0]["problem_slot_id"])
        self.assertTrue(all(len(slots)==10 for slots in family_slots.values()))
        for family in family_slots:
            maps={entry["map_id"] for entry in first if entry["family"]==family}
            self.assertEqual(len(maps),2)

    def test_limited_paired_plan_rejects_incomplete_policy_inputs(self):
        source=self.manifest_fixture()
        bad=copy.deepcopy(self.config)
        bad["profiles"]["pilot"]["planners"]=["bitstar_a0","bitstar_a3"]
        with self.assertRaisesRegex(ValueError,"four registered planners"):
            plan(bad,"pilot",source,order_policy=TRIAL_ORDER_PAIRED,max_problem_slots=50)
        with self.assertRaisesRegex(ValueError,"requires paired"):
            plan(self.config,"pilot",source,max_problem_slots=50)

    def test_locked_advanced_config_and_latin_square_plan(self):
        config=load_config(ROOT/"configs/a3_vs_advanced_v1.yaml","advanced")
        settings=config["profiles"]["advanced"]
        maps=[]
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                _,_,definitions,_=pilot_geometry(family,seed,40,40)
                qid,start,goal,_,label=next(item for item in definitions if item[0]=="q0")
                maps.append({"map_id":f"{family}_{seed}","family":family,"map_seed":seed,
                    "path":"fixture_only","sha256":f"fixture_{family}_{seed}","queries":[{
                    "query_id":qid,"start":start,"goal":goal,"geometry_label":label,
                    "feasibility":"feasible_witness_validated","reference":{"validation":{"valid":True}}}]})
        manifest={"maps":maps}
        first=plan(config,"advanced",manifest,order_policy=TRIAL_ORDER_LATIN)
        second=plan(config,"advanced",manifest,order_policy=TRIAL_ORDER_LATIN)
        self.assertEqual(first,second);self.assertEqual(len(first),400)
        audit=audit_latin_plan(first)
        self.assertEqual(audit["problem_slots"],100)
        self.assertEqual(set(audit["position_counts"]),set(ADVANCED_PLANNERS))
        self.assertTrue(all(counts==[25,25,25,25] for counts in audit["position_counts"].values()))
        self.assertEqual(len(planner_blocks(first)),100)
        self.assertEqual(first,plan(config,"advanced",manifest,order_policy=TRIAL_ORDER_LATIN))
        with self.assertRaisesRegex(ValueError,"does not permit"):
            plan(config,"advanced",manifest,order_policy=TRIAL_ORDER_LATIN,max_problem_slots=1)
        bad=copy.deepcopy(config);bad["profiles"]["advanced"]["map_seeds"][-1]=9999
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.yaml";path.write_text(yaml.safe_dump(bad),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"locked 400-trial protocol"):
                load_config(path,"advanced")
        with self.assertRaisesRegex(ValueError,"explicit --allow-advanced-benchmark"):
            require_explicit_allow(False)
        require_explicit_allow(True)

    def test_locked_sem_comparison_config_and_parity_plan(self):
        config=load_config(ROOT/"configs/a3_vs_sem_rrtstar_common_v1.yaml","sem_comparison")
        settings=config["profiles"]["sem_comparison"];maps=[]
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                _,_,definitions,_=pilot_geometry(family,seed,40,40)
                qid,start,goal,_,label=next(item for item in definitions if item[0]=="q0")
                maps.append({"map_id":f"{family}_{seed}","family":family,"map_seed":seed,
                    "path":"fixture_only","sha256":f"fixture_{family}_{seed}","queries":[{
                    "query_id":qid,"start":start,"goal":goal,"geometry_label":label,
                    "feasibility":"feasible_witness_validated","reference":{"validation":{"valid":True}}}]})
        manifest={"maps":maps}
        first=plan(config,"sem_comparison",manifest,order_policy=TRIAL_ORDER_SEM_COMPARISON)
        second=plan(config,"sem_comparison",manifest,order_policy=TRIAL_ORDER_SEM_COMPARISON)
        self.assertEqual(first,second);self.assertEqual(len(first),300)
        audit=audit_sem_comparison_plan(first,settings)
        self.assertEqual(audit["problem_slots"],150)
        self.assertEqual(audit["first_position_counts"],
                         {"bitstar_a3":75,"sem_rrtstar_common_v1":75})
        self.assertTrue(all(values=={"bitstar_a3":15,"sem_rrtstar_common_v1":15}
                            for values in audit["family_first_position_counts"].values()))
        self.assertTrue(all(values=={"bitstar_a3":25,"sem_rrtstar_common_v1":25}
                            for values in audit["seed_first_position_counts"].values()))
        self.assertEqual(set(audit["seed_first_position_counts"]),
                         {str(seed) for seed in settings["planner_seeds"]})
        self.assertEqual(audit,json.loads(json.dumps(audit)))
        self.assertEqual(len(planner_blocks(first,2)),150)
        sem_trial=next(entry for entry in first if entry["planner_id"]=="sem_rrtstar_common_v1")
        request=request_for(sem_trial,config,"sem_comparison")
        self.assertEqual((request["planning_budget_s"],request["sem_range_m"],
                          request["sem_neighbor_radius_m"]),(10.,1.,3.))
        with self.assertRaisesRegex(ValueError,"does not permit"):
            plan(config,"sem_comparison",manifest,order_policy=TRIAL_ORDER_SEM_COMPARISON,
                 max_problem_slots=1)
        bad=copy.deepcopy(config);bad["profiles"]["sem_comparison"]["planner_seeds"][-1]=99999
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.yaml";path.write_text(yaml.safe_dump(bad),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"locked 300-trial protocol"):
                load_config(path,"sem_comparison")
        with self.assertRaisesRegex(ValueError,"explicit --allow-sem-comparison"):
            require_sem_explicit_allow(False)
        require_sem_explicit_allow(True)

    def test_atomic_dump_retries_only_transient_permission_error(self):
        import experiment_common
        real_replace=experiment_common.os.replace
        calls=[]
        def transient(source,target):
            calls.append((source,target))
            if len(calls)==1:
                raise PermissionError(5,"transient sharing violation",str(target))
            return real_replace(source,target)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"state.json"
            with patch.object(experiment_common.os,"replace",side_effect=transient), \
                    patch.object(experiment_common.time,"sleep") as sleep:
                dump(target,{"status":"RUNNING"})
            self.assertEqual(read_json(target),{"status":"RUNNING"})
            self.assertEqual(len(calls),2)
            sleep.assert_called_once_with(.05)

    def test_atomic_dump_surfaces_persistent_permission_error(self):
        import experiment_common
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"state.json"
            with patch.object(experiment_common.os,"replace",
                              side_effect=PermissionError(5,"persistent sharing violation",str(target))), \
                    patch.object(experiment_common.time,"sleep") as sleep:
                with self.assertRaises(PermissionError):
                    dump(target,{"status":"RUNNING"})
            self.assertEqual(sleep.call_count,9)
            self.assertFalse(target.exists())

    def test_sem_generator_uses_declared_seed_once_and_never_replaces_failure(self):
        config={"profiles":{"sem_comparison":{"map_extent_m":[40.,40.],"grid_spacing_m":.5,
                    "families":["traversable_hill"],"map_seeds":[4001]}},
                "model":{"robot_radius_m":.3,"safety_margin_m":.1,"max_slope_deg":35.,
                         "state":"xy","numerical_abs_tolerance_m":1e-8,
                         "numerical_rel_tolerance":1e-8},
                "sem_comparison":{"required_query_ids":["q0"]}}
        with tempfile.TemporaryDirectory(prefix="sem_generator_") as directory:
            output=Path(directory)/"ok";output.mkdir()
            manifest=generate_sem_comparison(config,output)
            self.assertEqual((manifest["schema_version"],manifest["generator_version"]),(4,"4.0.0"))
            self.assertEqual(manifest["declared_map_seeds"],[4001])
            self.assertEqual(manifest["seed_replacements"],[])
            self.assertEqual([(item["map_seed"],item["queries"][0]["query_id"])
                              for item in manifest["maps"]],[(4001,"q0")])
            verify_dataset(config,manifest)
            corrupted=copy.deepcopy(manifest);corrupted["maps"][0]["queries"][0]["query_hash"]="changed"
            with self.assertRaisesRegex(ValueError,"map/query sidecar mismatch"):
                verify_dataset(config,corrupted)
            failed=Path(directory)/"failed";failed.mkdir()
            with patch("generate_scenarios.Oracle") as oracle:
                oracle.return_value.validate_path.return_value={"valid":False,
                    "errors":["synthetic qualification failure"]}
                with self.assertRaisesRegex(ValueError,"without replacement"):
                    generate_sem_comparison(config,failed)
            self.assertFalse((failed/"manifest.json").exists())

    def test_advanced_generator_uses_declared_seed_once_and_never_replaces_failure(self):
        config={"profiles":{"advanced":{"map_extent_m":[40.,40.],"grid_spacing_m":.5,
                    "families":["traversable_hill"],"map_seeds":[3001]}},
                "model":{"robot_radius_m":.3,"safety_margin_m":.1,"max_slope_deg":35.,
                         "state":"xy"},
                "advanced_benchmark":{"required_query_ids":["q0"]}}
        with tempfile.TemporaryDirectory(prefix="advanced_generator_") as directory:
            output=Path(directory)/"ok";output.mkdir()
            manifest=generate_advanced(config,output)
            self.assertEqual(manifest["declared_map_seeds"],[3001])
            self.assertEqual(manifest["seed_replacements"],[])
            self.assertEqual([(item["map_seed"],item["queries"][0]["query_id"]) for item in manifest["maps"]],[(3001,"q0")])
            failed=Path(directory)/"failed";failed.mkdir()
            with patch("generate_scenarios.Oracle") as oracle:
                oracle.return_value.validate_path.return_value={"valid":False,"errors":["synthetic qualification failure"]}
                with self.assertRaisesRegex(ValueError,"without replacement"):
                    generate_advanced(config,failed)
            self.assertFalse((failed/"manifest.json").exists())

    def test_advanced_dataset_recomputes_query_hash_and_witness_geometry(self):
        with tempfile.TemporaryDirectory(prefix="advanced_dataset_verify_") as directory:
            root=Path(directory);path=root/"flat_3001.t25d"
            metadata=write_map(path,[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],radius=0.,margin=0.,slope=35.,
                metadata={"map_id":"flat_3001","family":"flat","generator_version":"3.0.0",
                          "generator_seed":3001,"dataset_role":"held_out_evaluation",
                          "seed_selection":"predeclared_no_replacement_no_planner_results_used"})
            config={"profiles":{"advanced":{"families":["flat"],"map_seeds":[3001]}},
                    "advanced_benchmark":{"required_query_ids":["q0"]},
                    "model":{"state":"xy","numerical_abs_tolerance_m":1e-8,
                             "numerical_rel_tolerance":1e-8}}
            start=[.5,.5];goal=[1.5,1.5];xy=[start,goal]
            checked=Oracle(path).validate_path(xy,start,goal)
            reference={"reference_type":"validated_witness","cost_terrain_m":checked["cost_terrain_m"],
                "path_xy":xy,"path_sha256":digest(xy),"validation":checked,"witness_id":"w0",
                "map_sha256":metadata["data_sha256"],"provenance":"fixture predeclared witness"}
            query={"query_id":"q0","start":start,"goal":goal,"geometry_label":"fixture",
                "feasibility":"feasible_witness_validated",
                "witnesses":[{"witness_id":"w0","path_xy":xy,"sha256":digest(xy),"validation":checked}],
                "direct_edge_validation":checked,"reference":reference}
            query["query_hash"]=digest({"start":start,"goal":goal,"map_hash":metadata["data_sha256"],
                                        "model":config["model"]})
            sidecar=read_json(path.with_suffix(".json"));sidecar["queries"]=[query];dump(path.with_suffix(".json"),sidecar)
            maps={"schema_version":3,"profile":"advanced","generator_version":"3.0.0",
                "config_hash":digest(config),"held_out":True,"declared_map_seeds":[3001],
                "seed_replacement_policy":"forbidden","seed_replacements":[],
                "qualification_policy":"all declared family/seed maps and q0 witnesses must pass before planner execution; abort, never replace",
                "maps":[{"map_id":"flat_3001","family":"flat","map_seed":3001,"path":str(path),
                         "sha256":metadata["data_sha256"],"queries":[query]}]}
            verify_dataset(config,maps)
            tampered=copy.deepcopy(maps);tampered["maps"][0]["queries"][0]["query_hash"]="0"*64
            tampered_sidecar=copy.deepcopy(sidecar);tampered_sidecar["queries"]=tampered["maps"][0]["queries"]
            dump(path.with_suffix(".json"),tampered_sidecar)
            with self.assertRaisesRegex(ValueError,"query hash mismatch"):
                verify_dataset(config,tampered)

    def test_advanced_resume_clock_is_cumulative_and_incomplete_attempt_is_never_retried(self):
        protocol={"collection_soft_stop_s":5400.,"hard_finalize_s":7200.}
        clock=CampaignClock(5399.)
        with patch("run_advanced_campaign.time.perf_counter",side_effect=[100.,102.]):
            clock.start();self.assertEqual(clock.elapsed(),5401.)
        self.assertFalse(collection_block_allowed(5401.,protocol,False))
        self.assertTrue(collection_block_allowed(5401.,protocol,True))
        self.assertFalse(collection_block_allowed(7200.,protocol,True))
        config={"profiles":{"advanced":{"planning_budget_s":5.}},"execution":{"watchdog_grace_s":5.},
                "advanced_benchmark":{"collection_soft_stop_s":5400.,"validation_checkpoint_s":6600.,
                                      "hard_finalize_s":7200.}}
        now=dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)
        state={"status":"RUNNING","phase":"ACTIVE_TRIAL","elapsed_s":123.,"active_trial_id":"t0",
               "elapsed_before_active_trial_s":123.,"active_trial_started_utc":(now-dt.timedelta(hours=1)).isoformat(),
               "session_checkpoint_utc":(now-dt.timedelta(hours=1)).isoformat()}
        self.assertEqual(recovered_elapsed(state,config,now),(133.,10.))
        validation={"status":"RUNNING","phase":"OFFLINE_VALIDATION","elapsed_s":6000.,
                    "session_checkpoint_utc":(now-dt.timedelta(hours=1)).isoformat()}
        self.assertEqual(recovered_elapsed(validation,config,now),(6600.,600.))
        analysis={"status":"RUNNING","phase":"ANALYSIS_REPORT","elapsed_s":6700.,
                  "session_checkpoint_utc":(now-dt.timedelta(hours=1)).isoformat()}
        self.assertEqual(recovered_elapsed(analysis,config,now),(7200.,500.))
        clean=dict(analysis,status="PARTIAL")
        self.assertEqual(recovered_elapsed(clean,config,now),(6700.,0.))
        with tempfile.TemporaryDirectory(prefix="advanced_resume_") as directory:
            output=Path(directory);entry={"trial_id":"t0","planner_id":"bitstar_a3","planner_seed":91001,
                "map_id":"m","map_hash":"h","map_path":"fixture","family":"fixture","map_seed":3001,
                "query_id":"q0","query_hash":"q","start":[0.,0.],"goal":[1.,1.]}
            attempt=output/"t0";attempt.mkdir();dump(attempt/"attempt.json",{"status":"START_REQUESTED"})
            manifest={"experiment_id":"fixture","identity":{"config_hash":"c","code_sha256":"s"}}
            retained=retain_interrupted_attempt(output,entry,manifest,config)
            self.assertTrue(retained)
            trial=read_json(attempt/"trial.json")
            self.assertEqual((trial["execution_status"],trial["termination_reason"]),
                             ("CRASH","INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY"))
            self.assertFalse(retain_interrupted_attempt(output,entry,manifest,config))

    def test_report_only_resume_preserves_completed_stage_checkpoints(self):
        previous={"collection_elapsed_s":2155.55,"validation_checkpoint_elapsed_s":5090.19}
        self.assertEqual(stage_elapsed(5341.,previous,"collection_elapsed_s",True),2155.55)
        self.assertEqual(stage_elapsed(5341.,previous,"validation_checkpoint_elapsed_s",True),5090.19)
        self.assertEqual(stage_elapsed(5341.,previous,"collection_elapsed_s",False),5341.)
        self.assertEqual(stage_elapsed(5341.,{"collection_elapsed_s":float("nan")},
                                       "collection_elapsed_s",True),5341.)

    def test_advanced_resume_reuses_complete_report_and_allows_partial_report_directory(self):
        with tempfile.TemporaryDirectory(prefix="advanced_report_resume_") as directory:
            root=Path(directory);raw=root/"raw";report=root/"report";raw.mkdir();report.mkdir()
            dump(raw/"manifest.json",{"fixture":True})
            from experiment_common import file_hash
            inventory=freeze_raw_artifacts(raw)
            dump(report/"summary.json",{"raw_manifest_sha256":file_hash(raw/"manifest.json"),"status":"PARTIAL",
                                         "raw_artifact_inventory_sha256":inventory["inventory_sha256"],"figures":[]})
            for name in ("paired_bootstrap.json","anytime_curves.json","trial_metrics.json","reference_targets.json"):
                dump(report/name,[])
            (report/"REPORT.md").write_text("fixture\n",encoding="utf-8")
            (report/"trials.csv").write_text("trial_id\n",encoding="utf-8")
            _,artifacts=_report_artifacts(report,file_hash(raw/"manifest.json"),inventory["inventory_sha256"])
            dump(raw/"analysis_attempt.json",{"status":"COMPLETE","exit_code":0,
                 "raw_artifact_inventory_sha256":inventory["inventory_sha256"],
                 "report_artifact_inventory_sha256":artifacts["inventory_sha256"]})
            clock=CampaignClock(7199.5)
            result=run_analysis_with_deadline(raw,report,{"hard_finalize_s":7200.,"bootstrap_replicates":2000,
                                                         "bootstrap_seed":73132},clock,allow_existing=True)
            self.assertEqual(result["status"],"COMPLETE")
            self.assertTrue(result["reused_existing_report"])
            self.assertEqual(result["summary_status"],"PARTIAL")

    def test_raw_inventory_is_sorted_excludes_mutable_records_and_detects_tampering(self):
        with tempfile.TemporaryDirectory(prefix="advanced_inventory_") as directory:
            raw=Path(directory);dump(raw/"manifest.json",{"v":1});dump(raw/"campaign_state.json",{"v":1})
            dump(raw/"analysis_attempt.json",{"v":1});dump(raw/"trial"/"trial.json",{"v":1})
            inventory=freeze_raw_artifacts(raw)
            paths=[entry["path"] for entry in inventory["entries"]]
            self.assertEqual(paths,sorted(paths))
            self.assertEqual(paths,["manifest.json","trial/trial.json"])
            verify_raw_artifacts(raw)
            dump(raw/"campaign_state.json",{"v":2});dump(raw/"analysis_attempt.json",{"v":2})
            verify_raw_artifacts(raw)
            dump(raw/"trial"/"trial.json",{"v":2})
            with self.assertRaisesRegex(ValueError,"changed after analysis input freeze"):
                verify_raw_artifacts(raw)

    def test_analysis_insufficient_budget_always_records_attempt(self):
        with tempfile.TemporaryDirectory(prefix="advanced_analysis_budget_") as directory:
            root=Path(directory);raw=root/"raw";raw.mkdir();dump(raw/"manifest.json",{"fixture":True})
            result=run_analysis_with_deadline(raw,root/"report",
                {"hard_finalize_s":7200.,"bootstrap_replicates":2,"bootstrap_seed":1},CampaignClock(7198.))
            self.assertEqual(result["status"],"NOT_STARTED_INSUFFICIENT_HARD_BUDGET")
            self.assertEqual(read_json(raw/"analysis_attempt.json")["status"],result["status"])

    def test_offline_validation_batch_is_bounded_by_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="advanced_validation_budget_") as directory:
            raw=Path(directory);trial_dir=raw/"t0";trial_dir.mkdir()
            entry={"trial_id":"t0","map_path":"map"}
            dump(trial_dir/"trial.json",{"execution_status":"COMPLETED"})
            with patch("run_advanced_campaign.subprocess.run",
                       side_effect=subprocess.TimeoutExpired(["validator"],1.)) as launched:
                count,complete=validate_until_checkpoint(raw,{"trials":[entry]},CampaignClock(0.),10.,20.)
            self.assertEqual((count,complete),(0,False))
            self.assertLessEqual(launched.call_args.kwargs["timeout"],8.)

    def test_validation_artifact_is_not_terminal_before_atomic_trial_commit(self):
        with tempfile.TemporaryDirectory(prefix="advanced_validation_commit_") as directory:
            raw=Path(directory);trial_dir=raw/"t0";trial_dir.mkdir()
            manifest={"trials":[{"trial_id":"t0"}]}
            trial={"execution_status":"COMPLETED","success":True,"path_available":False}
            dump(trial_dir/"trial.json",trial);dump(trial_dir/"validation.json",{"valid":True})
            pending=campaign_counts(raw,manifest)
            self.assertEqual((pending["offline_validated"],pending["validation_pending"],pending["validated_success"]),(0,1,0))
            trial["validation_wall_s"]=0.25;dump(trial_dir/"trial.json",trial)
            committed=campaign_counts(raw,manifest)
            self.assertEqual((committed["offline_validated"],committed["validation_pending"],committed["validated_success"]),(1,0,1))

    def test_final_status_requires_complete_report_and_strict_hard_deadline(self):
        counts={"geometric_invalid":0,"not_run":0,"validation_pending":0}
        complete={"status":"COMPLETE","summary_status":"COMPLETE"}
        self.assertEqual(final_campaign_status(counts,complete,True),"COMPLETE")
        self.assertEqual(final_campaign_status(counts,dict(complete,summary_status="PARTIAL"),True),"PARTIAL")
        self.assertEqual(final_campaign_status(counts,complete,False),"PARTIAL")
        with tempfile.TemporaryDirectory(prefix="advanced_final_deadline_") as directory:
            state=enforce_final_deadline(directory,{"status":"COMPLETE","elapsed_s":7200.,
                                                    "hard_deadline_met":True},7200.)
            self.assertEqual(state["status"],"PARTIAL")
            self.assertFalse(state["hard_deadline_met"])

    def test_analysis_uses_fresh_staging_and_rejects_nonzero_exit(self):
        with tempfile.TemporaryDirectory(prefix="advanced_analysis_stage_") as directory:
            root=Path(directory);raw=root/"raw";raw.mkdir();dump(raw/"manifest.json",{"fixture":True})
            protocol={"hard_finalize_s":7200.,"bootstrap_replicates":2,"bootstrap_seed":1}
            def successful(command,**kwargs):
                stage=Path(command[command.index("--output")+1])
                self.write_advanced_report_fixture(stage,raw,"PARTIAL")
                return SimpleNamespace(returncode=0,stdout="ok",stderr="")
            with patch("run_advanced_campaign.subprocess.run",side_effect=successful):
                result=run_analysis_with_deadline(raw,root/"report",protocol,CampaignClock(10.))
            self.assertEqual(result["status"],"COMPLETE")
            self.assertTrue((root/"report"/"summary.json").exists())
            failed_raw=root/"failed_raw";failed_raw.mkdir();dump(failed_raw/"manifest.json",{"fixture":True})
            with patch("run_advanced_campaign.subprocess.run",
                       return_value=SimpleNamespace(returncode=7,stdout="",stderr="synthetic")):
                failed=run_analysis_with_deadline(failed_raw,root/"failed_report",protocol,CampaignClock(10.))
            self.assertEqual(failed["status"],"FAILED_NONZERO_EXIT")
            self.assertFalse((root/"failed_report").exists())

    def test_sem_protocol_dispatches_to_locked_common_objective_analyzer(self):
        with tempfile.TemporaryDirectory(prefix="sem_analysis_dispatch_") as directory:
            root=Path(directory);raw=root/"raw";raw.mkdir();dump(raw/"manifest.json",{"fixture":True})
            protocol={"protocol_id":"a3_vs_sem_rrtstar_common_v1","hard_finalize_s":7200.,
                      "bootstrap_replicates":2000,"bootstrap_seed":73135}
            with patch("run_advanced_campaign.subprocess.run",
                       return_value=SimpleNamespace(returncode=7,stdout="",stderr="synthetic")) as launched:
                result=run_analysis_with_deadline(raw,root/"report",protocol,CampaignClock(10.))
            command=launched.call_args.args[0]
            self.assertEqual(result["status"],"FAILED_NONZERO_EXIT")
            self.assertIn(str(ROOT/"tools/analyze_a3_sem_rrtstar_common.py"),command)
            self.assertNotIn(str(ROOT/"tools/analyze_results.py"),command)
            self.assertNotIn("--bootstrap-replicates",command)
            self.assertNotIn("--bootstrap-seed",command)

    def test_sem_report_artifacts_use_protocol_specific_schema(self):
        with tempfile.TemporaryDirectory(prefix="sem_report_schema_") as directory:
            report=Path(directory);manifest_hash="m"*64;inventory_hash="i"*64
            dump(report/"summary.json",{"protocol_id":"a3_vs_sem_rrtstar_common_v1",
                 "raw_manifest_sha256":manifest_hash,"raw_artifact_inventory_sha256":inventory_hash,
                 "status":"PARTIAL","figures":[]})
            dump(report/"paired_bootstrap.json",[]);dump(report/"trial_metrics.json",[])
            dump(report/"checkpoints.json",{});dump(report/"metric_coverage.json",{})
            dump(report/"descriptive_metrics.json",{})
            (report/"trials.csv").write_text("trial_id\n",encoding="utf-8")
            (report/"REPORT.md").write_text("synthetic schema fixture\n",encoding="utf-8")
            summary,artifacts=_report_artifacts(report,manifest_hash,inventory_hash)
            self.assertEqual(summary["protocol_id"],"a3_vs_sem_rrtstar_common_v1")
            self.assertTrue(artifacts["inventory_sha256"])
            self.assertFalse((report/"anytime_curves.json").exists())
            self.assertFalse((report/"reference_targets.json").exists())

    def test_raw_and_report_trees_must_not_be_ancestors(self):
        with tempfile.TemporaryDirectory(prefix="advanced_paths_") as directory:
            root=Path(directory);raw=root/"raw";report=root/"report"
            require_disjoint_output_trees(raw,report)
            for first,second in ((raw,raw),(raw,raw/"report"),(raw/"nested",raw)):
                with self.assertRaisesRegex(ValueError,"non-nested"):
                    require_disjoint_output_trees(first,second)

    def test_complete_resume_revalidates_raw_and_report_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="advanced_complete_recheck_") as directory:
            root=Path(directory);raw=root/"raw";report=root/"report";raw.mkdir()
            manifest={"trials":[]};dump(raw/"manifest.json",manifest)
            inventory=freeze_raw_artifacts(raw)
            summary,artifacts=self.write_advanced_report_fixture(report,raw,"COMPLETE")
            dump(raw/"analysis_attempt.json",{"status":"COMPLETE",
                "raw_artifact_inventory_sha256":inventory["inventory_sha256"],
                "report_artifact_inventory_sha256":artifacts["inventory_sha256"]})
            state={"status":"COMPLETE","phase":"FINALIZED","elapsed_s":100.,
                   "hard_deadline_met":True,"deadlines_s":{"hard_finalize":7200.},
                   "raw_artifact_inventory_sha256":inventory["inventory_sha256"]}
            verify_complete_campaign(raw,report,manifest,state)
            (report/"trials.csv").write_text("tampered\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"provenance/artifacts"):
                verify_complete_campaign(raw,report,manifest,state)

    def test_advanced_resume_requires_exact_identity_scenario_and_plan(self):
        with tempfile.TemporaryDirectory(prefix="advanced_identity_") as directory:
            output=Path(directory);identity={"config_hash":"c","code_sha256":"code","worker_sha256":"worker",
                "map_hashes":{"m":"map"},"map_manifest_sha256":"maps","dependency_lock_sha256":"deps",
                "dependency_archives":{},"patch_sha256":{},"ompl_library_sha256":"ompl",
                "trial_order_policy":TRIAL_ORDER_LATIN,"schedule_sha256":"schedule",
                "plan_sha256":"plan","advanced_protocol_sha256":"protocol"}
            config={"advanced_benchmark":{"protocol_id":"a3_vs_advanced_v1"},"profiles":{"advanced":{"planning_budget_s":5.}},
                    "execution":{"watchdog_grace_s":5.}}
            maps={"fixture":True};trials=[{"trial_id":"t"}];schedule={"policy":TRIAL_ORDER_LATIN}
            stored_environment={"e":1}
            dump(output/"manifest.json",{"identity":identity,"trials":trials,"schedule":schedule,
                                          "environment_sha256":digest(stored_environment)})
            dump(output/"campaign_state.json",{"protocol_id":"a3_vs_advanced_v1","elapsed_s":42.})
            dump(output/"resolved_config.json",config);dump(output/"map_manifest.json",maps);dump(output/"environment.json",stored_environment)
            (output/"source_at_run.zip").write_bytes(b"fixture archive")
            from experiment_common import file_hash
            dump(output/"source_manifest.json",{"archive_sha256":file_hash(output/"source_at_run.zip")})
            resolved=(config,maps,trials,schedule,identity)
            with patch("run_advanced_campaign.resolve_campaign_inputs",return_value=resolved), \
                 patch("run_advanced_campaign.environment",return_value=stored_environment):
                loaded=load_resume("config","maps",output,"worker")
            self.assertEqual(loaded[-2:],(42.,0.))
            with patch("run_advanced_campaign.resolve_campaign_inputs",return_value=resolved), \
                 patch("run_advanced_campaign.environment",return_value={"e":2}):
                with self.assertRaisesRegex(ValueError,"environment drift"):
                    load_resume("config","maps",output,"worker")
            changed=copy.deepcopy(identity);changed["plan_sha256"]="changed"
            with patch("run_advanced_campaign.resolve_campaign_inputs",return_value=(config,maps,trials,schedule,changed)), \
                 patch("run_advanced_campaign.environment",return_value=stored_environment):
                with self.assertRaisesRegex(ValueError,"resume identity mismatch"):
                    load_resume("config","maps",output,"worker")

    def test_timed_campaign_finishes_started_latin_block_at_soft_stop(self):
        with tempfile.TemporaryDirectory(prefix="advanced_timed_") as directory:
            root=Path(directory);output=root/"raw";report=root/"report";report.mkdir()
            planners=list(ADVANCED_PLANNERS);entries=[]
            for block in range(2):
                for index,planner in enumerate(planners):
                    entries.append({"trial_id":f"b{block}_{planner}","problem_slot_id":f"b{block}",
                        "shuffle_order":block*4+index,"planner_id":planner,"planner_seed":91001,
                        "map_id":f"m{block}","map_hash":"h","map_path":"fixture","family":"fixture",
                        "map_seed":3001+block,"query_id":"q0","query_hash":"q","start":[0.,0.],"goal":[1.,1.]})
            config={"advanced_benchmark":{"protocol_id":"a3_vs_advanced_v1","timer_scope":"fixture",
                        "collection_soft_stop_s":3.,"validation_checkpoint_s":10.,"hard_finalize_s":20.,
                        "bootstrap_replicates":20,"bootstrap_seed":73132},
                    "profiles":{"advanced":{"planning_budget_s":5.}},"execution":{"watchdog_grace_s":5.}}
            manifest={"experiment_id":"fixture","identity":{"config_hash":"c"},"trials":entries,
                      "history":[],"status":"PLANNED"}
            env={};now=[100.]
            def fake_prepare(*args):
                output.mkdir();dump(output/"manifest.json",manifest)
                dump(output/"campaign_state.json",{"status":"PLANNED","elapsed_s":None})
                return config,{},manifest,env
            def fake_trial(entry,cfg,target,*args,**kwargs):
                target.mkdir();now[0]+=1.
                trial=dict(entry,execution_status="COMPLETED",planner_status="NO_SOLUTION",success=False,
                           path_available=False,independent_valid=None,time_budget_s=5.)
                dump(target/"trial.json",trial);return trial
            with patch("run_advanced_campaign.prepare",side_effect=fake_prepare), \
                 patch("run_advanced_campaign.correctness_gate"), \
                 patch("run_advanced_campaign.run_trial",side_effect=fake_trial) as launched, \
                 patch("run_advanced_campaign.validate_until_checkpoint",return_value=(0,False)), \
                 patch("run_advanced_campaign.run_analysis_with_deadline",return_value={"status":"COMPLETE","summary_status":"PARTIAL"}), \
                 patch("run_advanced_campaign.time.perf_counter",side_effect=lambda:now[0]):
                result=run_campaign("config","maps",output,report,"worker")
            self.assertEqual(result,2)
            self.assertEqual(launched.call_count,4)
            state=read_json(output/"campaign_state.json")
            self.assertEqual(state["completed_blocks"],1)
            self.assertEqual(state["counts"]["not_run"],4)

    def test_timed_sem_campaign_finishes_started_pair_at_soft_stop(self):
        with tempfile.TemporaryDirectory(prefix="sem_timed_") as directory:
            root=Path(directory);output=root/"raw";report=root/"report";report.mkdir()
            entries=[]
            for block in range(2):
                for index,planner in enumerate(("bitstar_a3","sem_rrtstar_common_v1")):
                    entries.append({"trial_id":f"b{block}_{planner}","problem_slot_id":f"b{block}",
                        "shuffle_order":block*2+index,"planner_id":planner,"planner_seed":92001,
                        "map_id":f"m{block}","map_hash":"h","map_path":"fixture","family":"fixture",
                        "map_seed":4001+block,"query_id":"q0","query_hash":"q",
                        "start":[0.,0.],"goal":[1.,1.]})
            config={"sem_comparison":{"protocol_id":"fixture","timer_scope":"fixture",
                        "collection_soft_stop_s":1.5,"validation_checkpoint_s":20.,
                        "hard_finalize_s":50.,"bootstrap_replicates":2,"bootstrap_seed":73135,
                        "planner_block_size":2},
                    "profiles":{"sem_comparison":{"planning_budget_s":10.}},
                    "execution":{"watchdog_grace_s":5.}}
            manifest={"experiment_id":"fixture","identity":{"config_hash":"c"},"trials":entries,
                      "history":[],"status":"PLANNED"}
            now=[100.];observed_profiles=[]
            def fake_prepare(*args):
                output.mkdir();dump(output/"manifest.json",manifest)
                dump(output/"campaign_state.json",{"status":"PLANNED","elapsed_s":None})
                return config,{},manifest,{}
            def fake_trial(entry,cfg,target,*args,**kwargs):
                observed_profiles.append(kwargs.get("profile"));target.mkdir();now[0]+=1.
                trial=dict(entry,execution_status="COMPLETED",planner_status="NO_SOLUTION",
                           success=False,path_available=False,independent_valid=None,time_budget_s=10.)
                dump(target/"trial.json",trial);return trial
            with patch("run_advanced_campaign.prepare",side_effect=fake_prepare), \
                 patch("run_advanced_campaign.correctness_gate"), \
                 patch("run_advanced_campaign.run_trial",side_effect=fake_trial) as launched, \
                 patch("run_advanced_campaign.validate_until_checkpoint",return_value=(0,False)), \
                 patch("run_advanced_campaign.run_analysis_with_deadline",
                       return_value={"status":"COMPLETE","summary_status":"PARTIAL"}), \
                 patch("run_advanced_campaign.time.perf_counter",side_effect=lambda:now[0]):
                result=run_campaign("config","maps",output,report,"worker",
                                    campaign_kind="sem_comparison")
            self.assertEqual(result,2);self.assertEqual(launched.call_count,2)
            self.assertEqual(observed_profiles,["sem_comparison","sem_comparison"])
            state=read_json(output/"campaign_state.json")
            self.assertEqual((state["completed_blocks"],state["counts"]["not_run"]),(1,2))

    def test_campaign_does_not_start_trial_that_cannot_fit_hard_deadline_reserve(self):
        with tempfile.TemporaryDirectory(prefix="advanced_hard_reserve_") as directory:
            root=Path(directory);output=root/"raw";report=root/"report";entries=[]
            for index,planner in enumerate(ADVANCED_PLANNERS):
                entries.append({"trial_id":planner,"problem_slot_id":"slot","shuffle_order":index,
                    "planner_id":planner,"planner_seed":91001,"map_id":"m","map_hash":"h",
                    "map_path":"fixture","family":"fixture","map_seed":3001,"query_id":"q0",
                    "query_hash":"q","start":[0.,0.],"goal":[1.,1.]})
            config={"advanced_benchmark":{"protocol_id":"fixture","timer_scope":"fixture",
                        "collection_soft_stop_s":10.,"validation_checkpoint_s":10.5,"hard_finalize_s":11.,
                        "bootstrap_replicates":2,"bootstrap_seed":1},
                    "profiles":{"advanced":{"planning_budget_s":5.}},"execution":{"watchdog_grace_s":5.}}
            manifest={"experiment_id":"fixture","identity":{"config_hash":"c"},"trials":entries,
                      "history":[],"status":"PLANNED"}
            def fake_prepare(*args):
                output.mkdir();dump(output/"manifest.json",manifest)
                dump(output/"campaign_state.json",{"status":"PLANNED","elapsed_s":None})
                return config,{},manifest,{}
            with patch("run_advanced_campaign.prepare",side_effect=fake_prepare), \
                 patch("run_advanced_campaign.correctness_gate"), \
                 patch("run_advanced_campaign.run_trial") as launched, \
                 patch("run_advanced_campaign.run_analysis_with_deadline",
                       return_value={"status":"NOT_STARTED_INSUFFICIENT_HARD_BUDGET"}), \
                 patch("run_advanced_campaign.time.perf_counter",return_value=100.):
                result=run_campaign("config","maps",output,report,"worker")
            self.assertEqual(result,2)
            launched.assert_not_called()

    def test_native_final_log_counters_do_not_invent_missing_zeros(self):
        fixture=read_json(ROOT/"tests/fixtures/bitstar_counter_logs.json")
        success=bitstar_log_counters(fixture["success"])
        self.assertEqual(success["rewires"],7);self.assertEqual(success["samples_accepted"],80)
        failure=bitstar_log_counters(fixture["failure"])
        self.assertEqual(failure["rewires"],0);self.assertEqual(failure["samples_accepted"],40)
        interrupted=bitstar_log_counters(fixture["interrupted"])
        self.assertIsNone(interrupted["rewires"]);self.assertIsNone(interrupted["samples_accepted"])

    def test_partial_counts_keep_failures_and_unexecuted_distinct(self):
        statuses=[{"execution_status":"COMPLETED","success":True,"path_available":True,"independent_valid":True,"snapshots_valid":True},
            {"execution_status":"TIMEOUT","success":False},{"execution_status":"CRASH","termination_reason":"INTERRUPTED"},
            {"execution_status":"WATCHDOG"},{"execution_status":"BLOCKED"}]
        manifest={"trials":[{"trial_id":str(i)} for i in range(6)]}
        def exists(path):return path.parent.name!="5"
        with patch.object(Path,"exists",exists),patch("run_experiments.read_json",side_effect=statuses):
            summary=summarize(Path("fixture_only"),manifest)
        self.assertEqual(summary["executed"],4);self.assertEqual(summary["not_run"],2)
        for name in ("validated_success","timeout","crash","watchdog","blocked","interrupted"):self.assertEqual(summary[name],1)

    def test_oracle_spatial_index_matches_full_independent_clipping(self):
        parent=ROOT/"build/runner_geometry_regression";parent.mkdir(parents=True,exist_ok=True)
        directory=Path(tempfile.mkdtemp(prefix="run_",dir=parent));path=directory/"map.t25d"
        heights=[[.08*i+.05*j+.3*((i+j)%2) for i in range(9)] for j in range(8)]
        heights[3][4]=None
        write_map(path,heights,origin=(-2.,3.),spacing=(.7,.9),radius=.1,margin=0,slope=70,metadata={"correctness_test":True})
        oracle=Oracle(path);rng=random.Random(6201)
        pairs=[([-2.,3.],[-2.,3.]),([-1.3,3.9],[2.2,7.5]),([-.6,3.],[.1,9.3])]
        pairs += [([rng.uniform(-2,3.6),rng.uniform(3,9.3)],[rng.uniform(-2,3.6),rng.uniform(3,9.3)]) for _ in range(120)]
        for a,b in pairs:
            indexed=oracle.intervals(a,b);full=oracle.intervals(a,b,use_index=False)
            canonical=lambda rows:sorted((round(low,12),round(high,12),tuple(round(float(v),12) for v in coeff)) for low,high,coeff in rows if high-low>1e-13)
            self.assertEqual(canonical(indexed),canonical(full))

if __name__=="__main__":unittest.main()
