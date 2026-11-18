"""Synthetic in-memory registration, report and independent lineage checks.

No planner or experiment is launched; no synthetic artifact enters results.
"""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import pri_segment_protocol as protocol
import pri_pathsearch_protocol as previous
import run_pri_segment_smoke as runner
import run_pri_pathsearch as established
from pri_pathsearch_validation import _check_segment_lineage, _check_request, validate_trial


class SegmentProtocolTests(unittest.TestCase):
    def test_frozen_sources_include_new_correctness_and_operator(self):
        sources = protocol.source_hashes()
        for name in ("tests/integration/pri_segment_search_tests.cpp", "tests/test_pri_segment_worker.py",
                     "src/PriSegmentSearch.cpp", "include/terrain25d/PriSegmentSearch.hpp",
                     "tools/pri_segment_protocol.py", "cmake/PriSegmentTargets.cmake"):
            self.assertEqual(sources[name], protocol.file_hash(ROOT / name))

    def test_only_twenty_eight_physical_smoke_runs(self):
        config = protocol.load_config()
        plans = [protocol.inventory(config,phase) for phase in protocol.PHASES]
        self.assertEqual(sum(p["planned_trials"] for p in plans),28)
        for plan in plans:
            self.assertEqual(plan["map_count"],7)
            self.assertEqual(plan["planning_budget_s"],2.)
            self.assertEqual({e["method"] for e in plan["entries"]},set(protocol.METHODS))
            self.assertEqual(len({e["trial_id"] for e in plan["entries"]}),14)
        self.assertEqual([{k:v for k,v in e.items() if k!="phase"} for e in plans[0]["entries"]],
                         [{k:v for k,v in e.items() if k!="phase"} for e in plans[1]["entries"]])
        self.assertEqual(previous.load_config()["profiles"]["pilot"]["planned_trials"],375)

    def test_no_pilot_confirmation_full_or_selection(self):
        for phase in ("pilot","confirm","full"):
            with self.assertRaises(ValueError):
                protocol.inventory(protocol.default_config(),phase)
            with self.assertRaises(ValueError):
                runner._phase(phase)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.main([phase,"--output","must_not_be_created"])
        with self.assertRaises(ValueError):
            protocol.inventory(protocol.default_config(),"smoke_serial",1.05)
        config = protocol.default_config()
        config["analysis"]["pilot_registered"] = True
        with self.assertRaises(ValueError):
            protocol.validate_config(config)

    def test_real_mode_requests_and_spoofed_slot_rejected(self):
        config=protocol.default_config()
        for entry in protocol.inventory(config)["entries"][:2]:
            info=dict(map_id=entry["map_id"],query_id="q0",map_path="synthetic.t25d",map_sha256="map",
                cloud_path="synthetic.npy",cloud_sha256="cloud",start=[1.,1.],goal=[2.,2.])
            metadata=dict(field_sha256="field",cloud_sha256="cloud",builder_source_sha256="source")
            request=protocol.request_for(entry,info,config,field_path=ROOT/"tests/fixtures/synthetic_field.txt",
                                         field_metadata=metadata)
            self.assertEqual(request["pri_pathsearch"],{"mode":protocol.MODES[entry["method"]]})
            self.assertEqual(request["planning_budget_s"],2.)
            self.assertEqual(_check_request(request)[0],protocol.MODES[entry["method"]])
            self.assertEqual(request["objective_type"],"surface_length")
            with self.assertRaises(ValueError):
                protocol.request_for(dict(entry,planner_seed=12),info,config,field_path="f",field_metadata=metadata)

    def test_execution_engine_isolation_and_importable_spawn_targets(self):
        self.assertIs(established.protocol,previous)
        self.assertIs(runner._engine.protocol,protocol)
        self.assertIsNot(runner._engine,established)
        self.assertEqual(runner._engine._collect_trial.__module__,"run_pri_segment_smoke")
        self.assertEqual(runner._engine._validate_map.__module__,"run_pri_pathsearch")
        hashes=protocol.source_hashes()
        for source in ("tools/run_pri_segment_smoke.py","configs/a3t_pri_segment_v2.json",
                       "tests/test_pri_segment_protocol.py","include/terrain25d/PriSegmentSearch.hpp"):
            self.assertIn(source,hashes)

    def report_fixture(self):
        config=protocol.default_config();manifest=protocol.inventory(config)
        manifest.update(status="COMPLETE",sources={"synthetic":"fixture"},synthetic_fixture=True)
        rows=[]
        for entry in manifest["entries"]:
            entry["status"]="COMPLETED"
            selectors=[dict(alpha=a,success=True,attained_safe=True,selected_snapshot_id="candidate_0",
                final_cost_terrain_m=2.,final_pri=3.,final_risk_sum=12.,final_risk_count=4,
                final_exposure=6.) for a in protocol.ALPHAS]
            rows.append(dict(entry,execution_status="COMPLETED",success=True,correctness_errors=[],
                             selector_results=selectors,synthetic_fixture=True))
        return dict(rows=rows,attempted=14),manifest,config

    def test_report_keeps_physical_failures_and_correlated_selectors(self):
        summary,manifest,config=self.report_fixture()
        row=summary["rows"][0]
        row.update(success=False,correctness_errors=["synthetic_bad_selected_path"])
        report=runner.build_report(summary,manifest,config)
        self.assertEqual(report["status"],"FAILED_CORRECTNESS")
        self.assertEqual(report["planned"],14)
        self.assertEqual(report["success"],13)
        self.assertEqual(len(report["selector_trials"]),56)
        failed=[s for s in report["selector_trials"] if s["physical_trial_id"]==row["trial_id"]]
        self.assertTrue(all(not s["success"] and s["final_pri"] is None for s in failed))
        self.assertFalse(report["alpha_selection"])
        self.assertIsNone(report["selected_alpha"])
        self.assertEqual(len(report["paired_comparisons"]),28)

    def test_report_rejects_missing_success_selectors_and_nonfinite_metrics(self):
        for mutation in ("missing","nonfinite","duplicate"):
            summary,manifest,config=self.report_fixture()
            if mutation=="missing":summary["rows"][0]["selector_results"]=[]
            elif mutation=="nonfinite":summary["rows"][0]["selector_results"][0]["final_exposure"]=float("nan")
            else:summary["rows"].append(deepcopy(summary["rows"][0]))
            with self.assertRaises(ValueError):runner.build_report(summary,manifest,config)


class SegmentLineageTests(unittest.TestCase):
    def fixture(self):
        parent=dict(snapshot_id="candidate_0",proposal_id=0,move_type="bitstar",pri=10.,cost_terrain_m=5.,
            score_completed_s=.2,callback_s=.1,parent_ids=[])
        child=dict(snapshot_id="candidate_1",proposal_id=3,move_type="segment_reduce",pri=9.,cost_terrain_m=4.9,
            score_completed_s=.8,callback_s=.7,parent_ids=["candidate_0"],segment_pri_cap=9.9,
            segment_length_cap=5.,segment_first=1,segment_last=2)
        keys=("nodes_proposed","nodes_retained","edge_evaluations","edge_cache_hits","expansions",
            "labels_created","beam_discarded","distinct_sets_same_node","prefix_above_pri_kept","length_rejected",
            "invalid_edges","non_simple_rejected","completed_paths","pri_rejected","full_recomputations","cancel_checks","soft_stops")
        attempt=dict(proposal_id=3,parent_id="candidate_0",move_type="segment_reduce",first=1,last=2,
            pri_cap=9.9,length_cap=5.,began_s=.5,finished_s=.9,status="ACCEPTED",termination="WORK_COMPLETED",counters=dict.fromkeys(keys,0))
        worker=dict(anchor_snapshot_id="candidate_0",actual_solve_wall_s=1.,
            search_counters=dict(segment_proposals=1,proposals=3),segment_attempts=[attempt])
        paths={"candidate_0":[[0.,0.],[1.,1.],[2.,1.],[3.,0.]],
               "candidate_1":[[0.,0.],[1.,1.],[1.5,.5],[2.,1.],[3.,0.]]}
        measured={"candidate_0":dict(pri=10.,cost_terrain_m=5.),"candidate_1":dict(pri=9.,cost_terrain_m=4.9)}
        return [parent,child],worker,measured,paths,dict(planning_budget_s=2.)

    def errors(self,data):
        errors=[];_check_segment_lineage(*data,errors);return errors

    def test_valid_contextual_replacement(self):
        self.assertEqual(self.errors(self.fixture()),[])

    def test_tampered_context_target_or_counter_fails(self):
        for change,expected in (("context","outside_context"),("target","target_definition"),
                                ("metric","independent_target"),("bound","search_bound"),("receipt","inventory")):
            data=self.fixture();records,worker,measured,paths,request=data
            if change=="context":paths["candidate_1"][0]=[0.,.01]
            elif change=="target":worker["segment_attempts"][0]["pri_cap"]=10.
            elif change=="metric":measured["candidate_1"]["pri"]=10.
            elif change=="bound":worker["segment_attempts"][0]["counters"]["expansions"]=129
            else:worker["segment_attempts"]=[]
            self.assertTrue(any(expected in error for error in self.errors(data)),change)

    def test_contract_requires_a_strictly_shorter_complete_path(self):
        data=self.fixture();records,worker,measured,paths,request=data
        reduce=deepcopy(worker["segment_attempts"][0])
        reduce.update(status="NO_CANDIDATE",finished_s=.55)
        attempt=worker["segment_attempts"][0]
        attempt.update(proposal_id=4,move_type="segment_contract",pri_cap=10.,began_s=.6)
        worker["segment_attempts"]=[reduce,attempt]
        worker["search_counters"].update(segment_proposals=2,proposals=4)
        records[1].update(proposal_id=4,move_type="segment_contract",segment_pri_cap=10.)
        self.assertEqual(self.errors(data),[])
        records[1]["cost_terrain_m"]=5.
        self.assertTrue(any("contract_not_shorter" in e for e in self.errors(data)))

    def test_segment_uses_original_independent_four_selector_oracle(self):
        sys.path.insert(0,str(ROOT/"tests"))
        from test_pri_pathsearch_validation import PathSearchValidationTests
        from corridor_v4_execution import read,atomic_json
        fixtures=PathSearchValidationTests()
        with fixtures.workspace() as temporary:
            directory=Path(temporary)
            request,info,_,oracle,field,_=fixtures.fixture(directory,mode="segment")
            request["variant_id"]=protocol.SEGMENT
            worker=read(directory/"worker.json")
            worker.update(backend_id=protocol.SEGMENT,planner_id=protocol.SEGMENT,segment_attempts=[])
            worker["effective_pri_pathsearch"]["segment_search"]=deepcopy(protocol.SEGMENT_POLICY)
            worker["search_counters"]["segment_proposals"]=0
            atomic_json(directory/"worker.json",worker)
            row=validate_trial(directory,request,info,dict(status="COMPLETED"),oracle=oracle,field=field)
            self.assertEqual(row["correctness_errors"],[])
            self.assertTrue(row["success"])
            self.assertEqual(len(row["selector_results"]),4)
            worker["effective_pri_pathsearch"]["segment_search"]["prefix_PRI_pruning"]=True
            atomic_json(directory/"worker.json",worker)
            bad=validate_trial(directory,request,info,dict(status="COMPLETED"),oracle=oracle,field=field)
            self.assertIn("effective_segment_search_mismatch",bad["correctness_errors"])
            self.assertTrue(all(not s["success"] for s in bad["selector_results"]))


if __name__=="__main__":
    unittest.main()
