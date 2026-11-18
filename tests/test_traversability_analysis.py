"""Synthetic parser fixtures stay below tests/fixtures and carry explicit labels."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from experiment_common import digest,dump,file_hash,read_json
from analyze_a3_traversability import incumbent_at
from analyze_results import analyze
from independent_verifier import Oracle
from terrain_io import write_map
from validate_results import validate_trial

ROOT=Path(__file__).resolve().parents[1]
CONFIG={"feature_version":"cell_envelope_v1","lambda":1.,"slope_weight":.5,"clearance_weight":.5,"clearance_comfort_m":.4}


def fixture(folder):
    source=folder/"raw";source.mkdir()
    dump(source/"resolved_config.json",{"schema_version":8,"traversability":CONFIG,"profiles":{"smoke":{"planning_budget_s":5.}}})
    entries=[]
    for index,(status,lam) in enumerate((("COMPLETED",1.),("COMPLETED",2.),("TIMEOUT",1.),("BLOCKED",1.),("NOT_RUN",4.),("CRASH",4.))):
        policy=dict(CONFIG,**{"lambda":lam})
        entry=dict(trial_id=f"synthetic_{index}",family="synthetic",map_id="synthetic_map",map_hash="synthetic",query_id="q0",
                   planner_id="bitstar_a3",planner_seed=index,resolved_traversability=policy,
                   objective_config_sha256=digest({"objective_type":"surface_length_traversability","traversability":policy}))
        entries.append(entry)
        if status=="NOT_RUN":continue
        directory=source/entry["trial_id"]
        success=status=="COMPLETED"
        trial=dict(entry,execution_status=status,planner_status="EXACT_SOLUTION" if success else "NO_SOLUTION",
                   independent_valid=success,path_available=success,time_budget_s=5.,worker_wall_s=5.5,validation_wall_s=.3,
                   load_s=.1,preprocessing_s=.2,setup_s=.3,actual_solve_wall_s=5.,
                   effective_traversability=policy,requested_traversability=policy,objective_type="surface_length_traversability",
                   traversability_preprocessing_s=.15,counters={"motion_validity_calls":42},peak_rss_mb=16.)
        dump(directory/"trial.json",trial);events=[]
        for number,(time,length,risk) in enumerate(((1.,6.,4.),(2.,7.,2.),(6.,3.,0.)) if success else ()):
            sid=f"snapshot_{number}";snap=directory/"snapshots"/sid;snap.mkdir(parents=True)
            (snap/"path_xy.csv").write_text("x_m,y_m\n0,0\n1,0\n")
            (snap/"path_surface.csv").write_text("x_m,y_m,z_m\n0,0,0\n1,0,0\n")
            metrics=dict(cost_terrain_m=length,objective_value=length+lam*risk,risk_slope_integral_m=risk,
                         risk_clearance_integral_m=risk,risk_integral_m=risk,length_xy_m=length,
                         minimum_obstacle_footprint_clearance_m=.25,maximum_swept_footprint_slope_deg=12.)
            checked=dict(metrics,valid=True,synthetic_test_fixture=True,xy_sha256=file_hash(snap/"path_xy.csv"),
                         surface_sha256=file_hash(snap/"path_surface.csv"))
            dump(snap/"validation.json",checked)
            events.append(dict(metrics,snapshot_id=sid,elapsed_s=time,best_cost_terrain_m=length,independent_valid=True))
        dump(directory/"validation.json",{"valid":success,"synthetic_test_fixture":True})
        dump(directory/"validated_events.json",events)
    dump(source/"manifest.json",dict(schema_version=8,experiment_id="synthetic_traversability_analysis",profile="smoke",trials=entries,planned_trials=len(entries)))
    return source


class TraversabilityAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="traversability_analysis_",dir=ROOT/"tests/fixtures")
        self.folder=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_last_tuple_length_can_increase_and_lambda_groups_remain_separate(self):
        source=fixture(self.folder);out=self.folder/"report"
        summary=analyze(source,out,make_plots=False)
        self.assertEqual((summary["planned"],summary["executed"],summary["validated_success"],summary["timeout"],summary["blocked"],summary["not_run"]),(6,4,2,1,1,2))
        rows=read_json(out/"trial_metrics.json");first=rows[0]
        self.assertEqual((first["cost_terrain_m"],first["risk_integral_m"],first["objective_value"]),(7.,2.,9.))
        self.assertEqual(first["metric_tuple_at_budget"]["snapshot_id"],"snapshot_1")
        self.assertEqual(first["events_after_budget"],1)
        self.assertEqual(first["first_solution_time_s"],1.)
        self.assertAlmostEqual(first["first_solution_end_to_end_s"],1.6)
        self.assertAlmostEqual(first["end_to_end_run_wall_s"],5.8)
        self.assertEqual(first["metrics"]["motion_validity_calls"],42)
        self.assertEqual(first["traversability_preprocessing_s"],.15)
        self.assertEqual({g["lambda"] for g in summary["groups"]},{1.,2.,4.})
        self.assertEqual((rows[4]["status"],rows[4]["lambda"]),("NOT_RUN",4.))
        self.assertEqual((rows[5]["status"],rows[5]["lambda"]),("CRASH",4.))
        self.assertIsNone(rows[2]["objective_value"])
        self.assertIsNone(incumbent_at(first["events"],.5))
        self.assertEqual(summary["objective_comparison_policy"],"within_same_lambda_and_feature_weights_only")
    def test_tampered_path_invalidates_entire_trial(self):
        source=fixture(self.folder)
        (source/"synthetic_0/snapshots/snapshot_1/path_xy.csv").write_text("changed")
        summary=analyze(source,self.folder/"report",make_plots=False)
        self.assertEqual(summary["invalid"],1);self.assertEqual(summary["validated_success"],1)
    def test_crash_before_worker_output_retains_failure(self):
        directory=self.folder/"crash_no_output";directory.mkdir()
        path=directory/"flat.t25d";write_map(path,[[0.]*5 for _ in range(5)],metadata={"synthetic_test_fixture":True})
        dump(directory/"request.json",dict(map_path=str(path),start=[1.,1.],goal=[3.,3.],traversability=CONFIG))
        dump(directory/"trial.json",dict(map_hash=file_hash(path),execution_status="CRASH",path_available=False,time_budget_s=5.))
        trial=validate_trial(directory)
        self.assertFalse(trial["success"]);self.assertEqual(trial["execution_status"],"CRASH")
        self.assertIsNone(trial["effective_traversability"])
        self.assertIn("not_observed",trial["effective_traversability_unavailable_reason"])
    def test_real_oracle_validator_accepts_longer_safer_incumbent(self):
        directory=self.folder/"real_generated_fixture";directory.mkdir()
        path=directory/"flat.t25d";write_map(path,[[0.]*8 for _ in range(8)],radius=.1,margin=0.,slope=35.,
            rectangles=[(1.5,2.4,4.5,3.)],metadata={"synthetic_test_fixture":True})
        oracle=Oracle(path);policy=dict(CONFIG,**{"lambda":4.})
        paths=[[[1.2,2.2],[4.8,2.2]],[[1.2,2.2],[1.2,1.2],[4.8,1.2],[4.8,2.2]]]
        def save(folder,xy):
            folder.mkdir(parents=True,exist_ok=True);points=[]
            for a,b in zip(xy,xy[1:]):
                _,part=oracle.surface(a,b);points += part if not points else part[1:]
            (folder/"path_xy.csv").write_text("x_m,y_m\n"+"\n".join(",".join(map(str,p)) for p in xy)+"\n")
            (folder/"path_surface.csv").write_text("x_m,y_m,z_m\n"+"\n".join(",".join(map(str,p)) for p in points)+"\n")
            return oracle.validate_path(xy,paths[0][0],paths[0][-1],traversability=policy)
        events=[]
        for i,xy in enumerate(paths):
            metrics=save(directory/"snapshots"/f"s{i}",xy);self.assertTrue(metrics["valid"],metrics)
            events.append(dict(metrics,event_type="improvement",snapshot_id=f"s{i}",elapsed_s=float(i+1),best_cost_terrain_m=metrics["cost_terrain_m"]))
        self.assertGreater(events[1]["cost_terrain_m"],events[0]["cost_terrain_m"])
        self.assertLess(events[1]["objective_value"],events[0]["objective_value"])
        final=save(directory,paths[-1])
        dump(directory/"request.json",dict(map_path=str(path),start=paths[0][0],goal=paths[0][-1],traversability=policy))
        dump(directory/"trial.json",dict(final,map_hash=file_hash(path),execution_status="COMPLETED",planner_status="EXACT_SOLUTION",path_available=True,
            time_budget_s=5.,best_cost_terrain_m=final["cost_terrain_m"],requested_traversability=policy,
            effective_traversability=policy,objective_type="surface_length_traversability"))
        import json
        (directory/"progress.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n")
        trial=validate_trial(directory,oracle)
        self.assertTrue(trial["success"],read_json(directory/"validation.json"))
        self.assertTrue(trial["snapshots_valid"])


if __name__=="__main__":unittest.main()
