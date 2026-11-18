"""Independent raw-map and censored-data analysis regressions, test fixtures only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from experiment_common import dump
from terrain_io import write_map
from independent_verifier import Oracle
from terrain_compare_protocol import DEFAULT_TRAVERSABILITY, sha256
from terrain_compare_analysis import at_checkpoint, paired_bootstrap, validate_campaign, analyze_campaign


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_compare_analysis_",dir=ROOT/"tests/fixtures")
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def fixture(self,track="common",claimed_shift=0):
        map_path=self.root/"analytic.t25d"
        write_map(map_path,[[0.]*21 for _ in range(21)],spacing=(.4,.4),rectangles=[(2,3,6,5)],
                  metadata={"synthetic_test_fixture":True})
        start,goal=[1,4],[7,4]
        paths=[[start,[1.5,2.55],[6.5,2.55],goal],[start,[1.5,2.1],[6.5,2.1],goal]]
        oracle=Oracle(map_path)
        checked=[oracle.validate_path(p,start,goal,traversability=DEFAULT_TRAVERSABILITY) for p in paths]
        self.assertTrue(all(c["valid"] for c in checked),checked)
        self.assertLess(checked[0]["cost_terrain_m"],checked[1]["cost_terrain_m"])
        self.assertGreater(checked[0]["objective_value"],checked[1]["objective_value"])
        # For own-cost, native costs improve while common J gets worse.
        if track=="own":paths.reverse();checked.reverse()
        directory=self.root/"synthetic_trial";directory.mkdir()
        request=dict(map_path=str(map_path),start=start,goal=goal,planning_budget_s=2.)
        dump(directory/"request.json",request)
        events=[]
        for i,(path,c) in enumerate(zip(paths,checked)):
            event=dict(snapshot_id=f"s{i}",elapsed_s=.5+i*.5,path_xy=path,
                       planner_objective_value=c["objective_value"] if track=="common" else 20-i,
                       cost_terrain_m=c["cost_terrain_m"]+claimed_shift)
            events.append(event)
        dump(directory/"worker.json",dict(exact_goal=True,snapshots=events,
            phase_timings=dict(load_s=.1,conversion_s=.2,preprocessing_s=.3,setup_s=.4)))
        entry=dict(trial_id=directory.name,track=track,backend="putn",family="synthetic",
                   map_id="analytic",planner_seed=11,map_sha256=sha256(map_path),request=request,execution_status="COMPLETED")
        dump(self.root/"manifest.json",dict(entries=[entry],synthetic_test_fixture=True))
        return checked,entry
    def test_same_incumbent_tuple_length_up_objective_down(self):
        checked,_=self.fixture();result=validate_campaign(self.root)
        self.assertEqual(result["validated_success"],1,result)
        r=result["rows"][0];self.assertIsNone(at_checkpoint(r["events"],.49))
        self.assertEqual(at_checkpoint(r["events"],.75)["snapshot_id"],"s0")
        final=at_checkpoint(r["events"],2.)
        self.assertEqual(final["snapshot_id"],"s1")
        self.assertAlmostEqual(final["metrics"]["cost_terrain_m"],checked[1]["cost_terrain_m"])
        self.assertAlmostEqual(final["metrics"]["objective_value"],checked[1]["objective_value"])
        self.assertAlmostEqual(r["first_valid_latency_s"],1.5)
        # The Markdown-only report must not import matplotlib, even when a
        # notebook backend is inherited by the fresh worker interpreter.
        env=dict(os.environ,MPLBACKEND="synthetic_invalid_notebook_backend")
        code=("import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
              "from terrain_compare_analysis import analyze_campaign;"
              "analyze_campaign(Path(sys.argv[2]),Path(sys.argv[3]));"
              "assert not any(k=='matplotlib' or k.startswith('matplotlib.') for k in sys.modules)")
        completed=subprocess.run([sys.executable,"-c",code,str(ROOT/"tools"),
                                  str(self.root),str(self.root/"report")],
                                 env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertEqual(list((self.root/"report").glob("*.png")),[])
        report=(self.root/"report/REPORT.md").read_text(encoding="utf-8")
        for label in ("Median T_first", "Median cold-start latency", "Median L_H", "Median R_s", "Median R_c",
                      "Median R/L_H", "obstacle-only footprint clearance", "forbidden-region footprint clearance",
                      "max swept-footprint slope", "process-tree CPU", "throughput:", "CI 95%", "restricted_time"):
            self.assertIn(label,report)
        self.assertIn("| common | putn | 1 | 1 | 1 | COMPLETED: 1 |",report)
        self.assertIn("| common | putn | 1 | 0.5 | 1.5 |",report)
        self.assertTrue((self.root/"report/trials.csv").is_file())
        summary=json.loads((self.root/"report/summary.json").read_text())
        table=next(row for row in summary["tables"] if row["track"]=="common" and row["backend"]=="putn")
        self.assertAlmostEqual(table["median_cost_terrain_m"],checked[1]["cost_terrain_m"])
        self.assertAlmostEqual(table["median_objective_value"],checked[1]["objective_value"])
        output=json.loads((self.root/"report/anytime_tuples.json").read_text())
        self.assertEqual(next(v for v in output if v["checkpoint_s"]==2.)["snapshot"]["snapshot_id"],"s1")
    def test_own_cost_does_not_require_common_objective_to_improve(self):
        checked,_=self.fixture("own");result=validate_campaign(self.root)
        self.assertEqual(result["validated_success"],1,result)
        events=result["rows"][0]["events"]
        self.assertLess(events[1]["planner_objective_value"],events[0]["planner_objective_value"])
        self.assertGreater(events[1]["metrics"]["objective_value"],events[0]["metrics"]["objective_value"])
    def test_worker_claims_not_trusted_and_raw_map_hash_enforced(self):
        self.fixture(claimed_shift=1.)
        result=validate_campaign(self.root)
        self.assertEqual(result["validated_success"],0)
        self.assertEqual(result["correctness_failures"],1)
        path=self.root/"analytic.t25d";path.write_bytes(path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"raw map hash changed"):validate_campaign(self.root)
    def test_crash_and_unexecuted_slots_preserved(self):
        _,entry=self.fixture();manifest=json.loads((self.root/"manifest.json").read_text())
        crash=copy.deepcopy(entry);crash.update(trial_id="crash_no_output",backend="sgp",execution_status="CRASH")
        blocked=copy.deepcopy(entry);blocked.update(trial_id="blocked_source",backend="sem",track="own",execution_status="BLOCKED_SOURCE")
        planned=copy.deepcopy(entry);planned.update(trial_id="not_run",backend="a3",execution_status="PLANNED")
        success_a3=copy.deepcopy(entry);success_a3.update(trial_id="a3_valid",backend="a3")
        # Pair bootstrap directly with independently validated PUTN reused as an explicit test row.
        manifest["entries"].extend([crash,blocked,planned]);dump(self.root/"manifest.json",manifest)
        result=validate_campaign(self.root);self.assertEqual(result["slots"],4)
        crashed=next(r for r in result["rows"] if r["execution_status"]=="CRASH")
        self.assertEqual(crashed["budget_s"],2.)
        a=copy.deepcopy(result["rows"][0]);a["backend"]="a3"
        contrast=paired_bootstrap([a,crashed],"common","sgp","restricted_time",replicates=100)
        self.assertEqual(contrast["usable_pairs"],1)
        self.assertAlmostEqual(contrast["mean_difference"],.5-2.)
        self.assertEqual(paired_bootstrap([a,crashed],"common","sgp","cost_terrain_m",100)["usable_pairs"],0)
        analyze_campaign(self.root,self.root/"report")
        report=(self.root/"report/REPORT.md").read_text(encoding="utf-8")
        for status in ("CRASH: 1","BLOCKED_SOURCE: 1","PLANNED: 1"):
            self.assertIn(status,report)
        self.assertEqual(list((self.root/"report").glob("*.png")),[])
    def test_refuses_validation_during_collection(self):
        _,entry=self.fixture();entry["execution_status"]="RUNNING";dump(self.root/"manifest.json",dict(entries=[entry]))
        with self.assertRaisesRegex(ValueError,"overlap"):validate_campaign(self.root)


if __name__=="__main__":unittest.main()
