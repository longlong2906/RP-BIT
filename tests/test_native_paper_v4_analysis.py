"""Explicitly synthetic parser/geometry fixtures, never benchmark observations."""
import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from native_paper_v4_analysis import (validate_trial,validate_map,build_report,analyze_campaign,
    paired_bootstrap,group_summary,restricted_observation,write_report,dump,read,file_hash,DEFAULT_TRAVERSABILITY,
    path_csv,_native_raster,GOAL_ROUNDOFF_TOLERANCE_M,singleton_endpoint_errors)
from terrain_io import write_map
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField


def event(sid="synthetic_s0",t=1.,length=10.,safe=True,geometry=True,turns=1,pri=2.):
    return dict(snapshot_id=sid,elapsed_s=t,safe=safe,geometry_valid=geometry,
        metrics=dict(cost_terrain_m=length,length_xy_m=length,pri=pri,
                     maximum_swept_footprint_slope_deg=0.,sharp_turn_count=turns))


def row(backend="a3t_safe",seed=1,status="COMPLETED",safe=True,length=10.,time=.1,map_id="synthetic_map",family="synthetic"):
    return dict(trial_id=f"synthetic_{map_id}_{backend}_{seed}",backend=backend,family=family,map_id=map_id,query_id="q0",
        planner_seed=seed,timing_stratum="synthetic_machine",execution_status=status,budget_s=20.,
        success=safe,success_geometry=True,attained_safe=True,attained_geometry=True,
        first_safe_s=time,first_geometry_s=time,first_safe_latency_s=time+.6,first_geometry_latency_s=time+.6,
        solve_time_s=20.,events=[event(length=length,safe=safe,t=time)],correctness_errors=[],synthetic_fixture=True)


def entries(rows):
    return [{key:r[key] for key in ("trial_id","backend","family","map_id","query_id","planner_seed","timing_stratum","execution_status")} for r in rows]


def csv_write(path,points):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["x","y"] if len(points[0])==2 else ["x","y","z"]);writer.writerows(points)


def fixture(base,backend="a3t_old"):
    map_path=base/"synthetic_map.t25d"
    write_map(map_path,[[0.]*5 for _ in range(5)],radius=.1,margin=0.,
              metadata={"synthetic_test_fixture":True})
    axis=np.arange(.025,4.,.05);x,y=np.meshgrid(axis,axis)
    cloud=base/"synthetic_cloud.npy"
    np.save(cloud,np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))),allow_pickle=False)
    info=dict(map_path=str(map_path),map_sha256=file_hash(map_path),cloud_path=str(cloud),cloud_sha256=file_hash(cloud),
        bounds=[0.,0.,4.,4.],start=[1.,1.],goal=[3.,1.],map_id="synthetic_map",family="synthetic",mode="main")
    request=dict(backend_id=backend,map_path=str(map_path),start=info["start"],goal=info["goal"],planner_seed=11,
                 planning_budget_s=20.)
    if backend=="a3t_old":
        request["traversability"]=copy.deepcopy(DEFAULT_TRAVERSABILITY)
    if backend=="a3t_safe":
        request.update(safe_length=dict(policy_id="a3t_safe_length_v1",risk_cap=40.,slope_cap_deg=35.,atlas_sampling=True),
                       samples_per_batch=100,batch_after_first_solution=200)
    directory=base/"synthetic_trial";directory.mkdir()
    dump(directory/"request.json",request)
    source=dict(snapshot_id="s0",elapsed_s=.2,objective_value=2.,native_objective_value=2.,cost_terrain_m=2.,length_xy_m=2.,
                risk_slope_integral_m=0.,risk_clearance_integral_m=0.,risk_integral_m=0.)
    worker=dict(planner_status="EXACT_SOLUTION",snapshots=[source],actual_solve_wall_s=20.,
        solve_started_utc="2026-11-08T00:00:01+00:00",effective_traversability=copy.deepcopy(DEFAULT_TRAVERSABILITY))
    if backend=="a3t_safe":
        worker.update(requested_safe_length=request["safe_length"],
            effective_safe_length=dict(request["safe_length"],thresholds_immutable_during_solve=True),
            effective_policy=dict(safety_atlas_sampling=True,batch_policy="after_first_solution",samples_per_batch=100,
                                  batch_after_first_solution=200),batch_transition_count=1)
    csv_write(directory/"snapshots/s0/path_xy.csv",[[1.,1.],[3.,1.]])
    csv_write(directory/"snapshots/s0/path_surface.csv",[[1.,1.,0.],[2.,1.,0.],[3.,1.,0.]])
    csv_write(directory/"snapshots/s0/path_raw_xyz.csv",[[1.,1.,0.],[3.,1.,0.]])
    csv_write(directory/"path_xy.csv",[[1.,1.],[3.,1.]])
    dump(directory/"worker.json",worker)
    execution=dict(execution_status="COMPLETED",launch_utc="2026-11-08T00:00:00+00:00",field_build_s=.5,
                   timing_stratum="synthetic_machine")
    dump(directory/"execution.json",execution)
    return directory,request,info,execution,worker


class V4AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixtures=ROOT/"tests/fixtures"

    def test_singleton_roundoff_matches_frozen_model_and_is_euclidean(self):
        configured=read(ROOT/"configs/native_paper_kaggle_v4.yaml")["model"]["goal_roundoff_tolerance_m"]
        self.assertEqual(GOAL_ROUNDOFF_TOLERANCE_M,configured)
        tolerance=GOAL_ROUNDOFF_TOLERANCE_M
        for endpoint in (0,1):
            for axis in (0,1):
                path=[[0.,0.],[2.,2.]]
                expected=[point[:] for point in path]
                path[endpoint][axis]+=2*tolerance
                self.assertFalse(singleton_endpoint_errors(path,*expected)[1])
            path=[[0.,0.],[2.,2.]];expected=[point[:] for point in path]
            path[endpoint]=[value+.75*tolerance for value in path[endpoint]]
            self.assertFalse(singleton_endpoint_errors(path,*expected)[1])

    def test_reported_kaggle_decimal_roundtrip_one_ulp(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_decimal_roundtrip_",dir=self.fixtures) as tmp:
            path=Path(tmp)/"path_xy.csv"
            data=b"x_m,y_m\n4.1233834650699288,19.128785755065103\n36.003970273717563,19.128785755065103\n"
            path.write_bytes(data)
            start=[4.123383465069929,19.128785755065106]
            goal=[36.00397027371756,19.128785755065106]
            errors,valid=singleton_endpoint_errors(path_csv(path),start,goal)
            self.assertTrue(valid)
            self.assertEqual(errors,(math.ulp(start[1]),math.ulp(goal[1])))
            self.assertEqual(path.read_bytes(),data)

    def test_authentic_retained_endpoint_fixture_hashes_and_roundoff(self):
        directory=self.fixtures/"native_paper_v4_endpoint_roundoff"
        provenance=read(directory/"provenance.json")
        self.assertEqual(provenance["kind"],"authentic_retained_regression_data")
        self.assertIs(provenance["fabricated"],False)
        before={record["path"]:(directory/record["path"]).read_bytes() for record in provenance["files"]}
        for record in provenance["files"]:
            path=directory/record["path"]
            self.assertEqual(path.stat().st_size,record["size_bytes"])
            self.assertEqual(file_hash(path),record["sha256"])
        request=read(directory/"request.json")
        xy=path_csv(directory/"path_xy.csv")
        errors,valid=singleton_endpoint_errors(xy,request["start"],request["goal"])
        self.assertTrue(valid)
        self.assertEqual(errors,(math.ulp(request["start"][1]),math.ulp(request["goal"][1])))
        self.assertEqual(xy[0][1],provenance["observed_cpp_csv_y"])
        self.assertEqual(request["start"][1],provenance["expected_request_y"])
        self.assertEqual(before,{name:(directory/name).read_bytes() for name in before})

    def test_endpoint_roundoff_keeps_raw_paths_and_rejects_real_offsets_all_backends(self):
        for backend in ("a3t_old","a3t_safe","sem_native"):
            for endpoint in (0,1):
                for axis in (0,1):
                    for one_ulp in (True,False):
                        with self.subTest(backend=backend,endpoint=endpoint,axis=axis,one_ulp=one_ulp), tempfile.TemporaryDirectory(
                                prefix="synthetic_v4_endpoint_",dir=self.fixtures) as tmp:
                            directory,request,info,execution,worker=fixture(Path(tmp),backend)
                            xy=[request["start"][:],request["goal"][:]]
                            old=xy[endpoint][axis]
                            xy[endpoint][axis]=math.nextafter(old,-math.inf) if one_ulp else old+2*GOAL_ROUNDOFF_TOLERANCE_M
                            csv_write(directory/"snapshots/s0/path_xy.csv",xy)
                            csv_write(directory/"path_xy.csv",xy)
                            csv_write(directory/"snapshots/s0/path_raw_xyz.csv",[p+[0.] for p in xy])
                            if backend=="sem_native":
                                raster=Path(tmp)/"synthetic_raster.npz"
                                np.savez(raster,risk=np.zeros((40,40)),valid=np.ones((40,40),dtype=bool),bounds=np.array(info["bounds"]),resolution=.1)
                                worker["artifact_refs"]=dict(native_raster_path=str(raster),native_raster_sha256=file_hash(raster))
                                worker["snapshots"][0]["native_objective_value"]=9.6
                                dump(directory/"worker.json",worker)
                            before={str(p):p.read_bytes() for p in directory.rglob("*.csv")}
                            result=validate_trial(directory,request,info,execution)
                            self.assertEqual(result["success"],one_ulp,result["correctness_errors"])
                            if one_ulp:
                                self.assertEqual(result["correctness_errors"],[])
                                self.assertGreater(result["events"][0]["endpoint_errors_m"]["start" if endpoint==0 else "goal"],0.)
                            else:
                                self.assertIn("not_exact_start_goal",result["events"][0]["geometry_errors"])
                            self.assertEqual(before,{str(p):p.read_bytes() for p in directory.rglob("*.csv")})

    def test_roundoff_fix_does_not_relax_hard_risk_or_swept_slope(self):
        for failure in ("risk","slope"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(prefix="synthetic_v4_endpoint_cap_",dir=self.fixtures) as tmp:
                directory,request,info,execution,worker=fixture(Path(tmp),"a3t_safe")
                request["goal"][1]=math.nextafter(request["goal"][1],math.inf)
                dump(directory/"request.json",request)
                scorer=SharedSEMField(np.load(info["cloud_path"]),info["bounds"])
                oracle=NativePaperOracle(info["map_path"])
                if failure=="risk":
                    scorer.risk=np.full_like(scorer.risk,40.0001)
                    result=validate_trial(directory,request,info,execution,field=scorer,oracle=oracle)
                else:
                    original=oracle.validate_path
                    def steep(*args,**kwargs):
                        checked=original(*args,**kwargs);checked["maximum_swept_footprint_slope_deg"]=35.0001
                        return checked
                    with patch.object(oracle,"validate_path",side_effect=steep):
                        result=validate_trial(directory,request,info,execution,field=scorer,oracle=oracle)
                self.assertFalse(result["success"])
                self.assertIn("s0:hard_safety_contract_failed",result["correctness_errors"])
                self.assertNotIn("not_exact_start_goal",result["events"][0]["geometry_errors"])

    def test_final_unsafe_is_not_replaced_by_first_safe_and_late_is_excluded(self):
        r=row();r["events"] += [event("unsafe",19.,9.,safe=False),event("late",20.001,1.)]
        r["success"]=False
        report=build_report([r],entries([r]),20.,20)
        final=next(t for t in report["tables"] if t["scope"]=="all" and t["selection"]=="final_safe")
        first=next(t for t in report["tables"] if t["scope"]=="all" and t["selection"]=="first_safe")
        self.assertEqual((final["success"],first["success"]),(0,1))
        self.assertEqual(first["cost_terrain_m"]["mean"],10.)
        points=[p for p in report["attainment"] if p["backend"]=="a3t_safe" and p["criterion"]=="safe"]
        self.assertEqual(points[-1]["fraction"],1.)

    def test_psr_all_attempts_sti_ratio_and_missing_observation(self):
        a,b=row(length=10.),row(seed=2,length=30.)
        b["events"][0]["metrics"]["sharp_turn_count"]=2
        failed=row(seed=3,status="CRASH",safe=False);failed.update(success_geometry=False,attained_safe=False,attained_geometry=False)
        unstarted=row(seed=4,status="PLANNED",safe=False)
        result=group_summary([a,b,failed,unstarted])
        self.assertEqual((result["planned"],result["attempted"],result["success"]),(4,3,2))
        self.assertAlmostEqual(result["sti"],3/40)
        self.assertEqual(result["solve_time_s"]["n"],result["pri"]["n"])
        self.assertIsNone(result["restricted_first_safe_mean_s"])
        self.assertEqual(result["restricted_unavailable"],1)

    def test_restricted_timeout_is_censored_not_a_solution(self):
        a=row();b=row(seed=2,safe=False)
        b.update(attained_safe=False,first_safe_s=None)
        group=group_summary([a,b])
        self.assertEqual(group["censored_at_budget"],1)
        self.assertAlmostEqual(group["restricted_first_safe_mean_s"],10.05)
        self.assertIsNone(b["first_safe_s"])
        self.assertEqual(restricted_observation(b),20.)

    def test_paired_bootstrap_family_map_seed_and_stratum(self):
        rows=[row(length=8.),row("sem_native",length=10.,time=.5),row(seed=2,length=8.),row("sem_native",seed=2,length=10.,time=.5)]
        contrast=paired_bootstrap(rows,"sem_native","cost_terrain_m",30)
        self.assertEqual(contrast["ci95"],[-2.,-2.])
        rows[1]["timing_stratum"]="different"
        timing=paired_bootstrap(rows,"sem_native","first_safe_s",30)
        self.assertEqual((timing["excluded_stratum_mismatch"],timing["usable_pairs"]),(1,1))
        rows[2]["execution_status"]="CRASH";rows[2]["attained_safe"]=False
        restricted=paired_bootstrap(rows,"sem_native","restricted_first_safe_s",30)
        self.assertEqual(restricted["unavailable_pairs"],1)
        self.assertIsNone(restricted["mean_difference"])
        self.assertIsNone(restricted["ci95"])

    def test_invalid_duplicate_quality_and_missing_verdict(self):
        r=row()
        with self.assertRaises(ValueError):build_report([r,r],entries([r]),20.,10)
        r["events"][0]["metrics"]["pri"]=float("nan")
        with self.assertRaises(ValueError):build_report([r],entries([r]),20.,10)
        fresh=row();result=build_report([],entries([fresh]),20.,10)
        self.assertEqual(result["completion"],"FAILED_CORRECTNESS")

    def test_old_objective_audit_and_latency_does_not_double_charge_wrapper(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_old_",dir=self.fixtures) as tmp:
            directory,request,info,execution,worker=fixture(Path(tmp))
            result=validate_trial(directory,request,info,execution)
            self.assertTrue(result["success"],result["correctness_errors"])
            self.assertAlmostEqual(result["first_safe_latency_s"],1.2)
            worker["snapshots"][0]["native_objective_value"]=1.
            dump(directory/"worker.json",worker)
            changed=validate_trial(directory,request,info,execution)
            self.assertFalse(changed["success"])
            self.assertIn("s0:native_objective_mismatch",changed["correctness_errors"])

    def test_safe_contract_vs_old_unsafe_planning_failure(self):
        for backend in ("a3t_safe","a3t_old"):
            with tempfile.TemporaryDirectory(prefix="synthetic_v4_contract_",dir=self.fixtures) as tmp:
                directory,request,info,execution,worker=fixture(Path(tmp),backend)
                scorer=SharedSEMField(np.load(info["cloud_path"]),info["bounds"])
                risk=scorer.risk.copy();risk[9:11,19:21]=41.;scorer.risk=risk
                result=validate_trial(directory,request,info,execution,field=scorer)
                self.assertFalse(result["success"])
                self.assertEqual(bool(result["correctness_errors"]),backend=="a3t_safe")
                self.assertTrue(result["planning_failures"])

    def test_late_native_incumbent_does_not_trigger_exact_without_in_budget_error(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_late_",dir=self.fixtures) as tmp:
            directory,request,info,execution,worker=fixture(Path(tmp),"a3t_safe")
            worker["snapshots"][0]["elapsed_s"]=20.01;dump(directory/"worker.json",worker)
            result=validate_trial(directory,request,info,execution)
            self.assertFalse(result["success"])
            self.assertEqual(result["late_snapshot_count"],1)
            self.assertEqual(result["correctness_errors"],[])

    def test_worker_failures_without_events_cannot_pass_correctness_gate(self):
        failures=[("status",value) for value in ("ERROR","MODEL_ERROR","BLOCKED_DEPENDENCY")]
        failures += [("execution_status",value) for value in ("FAILED_CORRECTNESS","PROVENANCE_MISMATCH")]
        for backend in ("a3t_old","a3t_safe","sem_native"):
            for key,value in failures:
                with self.subTest(backend=backend,key=key,value=value), tempfile.TemporaryDirectory(
                        prefix="synthetic_v4_failed_worker_",dir=self.fixtures) as tmp:
                    directory,request,info,execution,worker=fixture(Path(tmp),backend)
                    worker.update(planner_status="TIMEOUT",snapshots=[])
                    worker[key]=value;dump(directory/"worker.json",worker)
                    result=validate_trial(directory,request,info,execution)
                    expected=("worker_execution_failure:" if key=="execution_status" else "worker_status_failure:")+value
                    self.assertIn(expected,result["correctness_errors"])
                    self.assertFalse(result["success"])
                    self.assertEqual(result["events"],[])
                    if key=="status": self.assertEqual(result["native_status"],value)

    def test_plain_timeout_without_events_remains_planning_failure(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_timeout_",dir=self.fixtures) as tmp:
            directory,request,info,execution,worker=fixture(Path(tmp))
            worker.update(planner_status="TIMEOUT",snapshots=[]);dump(directory/"worker.json",worker)
            result=validate_trial(directory,request,info,execution)
            self.assertFalse(result["success"])
            self.assertEqual(result["native_status"],"TIMEOUT")
            self.assertEqual(result["correctness_errors"],[])

    def test_native_compact_raster_reference_and_raw_xyz(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_sem_",dir=self.fixtures) as tmp:
            directory,request,info,execution,worker=fixture(Path(tmp),"sem_native")
            raster=Path(tmp)/"synthetic_native_raster.npz"
            np.savez(raster,risk=np.zeros((40,40)),valid=np.ones((40,40),dtype=bool),bounds=np.array(info["bounds"]),resolution=.1)
            worker["artifact_refs"]=dict(native_raster_path=str(raster),native_raster_sha256=file_hash(raster))
            worker["snapshots"][0]["native_objective_value"]=9.6
            dump(directory/"worker.json",worker)
            result=validate_trial(directory,request,info,execution)
            self.assertTrue(result["success"],result["correctness_errors"])
            csv_write(directory/"snapshots/s0/path_raw_xyz.csv",[[1.,1.,.2],[3.,1.,0.]])
            invalid=validate_trial(directory,request,info,execution)
            self.assertIn("s0:raw_xyz_off_heightfield",invalid["correctness_errors"])
            raster.write_bytes(raster.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"hash"):
                validate_trial(directory,request,info,execution)

    def test_canonical_metric_headers_and_compact_sidecar_relative_reference(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_refs_",dir=self.fixtures) as tmp:
            root=Path(tmp);directory=root/"wave/map/trial";directory.mkdir(parents=True)
            xy=directory/"xy.csv";xy.write_text("x_m,y_m\n1,2\n3,4\n")
            xyz=directory/"xyz.csv";xyz.write_text("x_m,y_m,z_m\n1,2,3\n")
            self.assertEqual(path_csv(xy),[[1.,2.],[3.,4.]])
            self.assertEqual(path_csv(xyz,3),[[1.,2.,3.]])
            raster=root/"shared/raster.npz";raster.parent.mkdir();raster.write_bytes(b"SYNTHETIC raster")
            refs=dict(native_raster_path="../../../shared/raster.npz",native_raster_sha256=file_hash(raster))
            dump(directory/"artifact_refs.json",refs)
            self.assertEqual(_native_raster(directory,{}),raster.resolve())
            with self.assertRaisesRegex(ValueError,"disagrees"):
                _native_raster(directory,{"artifact_refs":{}})

    def test_wave_receipts_ledger_report_without_raw_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_v4_wave_",dir=self.fixtures) as tmp:
            workspace=Path(tmp)
            directory,request,info,execution,worker=fixture(workspace)
            entry=dict(trial_id="synthetic_old",backend="a3t_old",family=info["family"],map_id=info["map_id"],
                query_id="q0",planner_seed=11,timing_stratum="synthetic_machine",directory=str(directory),request=request,execution_status="COMPLETED")
            first=validate_map(workspace,info,[entry])
            self.assertTrue(first[0]["success"])
            again=validate_map(workspace,info,[entry]);self.assertEqual(first,again)
            dump(workspace/"manifests/main.json",dict(entries=[entry],budget=20.,synthetic_fixture=True))
            report=analyze_campaign(workspace,workspace/"report",replicates=10,plots=False)
            self.assertEqual(report["completion"],"COMPLETE")
            self.assertTrue((workspace/"report/attainment.csv").is_file())
            ledger=workspace/"verdicts/main/synthetic_old.json"
            changed=read(ledger);changed["row"]["success"]=False;dump(ledger,changed)
            with self.assertRaisesRegex(ValueError,"hash"):
                analyze_campaign(workspace,workspace/"tampered_report",replicates=10,plots=False)
            # A cached trial verdict must not bypass raw-map identity checking.
            Path(info["map_path"]).write_bytes(Path(info["map_path"]).read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"raw map/cloud"):
                validate_map(workspace,info,[entry])


if __name__=="__main__":
    unittest.main()
