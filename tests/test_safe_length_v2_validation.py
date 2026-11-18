"""Synthetic oracle/provenance fixtures, never benchmark observations."""
import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_safe_length import dump,read
from safe_length_field import file_hash
from safe_length_v2_validation import (validate_trial,record_calibration_receipt,witness_on_registered_graph,
                                        risk_cap_comparison)
from terrain_io import write_map


def csv_path(path,points):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["x","y"]);writer.writerows(points)


class ValidationTests(unittest.TestCase):
    def fixture(self,directory,*,failure=False):
        terrain=directory/"synthetic.t25d"
        write_map(terrain,np.zeros((11,11)).tolist(),spacing=(1.,1.))
        x,y=np.meshgrid((np.arange(100)+.5)*.1,(np.arange(100)+.5)*.1)
        cloud=directory/"synthetic.cloud.npy"
        np.save(cloud,np.column_stack((x.ravel(),y.ravel(),np.zeros(10000))),allow_pickle=False)
        info=dict(map_path=str(terrain),map_sha256=file_hash(terrain),cloud_path=str(cloud),
                  cloud_sha256=file_hash(cloud),bounds=[0.,0.,10.,10.])
        policy=dict(policy_id="a3t_safe_length_v2",risk_slack=.5,slope_cap_deg=35.,atlas_sampling=False)
        request=dict(start=[2.,5.],goal=[8.,5.],planning_budget_s=2.,planner_id="sem_rrtstar_common_v1",
                     samples_per_batch=200,batch_after_first_solution=0,safe_length=policy)
        path=[request["start"],request["goal"]]
        calibration=dict(status="CALIBRATION_NO_CONNECTION" if failure else "CALIBRATED",
            b_graph=None if failure else 0.,effective_risk_cap=None if failure else .5,risk_slack=.5,
            grid_spacing_m=.1,endpoint_radius_m=.2,witness_used_as_incumbent=False,preprocessing_s=.01,
            risk_ceiling=40.,slope_cap_deg=35.,endpoint_radius_tie_ulps=16,
            endpoint_radius_tie_scale="max(1,abs(endpoint_xy),abs(grid_node_xy),radius)",
            witness_file=None if failure else "calibration_witness_xy.csv",witness_xy=[] if failure else path)
        effective=dict(policy_id=policy["policy_id"],risk_cap=None if failure else .5,slope_cap_deg=35.,
            atlas_sampling=False,thresholds_immutable_during_solve=True,witness_used_as_incumbent=False)
        worker=dict(planner_status="NO_SOLUTION" if failure else "EXACT_SOLUTION",calibration=calibration,
            requested_safe_length=policy,effective_safe_length=effective,snapshots=[],planner_solve_called=not failure)
        if not failure:
            csv_path(directory/"calibration_witness_xy.csv",path)
            csv_path(directory/"snapshots/synthetic_0/path_xy.csv",path)
            csv_path(directory/"path_xy.csv",path)
            worker["snapshots"]=[dict(snapshot_id="synthetic_0",elapsed_s=.1,native_objective_value=6.,
                                       cost_terrain_m=6.,path_xy=path)]
        dump(directory/"worker.json",worker)
        return request,info,worker

    def test_real_independent_raw_field_and_geometry_validate(self):
        with tempfile.TemporaryDirectory(prefix="v2_validation_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,_=self.fixture(directory)
            record_calibration_receipt(directory)
            result=validate_trial(directory,request,info,dict(status="COMPLETED",field_build_s=0.))
            self.assertEqual(result["correctness_errors"],[])
            self.assertTrue(result["success"])
            self.assertTrue(result["calibration"]["independent_valid"])
            self.assertEqual(result["final_pri"],0.)

    def test_wrong_cap_is_rejected_even_with_matching_receipt(self):
        with tempfile.TemporaryDirectory(prefix="v2_cap_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,worker=self.fixture(directory)
            worker["calibration"]["effective_risk_cap"]=2.
            worker["effective_safe_length"]["risk_cap"]=2.
            dump(directory/"worker.json",worker);record_calibration_receipt(directory)
            result=validate_trial(directory,request,info,dict(status="COMPLETED"))
            self.assertFalse(result["success"])
            self.assertIn("calibration_cap_formula_mismatch",result["correctness_errors"])

    def test_tampered_worker_and_witness_are_detected(self):
        with tempfile.TemporaryDirectory(prefix="v2_hash_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,worker=self.fixture(directory)
            record_calibration_receipt(directory)
            worker["calibration"]["witness_xy"]=[[2.,5.],[7.,5.]]
            dump(directory/"worker.json",worker)
            csv_path(directory/"calibration_witness_xy.csv",worker["calibration"]["witness_xy"])
            result=validate_trial(directory,request,info,dict(status="COMPLETED"))
            self.assertFalse(result["success"])
            for error in ("calibration_worker_hash_mismatch","calibration_witness_hash_mismatch",
                          "independent_calibration_witness_invalid"):
                self.assertIn(error,result["correctness_errors"])

    def test_graph_failure_never_counts_as_planner_success(self):
        with tempfile.TemporaryDirectory(prefix="v2_failure_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,_=self.fixture(directory,failure=True)
            record_calibration_receipt(directory)
            result=validate_trial(directory,request,info,dict(status="COMPLETED"))
            self.assertEqual(result["correctness_errors"],[])
            self.assertFalse(result["success"])
            self.assertFalse(result["attained_safe"])
            self.assertTrue(result["calibration"]["no_continuous_infeasibility_claim"])

    def test_crash_retains_snapshot_without_claiming_success(self):
        with tempfile.TemporaryDirectory(prefix="v2_crash_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,_=self.fixture(directory)
            record_calibration_receipt(directory)
            result=validate_trial(directory,request,info,dict(status="CRASH"))
            self.assertFalse(result["success"])
            self.assertFalse(result["attained_safe"])
            self.assertTrue(result["safe_snapshot_observed_in_failed_attempt"])
            self.assertEqual(len(result["events"]),1)

    def test_safe_non_graph_witness_cannot_certify_minimax(self):
        with tempfile.TemporaryDirectory(prefix="v2_nongraph_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp);request,info,worker=self.fixture(directory)
            # Safe flat geometry, but neither intermediate point nor long edges
            # belong to the declared eight-neighbour .1 m calibration graph.
            witness=[request["start"],[4.05,5.05],request["goal"]]
            worker["calibration"]["witness_xy"]=witness
            dump(directory/"worker.json",worker)
            csv_path(directory/"calibration_witness_xy.csv",witness)
            record_calibration_receipt(directory)
            result=validate_trial(directory,request,info,dict(status="COMPLETED"))
            self.assertFalse(result["success"])
            self.assertIn("calibration_witness_not_on_registered_graph",result["correctness_errors"])

    def test_graph_membership_accepts_zero_endpoint_edge_and_radius_ties(self):
        bounds=[0.,0.,10.,10.]
        start,goal=[2.,5.],[2.4,5.]
        self.assertTrue(witness_on_registered_graph([start,start,[2.1,5.],[2.2,5.],goal],start,goal,bounds))
        self.assertFalse(witness_on_registered_graph([start,[2.3,5.],goal],start,goal,bounds))
        self.assertFalse(witness_on_registered_graph([start,[2.1,5.],[2.3,5.],goal],start,goal,bounds))

    def test_graph_endpoint_identity_accepts_actual_one_ulp_json_roundoff(self):
        # Regression coordinates from the first pilot request. Synthetic graph
        # only: this does not relabel its preserved raw validation verdict.
        start=[4.106925299027667,10.385785123784817]
        goal=[36.47099230680929,10.385785123784817]
        parsed_start=[start[0],10.385785123784816]
        parsed_goal=[goal[0],10.385785123784816]
        self.assertEqual(start[1]-parsed_start[1],math.ulp(start[1]))
        lattice=[[i*.1,10.4] for i in range(41,366)]
        path=[parsed_start,*lattice,parsed_goal]
        self.assertTrue(witness_on_registered_graph(path,start,goal,[0.,0.,40.,40.]))
        # Endpoint ties must not admit an off-grid intermediate vertex.
        path[20]=[path[20][0]+1e-9,path[20][1]]
        self.assertFalse(witness_on_registered_graph(path,start,goal,[0.,0.,40.,40.]))

    def test_graph_endpoint_identity_ties_fail_closed_above_sixteen_ulps(self):
        start,goal=[2.,5.],[2.4,5.]
        bounds=[0.,0.,10.,10.]
        for offset,expected in ((16*math.ulp(5.),True),(17*math.ulp(5.),False),(2e-9,False)):
            with self.subTest(offset=offset):
                parsed_start=[start[0],start[1]+offset]
                parsed_goal=[goal[0],goal[1]+offset]
                path=[parsed_start,[2.1,5.],[2.2,5.],parsed_goal]
                self.assertEqual(witness_on_registered_graph(path,start,goal,bounds),expected)
                self.assertEqual(witness_on_registered_graph([parsed_start,parsed_goal],start,goal,bounds),expected)
                self.assertEqual(witness_on_registered_graph([parsed_start],start,start,bounds),expected)
        # A singleton requires an exactly identical requested start and goal;
        # numerical ties apply to witness parsing, not to physical query identity.
        self.assertFalse(witness_on_registered_graph([start],start,[2.,5.+math.ulp(5.)],bounds))
        self.assertFalse(witness_on_registered_graph([start],start,[2.,5.+17*math.ulp(5.)],bounds))
        self.assertFalse(witness_on_registered_graph([[2.,5.+8*math.ulp(5.)]],
            start,[2.,5.+17*math.ulp(5.)],bounds))

    def test_declared_risk_roundoff_is_fixed_and_fail_closed(self):
        cap=9.
        within=cap+32*math.ulp(cap)
        result=risk_cap_comparison(within,cap)
        self.assertTrue(result["passed"])
        self.assertTrue(result["within_roundoff"])
        self.assertEqual(result["raw_risk_max"],within)
        self.assertEqual(result["numeric_excess"],within-cap)
        self.assertFalse(risk_cap_comparison(cap+65*math.ulp(cap),cap)["passed"])
        self.assertFalse(risk_cap_comparison(cap+1e-8,cap)["passed"])
        self.assertFalse(risk_cap_comparison(within,cap,roundoff_ulps=0)["passed"])

    def test_actual_tilted_raw_cloud_delta_zero_field_parity(self):
        from native_point_cloud import generate_cloud
        from safe_length_field import build_field
        from native_paper_v3_sem_evaluator import SharedSEMField
        with tempfile.TemporaryDirectory(prefix="v2_tilted_parity_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            terrain=Path(tmp)/"synthetic_tilted.t25d"
            write_map(terrain,[[.24*(i*.2)+.10*(j*.2)+2. for i in range(101)] for j in range(101)],spacing=(.2,.2))
            points=generate_cloud(terrain)
            planner=build_field(points,[0.,0.,20.,20.])
            oracle=SharedSEMField(points,[0.,0.,20.,20.])
            self.assertTrue(np.array_equal(planner.support,oracle.valid))
            supported=planner.support
            differences=oracle.risk[supported]-planner.risk[supported]
            scales=np.maximum(1.,np.maximum(np.abs(oracle.risk[supported]),np.abs(planner.risk[supported])))
            allowances=64*np.spacing(scales)
            self.assertTrue(np.all(np.abs(differences)<=allowances))
            j,i=np.unravel_index(np.nanargmax(oracle.risk-planner.risk),planner.risk.shape)
            # At delta=0 the planner field value itself is the threshold. The
            # independent raw-cloud implementation retains its own raw value.
            comparison=risk_cap_comparison(float(oracle.risk[j,i]),float(planner.risk[j,i]))
            self.assertTrue(comparison["passed"])
            self.assertEqual(comparison["raw_risk_max"],float(oracle.risk[j,i]))


if __name__=="__main__":
    unittest.main()
