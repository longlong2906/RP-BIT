"""Labeled synthetic fixtures for the independent v4 validator, never run data."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

from corridor_v4_validation import (VERSION, _stats, cell_delta_diagnostics,
    ordered_subsequence_indices, path_is_simple, reconstruct_preview,
    record_corridor_receipt, validate_trial)
import test_corridor_v3_validation as legacy_tests
from test_corridor_v3_validation import synthetic_field, write_path
from run_safe_length import dump, read
from native_paper_oracle import NativePaperOracle
from terrain_io import write_map
from corridor_v4_validation import risk_exposure_v4, _original_parameter_exposure, _candidate_surface_length
from corridor_v3_validation import risk_exposure as legacy_risk_exposure


class CorridorV4ValidationTests(unittest.TestCase):
    def fixture(self, directory, selector="preview_pri", dilation=1, failure=False):
        request, info, worker, oracle, field = legacy_tests.CorridorValidationTests().fixture(directory, failure=failure)
        policy = request["safe_length"]
        policy["policy_id"] = "a3t_corridor_v4"
        policy["corridor_policy"].update(alpha=1.2, selector=selector, dilation_cells=dilation,
            representatives_per_bin=4, max_preview_candidates=64, max_preview_edge_checks=2000000)
        worker["requested_safe_length"] = copy.deepcopy(policy)
        worker["effective_safe_length"]["policy_id"] = policy["policy_id"]
        corridor = worker["corridor"]
        corridor.update(policy["corridor_policy"])
        corridor.update(preview_s=.01, preview_used_as_incumbent=False,
                        preview_length_decrease_ulps=64,preview_length_decrease_scale="max(1,abs(old),abs(new))",
                        counters=dict(labels_allocated=1,peak_memory_bytes=1000,preview_edge_checks=0))
        path = corridor["guide_xy"]
        corridor.update(raw_guide_xy=path, raw_guide_file=None if failure else "corridor_raw_guide_xy.csv",
                        preview_xy=path if not failure and selector == "preview_pri" else [],
                        preview_file="corridor_preview_xy.csv" if not failure and selector == "preview_pri" else None,
                        raw_guide_stats=None, preview_stats=None, candidate_diagnostics=[])
        if not failure:
            stats = _stats(oracle, field, path)
            corridor["raw_guide_stats"] = stats
            write_path(directory/"corridor_raw_guide_xy.csv", path)
            if selector == "preview_pri":
                corridor["preview_stats"] = stats
                write_path(directory/"corridor_preview_xy.csv", path)
            corridor["candidate_diagnostics"] = [dict(source="shortest", raw_node_ids=[961,962], raw_xy=path,
                preview_xy=corridor["preview_xy"], raw_stats=stats, preview_stats=corridor["preview_stats"],
                selected=True, preview_edge_checks=0)]
            from corridor_v3_validation import reconstruct_mask
            corridor["mask"] = reconstruct_mask(field, path, dilation)
        dump(directory/"worker.json", worker)
        return request, info, worker, oracle, field

    def check(self, directory, request, info, oracle, field, status="COMPLETED"):
        record_corridor_receipt(directory)
        return validate_trial(directory, request, info, dict(status=status, field_build_s=0.), oracle=oracle, field=field)

    def test_exact_simple_path_predicates_include_touch_overlap_and_crossing(self):
        for path in ([[0.,0.],[1.,1.],[0.,1.],[1.,0.]],
                     [[0.,0.],[2.,0.],[1.,0.]],
                     [[0.,0.],[2.,0.],[2.,1.],[1.,0.],[1.,-1.]],
                     [[0.,0.],[1.,0.],[1.,1.],[0.,0.]]):
            self.assertFalse(path_is_simple(path), path)
        for path in ([[0.,0.]], [[0.,0.],[0.,0.],[1.,0.],[2.,0.]],
                     [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]):
            self.assertTrue(path_is_simple(path), path)
        # Exact binary coordinates on opposite sides of a line are not joined
        # by an arbitrary physical tolerance.
        self.assertTrue(path_is_simple([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,float(np.nextafter(0.,1.))]]))

    def test_preview_is_farthest_visible_strict_shortening_and_subsequence(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_", dir=ROOT/"tests/fixtures") as tmp:
            _, _, _, oracle, field = self.fixture(Path(tmp))
            raw = [[.7,1.2],[1.,1.2],[1.,1.3],[1.8,1.3],[1.8,1.2],[2.2,1.2]]
            preview, checks = reconstruct_preview(oracle, field, raw)
            self.assertEqual(preview, [raw[0],raw[-1]])
            self.assertEqual(checks,1)
            self.assertEqual(ordered_subsequence_indices(raw,preview),[0,5])
            self.assertIsNone(ordered_subsequence_indices(raw,[raw[0],[1.1,1.2],raw[-1]]))
            straight=[[.7,1.2],[1.,1.2],[2.2,1.2]]
            same,_=reconstruct_preview(oracle,field,straight)
            self.assertEqual(same,straight)
            with self.assertRaisesRegex(ValueError,"preview_edge_check_limit"):
                reconstruct_preview(oracle,field,raw,0)

    def test_xy_shortcut_that_increases_surface_length_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            terrain=Path(tmp)/"synthetic_xy_surface_conflict.t25d"
            heights=[[.4*(2.-abs(x-3.))-.4*abs(y-3.) for x in range(7)] for y in range(7)]
            write_map(terrain,heights,spacing=(1.,1.),radius=.1,margin=0.)
            oracle=NativePaperOracle(terrain)
            field=synthetic_field(np.ones((6,6)))
            field.resolution=1.
            field.bounds=(0.,0.,6.,6.)
            raw=[[1.,3.],[3.,3.1],[5.,3.]]
            direct_xy=math.dist(raw[0],raw[-1])
            bent_xy=math.fsum(math.dist(a,b) for a,b in zip(raw,raw[1:]))
            direct_surface=oracle.surface(raw[0],raw[-1])[0]
            bent_surface=math.fsum(oracle.surface(a,b)[0] for a,b in zip(raw,raw[1:]))
            self.assertLess(direct_xy,bent_xy)
            self.assertGreater(direct_surface,bent_surface)
            self.assertLess(oracle._swept_slope(raw[0],raw[-1]),35.)
            preview,_=reconstruct_preview(oracle,field,raw)
            self.assertEqual(preview,raw)

    def test_exact_collinear_certificate_preserves_curve_and_attempt_limit(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp)
            _,_,_,oracle,field=self.fixture(directory)
            raw=[[.7+1.5*i/360,1.2] for i in range(361)]
            preview,checks=reconstruct_preview(oracle,field,raw)
            self.assertEqual(preview,raw)
            self.assertEqual(checks,64620)
            with self.assertRaisesRegex(ValueError,"preview_edge_check_limit"):
                reconstruct_preview(oracle,field,raw,64619)
            terrain=directory/"synthetic_collinear_terrain_fold.t25d"
            write_map(terrain,[[abs(.5*i-1.5)*.2 for i in range(7)] for _ in range(7)],
                      spacing=(.5,.5),radius=.1,margin=0.)
            folded=NativePaperOracle(terrain)
            path=[[.5+i/8,1.5] for i in range(17)]
            direct=folded.surface(path[0],path[-1])[0]
            split=math.fsum(folded.surface(a,b)[0] for a,b in zip(path,path[1:]))
            self.assertAlmostEqual(direct,split,places=12)
            preview,checks=reconstruct_preview(folded,field,path)
            self.assertEqual(preview,path)
            self.assertEqual(checks,120)

    def test_raw_preview_and_narrow_mask_modes_validate_with_diagnostics(self):
        for selector,dilation in (("raw_pri",1),("preview_pri",1),("preview_pri",0)):
            with self.subTest(selector=selector,dilation=dilation), tempfile.TemporaryDirectory(
                    prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                directory=Path(tmp)
                request,info,_,oracle,field=self.fixture(directory,selector,dilation)
                result=self.check(directory,request,info,oracle,field)
                self.assertEqual(result["correctness_errors"],[])
                self.assertTrue(result["success"])
                self.assertTrue(result["corridor"]["independent_valid"])
                self.assertEqual(result["cell_delta_diagnostics"]["selected_guide_to_final"]["added"]["count"],0)
                self.assertEqual(read(directory/"corridor_receipt.json")["version"],VERSION)

    def test_preview_mask_and_provenance_tampering_fail_closed(self):
        for tamper in ("hash","mask","stats","preview","resource"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                directory=Path(tmp)
                request,info,worker,oracle,field=self.fixture(directory)
                if tamper=="hash":
                    record_corridor_receipt(directory)
                    worker["corridor"]["total_s"]+=.01
                elif tamper=="mask": worker["corridor"]["mask"]=[1]*900
                elif tamper=="stats": worker["corridor"]["raw_guide_stats"]=dict(worker["corridor"]["raw_guide_stats"],pri=1.)
                elif tamper=="resource": worker["corridor"]["counters"]["preview_edge_checks"]=2000001
                else:
                    worker["corridor"]["preview_xy"]=[request["start"],[1.3,1.2],request["goal"]]
                    write_path(directory/"corridor_preview_xy.csv",worker["corridor"]["preview_xy"])
                dump(directory/"worker.json",worker)
                if tamper!="hash": record_corridor_receipt(directory)
                result=validate_trial(directory,request,info,dict(status="COMPLETED"),oracle=oracle,field=field)
                self.assertFalse(result["success"])
                self.assertTrue(result["correctness_errors"])

    def test_guidance_failure_and_interruption_do_not_become_success(self):
        for failure,status in ((True,"COMPLETED"),(False,"INTERRUPTED")):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                directory=Path(tmp)
                request,info,_,oracle,field=self.fixture(directory,failure=failure)
                result=self.check(directory,request,info,oracle,field,status)
                self.assertEqual(result["correctness_errors"],[])
                self.assertFalse(result["success"])
                self.assertFalse(result["attained_safe"])

    def test_cell_delta_retains_unique_cells_and_risk_sum_identity(self):
        field=synthetic_field(np.arange(100).reshape(10,10)/10)
        a=[[.15,.25],[.85,.25]]
        b=[[.15,.25],[.15,.35],[.85,.35],[.85,.25]]
        result=cell_delta_diagnostics(field,a,b)
        self.assertAlmostEqual(result["after"]["risk_sum"]-result["before"]["risk_sum"],
            result["added"]["risk_sum"]-result["removed"]["risk_sum"])
        self.assertEqual(result,cell_delta_diagnostics(field,a[::-1],b[::-1]))

    def test_cell_delta_distinguishes_equal_sum_count_and_exposes_dilution(self):
        field=synthetic_field(np.full((10,10),2.))
        start,goal=[.15,.25],[.85,.25]
        upper=[start,[.15,.35],[.85,.35],goal]
        lower=[start,[.15,.15],[.85,.15],goal]
        result=cell_delta_diagnostics(field,upper,lower)
        self.assertEqual(result["before"]["risk_count"],result["after"]["risk_count"])
        self.assertEqual(result["before"]["risk_sum"],result["after"]["risk_sum"])
        self.assertGreater(result["added"]["count"],0)
        self.assertGreater(result["removed"]["count"],0)
        self.assertNotEqual(result["added"]["cells"],result["removed"]["cells"])
        dilution_field=synthetic_field(np.zeros((10,10)))
        dilution_field.risk[2,1]=10.
        dilution=cell_delta_diagnostics(dilution_field,[start,goal],upper)
        self.assertEqual(dilution["before"]["risk_sum"],dilution["after"]["risk_sum"])
        self.assertGreater(dilution["after"]["risk_count"],dilution["before"]["risk_count"])
        self.assertLess(dilution["after"]["pri"],dilution["before"]["pri"])

    def test_last_in_budget_invalid_incumbent_is_not_rescued(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp)
            request,info,worker,oracle,field=self.fixture(directory)
            bad=[request["start"],[2.1,1.2]]
            event=dict(snapshot_id="synthetic_1",elapsed_s=.2,native_objective_value=1.4,cost_terrain_m=1.4,path_xy=bad)
            worker["snapshots"].append(event)
            write_path(directory/"snapshots/synthetic_1/path_xy.csv",bad)
            write_path(directory/"path_xy.csv",bad)
            dump(directory/"worker.json",worker)
            result=self.check(directory,request,info,oracle,field)
            self.assertFalse(result["success"])
            self.assertEqual(result["selected_snapshot_id"],"synthetic_1")
            self.assertIn("synthetic_1:independent_geometry_invalid",result["correctness_errors"])
            self.assertIsNone(result["first_safe_s"])

    def test_late_event_is_checked_without_replacing_in_budget_selection(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            directory=Path(tmp)
            request,info,worker,oracle,field=self.fixture(directory)
            event=dict(worker["snapshots"][0],snapshot_id="synthetic_late",elapsed_s=2.1)
            worker["snapshots"].append(event)
            write_path(directory/"snapshots/synthetic_late/path_xy.csv",event["path_xy"])
            dump(directory/"worker.json",worker)
            result=self.check(directory,request,info,oracle,field)
            self.assertTrue(result["success"])
            self.assertEqual(result["selected_snapshot_id"],"synthetic_0")
            self.assertEqual(result["late_snapshot_count"],1)

    def test_candidate_node_ids_and_ranking_are_audited(self):
        for mutation in ("ids","rank"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(prefix="corridor_v4_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                directory=Path(tmp)
                request,info,worker,oracle,field=self.fixture(directory)
                candidate=worker["corridor"]["candidate_diagnostics"][0]
                if mutation=="ids":candidate["raw_node_ids"][0]=0
                else:
                    other=copy.deepcopy(candidate)
                    other.update(selected=False,source="beam")
                    other["preview_stats"]["pri"]=-1.
                    worker["corridor"]["candidate_diagnostics"].append(other)
                dump(directory/"worker.json",worker)
                result=self.check(directory,request,info,oracle,field)
                self.assertFalse(result["success"])
                expected="candidate_node_ids_mismatch:0" if mutation=="ids" else "selected_candidate_not_minimum_published_PRI_length"
                self.assertIn(expected,result["correctness_errors"])


class ExposureRepairTests(unittest.TestCase):
    def test_candidate_length_near_diagonal_exact_coverage(self):
        with tempfile.TemporaryDirectory(prefix="v4_candidate_length_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"synthetic_flat_near_diagonal.t25d"
            write_map(path,np.zeros((101,101)).tolist(),spacing=(.2,.2))
            oracle=NativePaperOracle(path)
            a,b=[13.3,5.300000000000001],[18.,10.]
            with self.assertRaisesRegex(ValueError,"NoData gap"):
                oracle.surface(a,b)
            expected=math.dist(a,b)
            self.assertAlmostEqual(_candidate_surface_length(oracle,a,b),expected,places=12)
            self.assertAlmostEqual(_candidate_surface_length(oracle,b,a),expected,places=12)
            mid=[(x+y)/2 for x,y in zip(a,b)]
            self.assertAlmostEqual(_candidate_surface_length(oracle,a,mid)+_candidate_surface_length(oracle,mid,b),expected,places=12)
            ordinary=[[2.,3.],[6.,4.]]
            self.assertEqual(_candidate_surface_length(oracle,*ordinary),oracle.surface(*ordinary)[0])

    def test_candidate_length_exact_fallback_keeps_true_nodata_and_bounds(self):
        with tempfile.TemporaryDirectory(prefix="v4_candidate_length_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"synthetic_missing_near_diagonal.t25d"
            heights=np.zeros((101,101));heights[37,77]=np.nan
            write_map(path,heights.tolist(),spacing=(.2,.2))
            oracle=NativePaperOracle(path)
            with self.assertRaisesRegex(ValueError,"NoData gap"):
                _candidate_surface_length(oracle,[13.3,5.300000000000001],[18.,10.])
            with self.assertRaisesRegex(ValueError,"outside map"):
                _candidate_surface_length(oracle,[13.3,5.300000000000001],[21.,13.])

    def test_known_rounded_subchords_and_original_triangle_gap(self):
        with tempfile.TemporaryDirectory(prefix="v4_exposure_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"synthetic_plane.t25d"
            heights=[[.1*(i*.2)+.05*(j*.2) for i in range(101)] for j in range(101)]
            write_map(path,heights,spacing=(.2,.2))
            oracle=NativePaperOracle(path);field=synthetic_field(np.full((200,200),2.))
            edges=[[[3.6,9.3],[15.8,9.700000000000001]],
                   [[14.9,13.100000000000001],[18.,10.]],
                   [[13.5,8.3],[13.600000000000001,8.3]]]
            observed_gap=False
            for edge in edges:
                a,b=edge;dx,dy=b[0]-a[0],b[1]-a[1]
                expected=2.*math.sqrt(dx*dx+dy*dy+(.1*dx+.05*dy)**2)
                try:legacy_risk_exposure(oracle,field,edge)
                except ValueError as error:
                    self.assertEqual(str(error),"NoData gap");observed_gap=True
                for points in (edge,edge[::-1],[a,[(a[0]+b[0])/2,(a[1]+b[1])/2],b]):
                    self.assertAlmostEqual(risk_exposure_v4(oracle,field,points),expected,places=10)
                    self.assertAlmostEqual(_original_parameter_exposure(oracle,field,points),expected,places=10)
            self.assertTrue(observed_gap)

    def test_unchanged_success_and_real_missing_data_remain_failures(self):
        with tempfile.TemporaryDirectory(prefix="v4_exposure_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"synthetic_flat.t25d";write_map(path,np.zeros((11,11)).tolist(),spacing=(.2,.2),radius=.01,margin=0.)
            oracle=NativePaperOracle(path);field=synthetic_field(np.full((20,20),2.));points=[[.3,.3],[1.7,1.6]]
            self.assertEqual(risk_exposure_v4(oracle,field,points),legacy_risk_exposure(oracle,field,points))
            heights=np.zeros((11,11));heights[5,5]=np.nan
            missing=Path(tmp)/"synthetic_missing.t25d";write_map(missing,heights.tolist(),spacing=(.2,.2),radius=.01,margin=0.)
            with self.assertRaisesRegex(ValueError,"NoData gap"):
                risk_exposure_v4(NativePaperOracle(missing),field,[[.3,1.],[1.7,1.]])
            field.valid[10,10]=False
            with self.assertRaisesRegex(ValueError,"missing_sem_support"):
                risk_exposure_v4(oracle,field,[[.3,1.],[1.7,1.]])

    def test_nonplanar_mesh_gradient_changes(self):
        with tempfile.TemporaryDirectory(prefix="v4_exposure_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"synthetic_ridge.t25d"
            write_map(path,[[abs(i-5)*.04 for i in range(11)] for _ in range(11)],spacing=(.2,.2),radius=.01,margin=0.)
            oracle=NativePaperOracle(path);field=synthetic_field(np.full((20,20),3.));points=[[.3,.8],[1.7,1.2]]
            expected=3.*oracle.surface(*points)[0]
            self.assertAlmostEqual(_original_parameter_exposure(oracle,field,points),expected,places=11)



if __name__=="__main__":
    unittest.main()
