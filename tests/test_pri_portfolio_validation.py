"""SYNTHETIC independent-oracle fixtures; not experimental planner results."""
import copy
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField
from pri_portfolio_validation import (audit_selection, path_metrics, reconstruct_masks, validate_trial,
                                      verify_mask_input, reconstruct_numerical_input)
from safe_length_field import file_hash, build_field_file
from native_point_cloud import generate_cloud
from run_safe_length import dump
from terrain_io import write_map
from test_corridor_v3_validation import write_path
from test_safe_length_oracle import write_exchange


def flood_fill_mask_reference(risk, valid, anchor_cells, endpoint_cells):
    """Test-only eight-neighbor flood fill, independent of SciPy labeling.

    Work from the independently reconstructed numerical field, not the
    verified/serialized array being tested. Exact component memberships and
    ranking remain checked even when libm/SciPy change last-bit risks across
    platforms. The arithmetic for PRI and quantiles follows the registration.
    """
    ny, nx = risk.shape
    anchor = {j*nx+i for i,j in anchor_cells}
    endpoints = {j*nx+i for i,j in endpoint_cells}
    values = sorted(float(risk.ravel()[cell]) for cell in anchor)
    pri = float(sum(map(Fraction, values), Fraction())/len(values))
    rankings, thresholds = [], []
    for fraction in (.9, .75):
        position = fraction*(len(values)-1)
        lower = int(position)
        quantile = values[lower]+(position-lower)*(values[min(lower+1,len(values)-1)]-values[lower])
        threshold = max(pri, quantile)
        thresholds.append(threshold)
        pending = {j*nx+i for j in range(ny) for i in range(nx)
                   if valid[j,i] and math.isfinite(float(risk[j,i])) and threshold < risk[j,i] <= 40.}
        ranked = []
        while pending:
            seed = min(pending)
            pending.remove(seed)
            component, frontier = {seed}, [seed]
            while frontier:
                cell = frontier.pop()
                row, column = divmod(cell, nx)
                for y in range(max(0,row-1),min(ny,row+2)):
                    for x in range(max(0,column-1),min(nx,column+2)):
                        neighbor = y*nx+x
                        if neighbor in pending:
                            pending.remove(neighbor)
                            component.add(neighbor)
                            frontier.append(neighbor)
            crossing = component & anchor
            if crossing and component.isdisjoint(endpoints):
                surplus = math.fsum(float(risk.ravel()[cell])-pri for cell in sorted(crossing))
                ranked.append((-surplus,min(component),component))
        rankings.append(sorted(ranked,key=lambda item:item[:2]))
    return [dict(threshold=threshold,excluded_cells=sorted(set().union(*(item[2] for item in ranked[:count]))))
            for threshold,ranked,count in ((thresholds[0],rankings[0],1),
                                          (thresholds[1],rankings[1],1),
                                          (thresholds[1],rankings[1],2))]


class PortfolioValidationTests(unittest.TestCase):
    def workspace(self):
        return tempfile.TemporaryDirectory(prefix="synthetic_pri_validator_", dir=ROOT/"tests/fixtures")

    def test_portable_exact_component_reference_preserves_ties_and_endpoint_exclusion(self):
        risk=np.zeros((5,40),dtype=np.float64);valid=np.ones_like(risk,dtype=bool)
        risk[2:4,10]=4.;risk[2:4,30]=4.;risk[2,1]=8.
        anchor=[(i,2) for i in range(1,39)];endpoints={(1,2),(38,2)}
        expected=flood_fill_mask_reference(risk,valid,anchor,endpoints)
        # Two equal-surplus components tie by the smallest row-major cell;
        # the higher-risk component touching the start must be excluded.
        known=[[90,130],[90,130],[90,110,130,150]]
        self.assertEqual([row['excluded_cells'] for row in expected],known)
        field=SimpleNamespace(risk=risk,valid=valid,
            path_cells=lambda points:[tuple(points[0])] if len(points)==1 else anchor)
        actual=reconstruct_masks(field,[[1,2],[38,2]],[1,2],[38,2])
        self.assertEqual([row['excluded_cells'] for row in actual],known)
        self.assertEqual([row['oracle_status'] for row in actual],['READY','DUPLICATE','READY'])

    def fixture(self, directory, *, detour=False):
        terrain = directory/"synthetic_map.t25d"
        write_map(terrain, np.zeros((7,7)).tolist(), spacing=(.5,.5))
        x,y = np.meshgrid(np.arange(.025,3.,.05),np.arange(.025,3.,.05))
        cloud = directory/"synthetic_cloud.npy"
        np.save(cloud, np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size))), allow_pickle=False)
        info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud),
                    cloud_sha256=file_hash(cloud), bounds=[0.,0.,3.,3.])
        field = SharedSEMField(np.load(cloud,allow_pickle=False), info["bounds"])
        field.risk,field.valid = field.risk.copy(),field.valid.copy()
        if detour:
            field.risk[12:14,14:16] = 20.
        oracle = NativePaperOracle(terrain)
        exchange=directory/"synthetic_field.txt"
        write_exchange(exchange,field.valid,field.risk,bounds=info["bounds"])
        safety = dict(policy_id="a3t_safe_length_v1",risk_cap=40.,slope_cap_deg=35.,atlas_sampling=True,
                      field_path=str(exchange),field_sha256=file_hash(exchange))
        request = dict(start=[.7,1.25],goal=[2.2,1.25],planning_budget_s=2.,planner_id="bitstar_a3",
                       objective_type="surface_length",safe_length=safety,pri_portfolio=dict(length_cap_ratio=1.1))
        paths = [[request["start"],request["goal"]]]
        if detour:
            paths.append([request["start"],[1.45,1.45],request["goal"]])
        candidates = []
        for i,path in enumerate(paths):
            values = path_metrics(oracle,field,path,exposure=True)
            candidates.append(dict(snapshot_id=f"synthetic_{i}",event_type="improvement",path_xy=path,
                                   elapsed_s=.2+i,score_completed_s=.2+i,callback_s=.1+i,episode_id=0 if i==0 else 2,
                                   callback_id=0,anchor_phase=i==0,eligible=True,published=True,feasible=True,
                                   reason="ANCHOR_SELECTED" if i==0 else "ELIGIBLE",**values))
        worker = dict(requested_safe_length=copy.deepcopy(safety),effective_safe_length=copy.deepcopy(safety),
                      requested_pri_portfolio=copy.deepcopy(request["pri_portfolio"]),effective_pri_portfolio=copy.deepcopy(request["pri_portfolio"]),
                      objective_type="surface_length",planner_status="EXACT_SOLUTION",execution_status="COMPLETED",
                      portfolio_stage="ANCHOR_LOCKED",actual_solve_wall_s=2.001,path_available=True,
                      anchor_snapshot_id="synthetic_0",selected_snapshot_id=candidates[-1]["snapshot_id"],
                      length_cap_m=1.1*candidates[0]["cost_terrain_m"],
                      effective_policy=dict(planner_class="ompl::geometric::BITstar",motion_objective="surface_length",
                          k_nearest=True,jit=False,safety_atlas_sampling=True,edge_lower_bound="d_phi",cost_to_go_lower_bound="d_phi",
                          informed_sampling="I_phi",external_length_cap_in_sampler=False,anchor_injected_as_incumbent=False),
                      episodes=[dict(episode_id=0,planner_instance_id=1,fresh_planner=True,anchor_injected=False,solve_called=True)])
        if detour:
            worker["episodes"].append(dict(episode_id=2,planner_instance_id=2,fresh_planner=True,anchor_injected=False,solve_called=True))
        masks = reconstruct_masks(field,paths[0],request["start"],request["goal"])
        mask_rows = []
        for mask in masks:
            empty = mask["oracle_status"] == "EMPTY"
            row = dict(episode_id=mask["episode_id"],threshold=mask["threshold"],excluded_cells=mask["excluded_cells"],
                       mask_sha256=None if empty else mask["mask_sha256"], mask_file=None,
                       status={"EMPTY":"SKIPPED_NO_COMPONENT","DUPLICATE":"SKIPPED_DUPLICATE","READY":"READY"}[mask["oracle_status"]])
            if not empty:
                row["mask_file"] = f"masks/mask_{mask['episode_id']}.bin"
                path = directory/row["mask_file"]; path.parent.mkdir(parents=True,exist_ok=True)
                data = np.ones(field.risk.size,dtype=np.uint8);data[row["excluded_cells"]] = 0
                path.write_bytes(data.tobytes())
            mask_rows.append(row)
        dump(directory/"portfolio_masks.json",dict(nx=field.risk.shape[1],ny=field.risk.shape[0],encoding="uint8_row_major_allowed",masks=mask_rows))
        self.save(directory,worker,candidates)
        return request,info,worker,oracle,field,candidates

    def save(self,directory,worker,candidates):
        for row in candidates:
            write_path(directory/"snapshots"/row["snapshot_id"]/"path_xy.csv",row["path_xy"])
        worker["snapshots"] = [dict(r) for r in candidates if r["published"]]
        selected = next((r for r in candidates if r["snapshot_id"] == worker.get("selected_snapshot_id")),None)
        if selected:
            write_path(directory/"path_xy.csv",selected["path_xy"])
            worker.update({k:selected[k] for k in ("cost_terrain_m","best_cost_terrain_m","length_xy_m","objective_value","pri","risk_sum","risk_count","sem_exposure_risk_m","sem_exposure_normalized_m")})
        dump(directory/"worker.json",worker)
        (directory/"portfolio_candidates.jsonl").write_text("".join(json.dumps(r)+"\n" for r in candidates),encoding="utf-8")
        (directory/"progress.jsonl").write_text("".join(json.dumps(r)+"\n" for r in worker["snapshots"]),encoding="utf-8")

    def check(self,directory,fixture,**execution):
        request,info,_,oracle,field,_ = fixture
        return validate_trial(directory,request,info,dict(status="COMPLETED",field_build_s=0.,**execution),
                              oracle=oracle,field=field,numerical_reference=field)

    def test_raw_reconstruction_and_constant_zero_risk(self):
        with self.workspace() as tmp:
            directory=Path(tmp); fixture=self.fixture(directory)
            request,info=fixture[:2]
            row=validate_trial(directory,request,info,dict(status="COMPLETED",field_build_s=0.))
            self.assertTrue(row["success"],row["correctness_errors"])
            self.assertEqual(row["final_pri"],0.)
            self.assertEqual(row["final_exposure"],0.)
            self.assertAlmostEqual(row["final_cost_terrain_m"],1.5)

    def test_pri_improvement_can_increase_length_under_cap(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory,detour=True)
            a,b=fixture[-1]
            self.assertGreater(b["cost_terrain_m"],a["cost_terrain_m"])
            self.assertLess(b["pri"],a["pri"])
            row=self.check(directory,fixture)
            self.assertTrue(row["success"],row["correctness_errors"])
            self.assertEqual(row["selected_snapshot_id"],"synthetic_1")

    def test_unique_cells_split_reverse_and_repetition_invariant(self):
        with self.workspace() as tmp:
            fixture=self.fixture(Path(tmp),detour=True)
            oracle,field=fixture[3:5];a,b=fixture[0]["start"],fixture[0]["goal"]
            paths=([a,b],[b,a],[a,[1.45,1.25],b],[a,b,a,b])
            metrics=[path_metrics(oracle,field,p) for p in paths]
            self.assertEqual({m["risk_count"] for m in metrics}, {metrics[0]["risk_count"]})
            self.assertEqual({m["pri"] for m in metrics}, {metrics[0]["pri"]})
            self.assertAlmostEqual(metrics[-1]["cost_terrain_m"],3*metrics[0]["cost_terrain_m"])

    def test_tampered_pri_and_length_are_rejected(self):
        for key in ("pri","cost_terrain_m","risk_count"):
            with self.subTest(key=key),self.workspace() as tmp:
                directory=Path(tmp);fixture=self.fixture(directory)
                fixture[-1][0][key]+=1
                self.save(directory,fixture[2],fixture[-1])
                row=self.check(directory,fixture)
                self.assertFalse(row["success"])
                self.assertTrue(any("metric_mismatch:"+key in e for e in row["correctness_errors"]))

    def test_cap_violation_and_invalid_final_never_rescue_anchor(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory,detour=True)
            fixture[0]["pri_portfolio"]["length_cap_ratio"]=1.
            fixture[2]["requested_pri_portfolio"]["length_cap_ratio"]=1.
            fixture[2]["effective_pri_portfolio"]["length_cap_ratio"]=1.
            fixture[2]["length_cap_m"]=fixture[-1][0]["cost_terrain_m"]
            self.save(directory,fixture[2],fixture[-1])
            row=self.check(directory,fixture)
            self.assertFalse(row["success"])
            self.assertEqual(row["selected_snapshot_id"],"synthetic_1")
            self.assertTrue(any("cap_or_deadline" in e for e in row["correctness_errors"]))

    def test_late_scored_candidate_kept_but_never_selected(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory,detour=True)
            late=fixture[-1][1]
            late.update(elapsed_s=2.1,score_completed_s=2.1,eligible=False,published=False,reason="SCORING_AFTER_DEADLINE")
            fixture[2]["selected_snapshot_id"]="synthetic_0"
            self.save(directory,fixture[2],fixture[-1])
            row=self.check(directory,fixture)
            self.assertTrue(row["success"],row["correctness_errors"])
            self.assertEqual(row["candidate_count"],2)
            self.assertEqual(row["selected_snapshot_id"],"synthetic_0")

    def test_false_rejection_of_better_candidate_is_detected(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory,detour=True)
            fixture[-1][1].update(eligible=False,published=False)
            fixture[2]["selected_snapshot_id"]="synthetic_0"
            self.save(directory,fixture[2],fixture[-1])
            row=self.check(directory,fixture)
            self.assertFalse(row["success"])
            self.assertIn("synthetic_1:candidate_admission_mismatch",row["correctness_errors"])

    def test_masks_preserve_endpoint_components_and_use_closed_boundaries(self):
        with self.workspace() as tmp:
            fixture=self.fixture(Path(tmp),detour=True);field=fixture[4]
            field.risk[:]=1.
            field.risk[12,7:9]=20.  # component includes start cell, must remain allowed
            masks=reconstruct_masks(field,[fixture[0]["start"],fixture[0]["goal"]],fixture[0]["start"],fixture[0]["goal"])
            self.assertTrue(all(not m["excluded_cells"] for m in masks))
            p=[1.,1.]
            touched=field.path_cells([p])
            self.assertEqual(len(touched),4)

    def test_mask_tampering_or_planner_instance_reuse_fails(self):
        for mode in ("mask","instance"):
            with self.subTest(mode=mode),self.workspace() as tmp:
                directory=Path(tmp);fixture=self.fixture(directory,detour=True)
                if mode=="mask":
                    data=json.loads((directory/"portfolio_masks.json").read_text())
                    data["masks"][1]["excluded_cells"]=[]
                    dump(directory/"portfolio_masks.json",data)
                else:
                    fixture[2]["episodes"][-1]["planner_instance_id"]=1
                    self.save(directory,fixture[2],fixture[-1])
                self.assertFalse(self.check(directory,fixture)["success"])

    def test_no_anchor_quarter_straddler_is_kept_and_omission_detected(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory)
            candidate=fixture[-1][0]
            candidate.update(elapsed_s=.6,score_completed_s=.6,reason="NO_ANCHOR_FALLBACK")
            fixture[2].update(anchor_snapshot_id=None,length_cap_m=None,portfolio_stage="NO_ANCHOR_CONTINUE")
            self.save(directory,fixture[2],fixture[-1])
            self.assertTrue(self.check(directory,fixture)["success"])
            candidate["eligible"]=False
            self.save(directory,fixture[2],fixture[-1])
            self.assertIn("synthetic_0:fallback_candidate_admission_mismatch",self.check(directory,fixture)["correctness_errors"])

    def test_invalid_support_and_incomplete_endpoint_cannot_be_success(self):
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory)
            fixture[4].valid[12,15]=False
            row=self.check(directory,fixture)
            self.assertFalse(row["success"])
            self.assertTrue(row["correctness_errors"])
        with self.workspace() as tmp:
            directory=Path(tmp);fixture=self.fixture(directory)
            fixture[0]["goal"]=[2.3,1.25]
            self.assertFalse(self.check(directory,fixture)["success"])

    def test_verified_numerical_input_rejects_tampering_and_hard_boundary_change(self):
        for mode in ("hash","risk","one_bit_risk","support","hard_boundary","grid","unsupported_encoding"):
            with self.subTest(mode=mode),self.workspace() as tmp:
                directory=Path(tmp);fixture=self.fixture(directory)
                raw=fixture[4];policy=fixture[0]["safe_length"];path=Path(policy["field_path"])
                risk=raw.risk.copy();support=raw.valid.copy();bounds=raw.bounds
                if mode in ("hash","risk"):
                    risk[12,15]+=1e-10
                elif mode=="one_bit_risk":
                    risk[12,15]=np.nextafter(0.,np.inf)  # far smaller than the old envelope
                elif mode=="support":
                    support[12,15]=False
                elif mode=="hard_boundary":
                    raw.risk[12,15]=40.;risk[12,15]=np.nextafter(40.,np.inf)
                elif mode=="grid":
                    bounds=(0.,0.,3.1,3.)
                elif mode=="unsupported_encoding":
                    raw.valid[0,0]=False;support[0,0]=False
                write_exchange(path,support,risk,bounds=bounds)
                if mode=="unsupported_encoding":
                    lines=path.read_text().splitlines();lines[2]="0 1";path.write_text("\n".join(lines)+"\n")
                if mode!="hash":
                    policy["field_sha256"]=file_hash(path)
                with self.assertRaises(ValueError):
                    verify_mask_input(raw,policy,raw)

    def test_20m_tilted_rounding_changes_components_but_verified_input_replays_exactly(self):
        with self.workspace() as tmp:
            directory=Path(tmp);terrain=directory/"synthetic_tilted_20m.t25d"
            write_map(terrain,[[.24*(i*.2)+.10*(j*.2)+2. for i in range(101)] for j in range(101)],
                      spacing=(.2,.2),metadata=dict(test_fixture=True,purpose="synthetic_20m_rounding_regression"))
            cloud=directory/"synthetic_cloud.npy";np.save(cloud,generate_cloud(terrain),allow_pickle=False)
            exchange=directory/"synthetic_field.txt"
            metadata=build_field_file(cloud,terrain,[0.,0.,20.,20.],exchange)
            raw=SharedSEMField(np.load(cloud,allow_pickle=False),[0.,0.,20.,20.])
            numerical=reconstruct_numerical_input(np.load(cloud,allow_pickle=False),raw.bounds)
            mask_input,audit=verify_mask_input(raw,dict(field_path=str(exchange),field_sha256=metadata["field_sha256"]),numerical)
            self.assertGreater(audit["different_binary64_risk_cells"],0)
            self.assertLess(audit["maximum_absolute_risk_difference"],1e-13)
            path=[[2.,10.],[18.,10.]]
            verified_masks=reconstruct_masks(mask_input,path,*path)
            raw_masks=reconstruct_masks(raw,path,*path)
            expected=flood_fill_mask_reference(numerical.risk,numerical.valid,raw.path_cells(path),
                set(raw.path_cells([path[0]]))|set(raw.path_cells([path[-1]])))
            for mask,reference in zip(verified_masks,expected,strict=True):
                self.assertEqual(mask["threshold"],reference["threshold"])
                self.assertEqual(mask["excluded_cells"],reference["excluded_cells"])
                bitmap=np.ones(numerical.risk.size,dtype=np.uint8)
                bitmap[reference["excluded_cells"]]=0
                self.assertEqual(mask["mask_sha256"],hashlib.sha256(bitmap.tobytes()).hexdigest())
            self.assertTrue(any(reference["excluded_cells"] for reference in expected))
            self.assertNotEqual([m["excluded_cells"] for m in verified_masks],[m["excluded_cells"] for m in raw_masks])
            np.testing.assert_array_equal(mask_input.risk[numerical.valid].view(np.uint64),
                                          numerical.risk[numerical.valid].view(np.uint64))
            self.assertFalse(hasattr(mask_input,"score_path"))
            self.assertIsNot(mask_input.risk,raw.risk)

    def test_mixed_41003_exact_replay_preserves_physical_oracle_and_detects_one_bit_change(self):
        from g5_dataset import geometry_definition
        with self.workspace() as tmp:
            directory=Path(tmp);terrain=directory/"synthetic_mixed_41003.t25d"
            height,rectangles,_,_=geometry_definition("mixed_obstacles",41003)
            write_map(terrain,[[height(i*.2,j*.2) for i in range(201)] for j in range(201)],
                      spacing=(.2,.2),rectangles=rectangles,
                      metadata=dict(test_fixture=True,purpose="synthetic_exact_arithmetic_regression"))
            cloud=directory/"synthetic_cloud.npy";np.save(cloud,generate_cloud(terrain),allow_pickle=False)
            exchange=directory/"synthetic_field.txt"
            metadata=build_field_file(cloud,terrain,[0.,0.,40.,40.],exchange)
            points=np.load(cloud,allow_pickle=False)
            raw=SharedSEMField(points,[0.,0.,40.,40.])
            original_risk=raw.risk.copy()
            numerical=reconstruct_numerical_input(points,raw.bounds)
            policy=dict(field_path=str(exchange),field_sha256=metadata["field_sha256"])
            mask_input,audit=verify_mask_input(raw,policy,numerical)
            self.assertTrue(audit["exact_numerical_replay"])
            # The number of old-envelope outliers varies with numerical
            # libraries. Check the full diagnostic against a scalar reference
            # and require this fixture still exercises an actual outlier.
            deltas=[abs(float(a)-float(b)) for a,b in zip(numerical.risk[numerical.valid],raw.risk[raw.valid],strict=True)]
            allowances=[128.*2.**-52*max(1.,abs(float(value))) for value in raw.risk[raw.valid]]
            outliers=sum(delta>allowance for delta,allowance in zip(deltas,allowances,strict=True))
            self.assertEqual(audit["old_rounding_envelope_violation_count"],outliers)
            self.assertGreater(outliers,0)
            self.assertEqual(audit["maximum_old_rounding_envelope_ratio"],
                             max(delta/allowance for delta,allowance in zip(deltas,allowances,strict=True)))
            self.assertGreater(audit["maximum_old_rounding_envelope_ratio"],1.)
            np.testing.assert_array_equal(raw.risk,original_risk)
            np.testing.assert_array_equal(mask_input.risk[numerical.valid].view(np.uint64),
                                          numerical.risk[numerical.valid].view(np.uint64))
            self.assertTrue(mask_input.valid[135,343])
            self.assertTrue(math.isfinite(float(mask_input.risk[135,343])))
            lines=exchange.read_text().splitlines()
            changed=np.nextafter(mask_input.risk[135,343],np.inf)
            lines[2+135*400+343]="1 "+format(changed,".17g")
            exchange.write_text("\n".join(lines)+"\n",encoding="ascii")
            policy["field_sha256"]=file_hash(exchange)
            with self.assertRaisesRegex(ValueError,"exact_arithmetic_replay_mismatch"):
                verify_mask_input(raw,policy,numerical)


if __name__=="__main__":
    unittest.main()
