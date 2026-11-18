"""SYNTHETIC oracle/selector fixtures, never experimental planner results."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
sys.path.insert(0, str(ROOT/"tests"))

from pri_pathsearch_validation import ALPHAS, POLICY, _MemoField, path_metrics, simple_path, validate_trial
from run_safe_length import dump
import test_pri_portfolio_validation as portfolio_fixtures
from test_corridor_v3_validation import write_path


class PathSearchValidationTests(unittest.TestCase):
    def workspace(self):
        return portfolio_fixtures.PortfolioValidationTests().workspace()

    def fixture(self, directory, *, detour=False, mode="full", late_anchor=False):
        request, info, _, oracle, field, old = portfolio_fixtures.PortfolioValidationTests().fixture(directory, detour=detour)
        request.pop("pri_portfolio")
        request["pri_pathsearch"] = dict(mode=mode)
        candidates = []
        for index, row in enumerate(old):
            item = dict(row)
            item.update(snapshot_id=f"candidate_{index}", candidate_id=index, proposal_id=index,
                        parent_ids=[] if index == 0 else ["candidate_0"],
                        move_type="bitstar" if index == 0 else "deform", accepted=True,
                        pool_admitted=True, published_alphas=[], anchor_phase=index == 0,
                        elapsed_s=.2 if index == 0 else 1.2,
                        score_completed_s=.2 if index == 0 else 1.2, callback_s=.1 if index == 0 else 1.1)
            candidates.append(item)
        if late_anchor:
            candidates[0].update(score_completed_s=.6, elapsed_s=.6, callback_s=.55)
        anchored = mode != "length" and not late_anchor
        counters = dict(proposals=int(detour), shortcut_proposals=0, deform_proposals=int(detour), splice_proposals=0,
                        accepted=len(candidates), pool_evictions=0, edge_cache_hits=0, edge_evaluations=len(candidates),
                        incidence_adds=int(detour), incidence_removes=int(detour), full_recomputations=len(candidates),
                        current_pool_size=len(candidates), peak_pool_size=len(candidates), peak_edge_cache_bytes=1000,
                        archive_bytes=1000, rejected={})
        policy = request["safe_length"]
        worker = dict(backend_id="a3t_pri_pathsearch_v1", objective_type="surface_length", planner_instances=1, state_dimension=2, ompl_version="1.7.0",
            requested_pri_pathsearch=copy.deepcopy(request["pri_pathsearch"]), effective_pri_pathsearch=dict(POLICY, mode=mode),
            requested_safe_length=copy.deepcopy(policy), effective_safe_length=copy.deepcopy(policy),
            effective_policy=dict(planner_class="ompl::geometric::BITstar", motion_objective="surface_length", k_nearest=True,
                jit=False, safety_atlas_sampling=True, edge_lower_bound="d_phi", cost_to_go_lower_bound="d_phi", informed_sampling="I_phi",
                external_length_cap_in_sampler=False, anchor_injected_as_incumbent=False, nearest_neighbor_metric="euclidean_xy", pruning=True),
            timing_origin="before_initial_policy_and_planner_creation", primary_output_alpha=1.1,
            candidate_archive="pathsearch_candidates.jsonl", planner_status="EXACT_SOLUTION", execution_status="COMPLETED",
            actual_solve_wall_s=2.001, anchor_snapshot_id="candidate_0" if anchored else None,
            anchor_locked_s=.5 if anchored else None, pathsearch_stage="ANCHOR_LOCKED" if anchored else "LENGTH_BASELINE" if mode == "length" else "NO_ANCHOR_CONTINUE",
            optimizer_resource_limit=False, candidate_count=len(candidates), search_counters=counters,
            episodes=[dict(episode_id=0, planner_instance_id=1, anchor_injected=False, phase="anchor", fresh_planner=True)])
        events = []
        for index, candidate in enumerate(candidates):
            for alpha in ALPHAS:
                chosen = index == 0 or (anchored and candidate["cost_terrain_m"] <= alpha*candidates[0]["cost_terrain_m"]
                                        and candidate["pri"] < candidates[0]["pri"])
                if chosen:
                    candidate["published_alphas"].append(alpha)
                    events.append(dict(candidate, event_id=f"publication_{len(events)}", alpha=alpha,
                                       elapsed_s=candidate["score_completed_s"], publication_reason="LENGTH_FALLBACK" if index == 0 else "PRI_IMPROVEMENT"))
            if index == 0 and anchored:
                for alpha in ALPHAS:
                    events.append(dict(candidate, event_id=f"publication_{len(events)}", alpha=alpha,
                                       elapsed_s=.5, publication_reason="ANCHOR_LOCK"))
        worker["snapshots"] = events
        worker["selectors"] = []
        metric_keys = ("cost_terrain_m", "best_cost_terrain_m", "length_xy_m", "objective_value", "pri", "risk_sum", "risk_count",
                       "sem_exposure_risk_m", "sem_exposure_normalized_m")
        for alpha in ALPHAS:
            cap = alpha*candidates[0]["cost_terrain_m"] if anchored else None
            eligible = [c for c in candidates if cap is None or c["cost_terrain_m"] <= cap]
            key = (lambda c:(c["pri"], c["cost_terrain_m"], c["candidate_id"])) if anchored else (lambda c:(c["cost_terrain_m"], c["pri"], c["candidate_id"]))
            chosen = min(eligible, key=key)
            worker["selectors"].append(dict(alpha=alpha, length_cap_m=cap, selected_snapshot_id=chosen["snapshot_id"],
                                             path_available=True, **{k:chosen[k] for k in metric_keys}))
        selected = next(c for c in candidates if c["snapshot_id"] == worker["selectors"][-1]["selected_snapshot_id"])
        worker.update(selected_snapshot_id=selected["snapshot_id"], path_available=True,
                      **{k:selected[k] for k in metric_keys})
        fixture = request, info, worker, oracle, field, candidates
        self.save(directory, fixture)
        return fixture

    def save(self, directory, fixture):
        _, _, worker, _, _, candidates = fixture
        for candidate in candidates:
            write_path(directory/"snapshots"/candidate["snapshot_id"]/"path_xy.csv", candidate["path_xy"])
        selected = next((c for c in candidates if c["snapshot_id"] == worker.get("selected_snapshot_id")), None)
        if selected:
            write_path(directory/"path_xy.csv", selected["path_xy"])
        dump(directory/"worker.json", worker)
        (directory/"pathsearch_candidates.jsonl").write_text("".join(json.dumps(c)+"\n" for c in candidates), encoding="utf-8")
        (directory/"progress.jsonl").write_text("".join(json.dumps(c)+"\n" for c in worker["snapshots"]), encoding="utf-8")

    def check(self, directory, fixture, **execution):
        request, info, _, oracle, field, _ = fixture
        return validate_trial(directory, request, info, {"status":"COMPLETED", "field_build_s":0., **execution},
                              oracle=oracle, field=field, numerical_reference=field)

    def test_raw_reconstruction_all_four_selectors(self):
        with self.workspace() as tmp:
            directory = Path(tmp); fixture = self.fixture(directory)
            row = validate_trial(directory, fixture[0], fixture[1], dict(status="COMPLETED", field_build_s=0.))
            self.assertTrue(row["success"], row["correctness_errors"])
            self.assertEqual(len(row["selector_results"]), 4)
            self.assertEqual(row["final_pri"], 0.)
            self.assertEqual(row["final_exposure"], 0.)
            self.assertAlmostEqual(row["final_cost_terrain_m"], 1.5)
            self.assertEqual(row["first_safe_s"], .2)
            self.assertEqual(len(row["events"]), 8)

    def test_caps_select_different_whole_paths_without_four_runs(self):
        with self.workspace() as tmp:
            directory = Path(tmp); fixture = self.fixture(directory, detour=True)
            row = self.check(directory, fixture)
            self.assertTrue(row["success"], row["correctness_errors"])
            self.assertEqual([s["selected_snapshot_id"] for s in row["selector_results"]], ["candidate_0", "candidate_0", "candidate_1", "candidate_1"])
            self.assertGreater(row["final_cost_terrain_m"], 1.5)
            self.assertLess(row["final_pri"], row["selector_results"][0]["final_pri"])

    def test_length_and_no_anchor_scalar_fallback(self):
        for mode, late in (("length", False), ("full", True)):
            with self.subTest(mode=mode), self.workspace() as tmp:
                directory = Path(tmp); fixture = self.fixture(directory, mode=mode, late_anchor=late)
                row = self.check(directory, fixture)
                self.assertTrue(row["success"], row["correctness_errors"])
                self.assertEqual(row["selector_results"], [] if mode == "length" else row["selector_results"])
                self.assertIsNone(row["length_cap_m"])

    def test_invalid_endpoint_invalidates_all_selectors_no_rescue(self):
        with self.workspace() as tmp:
            directory = Path(tmp); fixture = self.fixture(directory, detour=True)
            fixture[0]["goal"] = [2.3, 1.25]
            row = self.check(directory, fixture)
            self.assertFalse(row["success"])
            self.assertTrue(all(not s["success"] for s in row["selector_results"]))
            self.assertEqual(row["selected_snapshot_id"], "candidate_1")
            self.assertTrue(any("hard_safety" in error for error in row["correctness_errors"]))

    def test_tampered_stats_and_field_are_rejected(self):
        for key in ("cost_terrain_m", "pri", "risk_count"):
            with self.subTest(key=key), self.workspace() as tmp:
                directory = Path(tmp); fixture = self.fixture(directory)
                fixture[-1][0][key] += 1
                self.save(directory, fixture)
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertTrue(any("metric_mismatch:"+key in error for error in row["correctness_errors"]))

    def test_reject_parent_cycle_and_false_ablation(self):
        for issue in ("parent", "mode", "instances"):
            with self.subTest(issue=issue), self.workspace() as tmp:
                directory = Path(tmp); fixture = self.fixture(directory, detour=True)
                if issue == "parent":
                    fixture[-1][1]["parent_ids"] = ["candidate_1"]
                elif issue == "mode":
                    fixture[0]["pri_pathsearch"]["mode"] = "shortcut"
                    fixture[2]["requested_pri_pathsearch"]["mode"] = "shortcut"
                    fixture[2]["effective_pri_pathsearch"]["mode"] = "shortcut"
                else:
                    fixture[2]["planner_instances"] = 2
                self.save(directory, fixture)
                self.assertFalse(self.check(directory, fixture)["success"])

    def test_selector_tampering_is_not_repaired_by_oracle(self):
        with self.workspace() as tmp:
            directory = Path(tmp); fixture = self.fixture(directory, detour=True)
            fixture[2]["selectors"][-1]["selected_snapshot_id"] = "candidate_0"
            self.save(directory, fixture)
            row = self.check(directory, fixture)
            self.assertFalse(row["success"])
            self.assertEqual(row["selector_results"][-1]["selected_snapshot_id"], "candidate_0")
            self.assertTrue(any("prevalidation_selection" in error for error in row["correctness_errors"]))

    def test_late_scoring_and_backdated_publication_fail(self):
        for issue in ("late", "backdate"):
            with self.subTest(issue=issue), self.workspace() as tmp:
                directory = Path(tmp); fixture = self.fixture(directory, detour=True)
                if issue == "late":
                    fixture[-1][1].update(score_completed_s=2.1, elapsed_s=2.1)
                else:
                    fixture[2]["snapshots"][-1]["elapsed_s"] = .9
                self.save(directory, fixture)
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertIsNone(row["first_safe_s"])

    def test_interrupted_attempt_keeps_metrics_not_success(self):
        with self.workspace() as tmp:
            directory = Path(tmp); fixture = self.fixture(directory)
            row = self.check(directory, fixture, status="INTERRUPTED")
            self.assertFalse(row["success"])
            self.assertIsNone(row["first_safe_s"])
            self.assertAlmostEqual(row["final_cost_terrain_m"], 1.5)

    def test_simple_path_exact_contacts_and_adjacent_ulp(self):
        self.assertTrue(simple_path([[0., 0.], [1., 0.], [2., 0.]]))
        self.assertTrue(simple_path([[0., 0.], [0., 0.], [1., 0.]]))
        self.assertFalse(simple_path([[0., 0.], [2., 0.], [1., 0.]]))
        self.assertFalse(simple_path([[0., 0.], [2., 2.], [0., 2.], [2., 0.]]))
        self.assertFalse(simple_path([[0., 0.], [2., 0.], [1., 1.], [1., 0.]]))
        self.assertTrue(simple_path([[0., 0.], [2., 0.], [1., 1.], [1., float(np.nextafter(0., 1.))]]))

    def test_unique_cells_overlap_split_reverse_and_ulp_memo(self):
        with self.workspace() as tmp:
            fixture = self.fixture(Path(tmp), detour=True)
            oracle, field = fixture[3:5]
            a, b = fixture[0]["start"], fixture[0]["goal"]
            paths = [[a,b], [b,a], [a,[1.45,1.25],b], [a,b,a,b]]
            memo = _MemoField(field)
            values = [path_metrics(oracle, memo, path) for path in paths]
            self.assertEqual({v["risk_count"] for v in values}, {values[0]["risk_count"]})
            self.assertEqual({v["pri"] for v in values}, {values[0]["pri"]})
            for shift in (-math.inf, math.inf):
                point = [float(np.nextafter(1., shift)), 1.]
                self.assertEqual(memo.path_cells([point, b]), field.path_cells([point, b]))
            self.assertGreater(memo.hits, 0)

    def test_missing_snapshot_progress_and_no_worker(self):
        for issue in ("snapshot", "progress", "worker"):
            with self.subTest(issue=issue), self.workspace() as tmp:
                directory = Path(tmp); fixture = self.fixture(directory)
                target = directory/"snapshots/candidate_0/path_xy.csv" if issue == "snapshot" else directory/(issue+".jsonl" if issue == "progress" else "worker.json")
                target.unlink()
                row = self.check(directory, fixture)
                self.assertFalse(row["success"])
                self.assertTrue(row["correctness_errors"])


if __name__ == "__main__":
    unittest.main()
