"""Synthetic statistical and independent cost-audit fixtures; no benchmark data."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from native_paper_analysis import *
from native_paper_analysis import _validate_group
from terrain_io import write_map


def fake_row(backend='a3',length=10.,turns=1,status='COMPLETED',success=True,seed=1):
    return dict(trial_id=f'fixture_{backend}_{seed}',backend=backend,family='fixture',map_id='fixture',planner_seed=seed,
        execution_status=status,success=success,budget_s=15.,timing_stratum='fixture-machine',solve_time_s=15.,first_solution_s=1.,first_valid_latency_s=2.,
        events=[dict(elapsed_s=1.,independent_valid=True,metrics=dict(cost_terrain_m=length,length_xy_m=length,sharp_turn_count=turns,
            minimum_footprint_clearance_m=1.,minimum_obstacle_footprint_clearance_m=1.,maximum_swept_footprint_slope_deg=0.))] if success else [])


class NativeAnalysis(unittest.TestCase):
    def setUp(self):
        self.fixtures=ROOT/'tests/fixtures/native_paper'
        self.fixtures.mkdir(parents=True,exist_ok=True)

    def test_ratio_and_failure_denominator(self):
        rows=[fake_row(length=10,turns=1),fake_row(length=30,turns=2,seed=2),fake_row(status='CRASH',success=False,seed=3),fake_row(status='NOT_RUN',success=False,seed=4)]
        s=group_summary(rows)
        self.assertEqual((s['planned'],s['attempted'],s['success']),(4,3,2))
        self.assertAlmostEqual(s['psr'],2/3)
        self.assertAlmostEqual(s['sti'],3/40)
        self.assertEqual(s['cost_terrain_m']['n'],2)
        self.assertIsNone(group_summary([fake_row(status='TIMEOUT',success=False)])['sti'])

    def test_turn_threshold_duplicates_and_reversal(self):
        self.assertEqual(sharp_turn_count([[0,0],[1,0],[1,0],[1,1],[0,1]]),2)
        self.assertEqual(sharp_turn_count([[0,0],[1,0],[1+math.cos(math.pi/6),.5]]),0)
        self.assertEqual(sharp_turn_count([[0,0],[1,0],[0,0]]),1)

    def test_native_putn_formula(self):
        points=[dict(traversability=.2,center_of_mass=[0,0,.7]),dict(traversability=.4,center_of_mass=[3,4,.7])]
        self.assertAlmostEqual(putn_path_cost(points),5*(1+.1*(1/.8+1/.6-2)))
        points[0]['traversability']=1.
        with self.assertRaises(ValueError):putn_path_cost(points)

    def test_last_native_incumbent_does_not_reselect_geometry(self):
        row=fake_row(length=10)
        row['events'].append(dict(elapsed_s=2.,native_objective_value=11.,independent_valid=True,metrics={'cost_terrain_m':12.}))
        self.assertEqual(final_metrics(row)['cost_terrain_m'],12.)

    def test_paired_coverage_and_zero_success(self):
        rows=[fake_row('a3',length=8.),fake_row('sem',length=10.),fake_row('a3',seed=2,status='CRASH',success=False),fake_row('sem',seed=2)]
        a=paired_bootstrap(rows,'sem','cost_terrain_m',50)
        self.assertEqual(a['usable_pairs'],1);self.assertEqual(a['ci95'],[-2.,-2.])
        self.assertEqual(paired_bootstrap(rows,'sem','psr',50)['usable_pairs'],2)
        self.assertIsNone(paired_bootstrap(rows,'putn','cost_terrain_m',50)['ci95'])
        rows[1]['timing_stratum']='different-machine'
        time_pairs=paired_bootstrap(rows,'sem','solve_time_s',50)
        self.assertEqual(time_pairs['planned_pairs'],2)
        self.assertEqual(time_pairs['excluded_stratum_mismatch'],1)
        self.assertEqual(time_pairs['coverage'],0.)
        self.assertEqual(paired_bootstrap(rows,'sem','cost_terrain_m',50)['usable_pairs'],1)

    def test_native_raster_inventory_is_bound_to_worker(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_raster_hash_',dir=self.fixtures) as d:
            root=Path(d);p=root/'raster.npz';p.write_bytes(b'PARSER_FIXTURE_ONLY');meta=root/'raster.json';meta.write_text('{}')
            hashes={p.name:sha256(p),meta.name:sha256(meta)}
            verify_native_artifacts(root,hashes)
            p.write_bytes(b'CHANGED');meta.write_text(json.dumps({'new_hash':sha256(p)}))
            with self.assertRaisesRegex(ValueError,'inventory_or_hash'):verify_native_artifacts(root,hashes)
            p.unlink()
            with self.assertRaises(ValueError):verify_native_artifacts(root,hashes)

    def test_receipts_and_geometry_reject_tampered_native_cost(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_validation_',dir=self.fixtures) as d:
            root=Path(d);mp=root/'fixture.t25d';write_map(mp,[[0.]*5 for _ in range(5)])
            request=dict(map_path=str(mp),start=[1,1],goal=[3,1],planning_budget_s=15.)
            entry=dict(trial_id='synthetic_sgp_fixture',backend='sgp',track='native',family='fixture',map_id='fixture',query_id='q0',planner_seed=1,execution_status='COMPLETED',map_sha256=sha256(mp),request=request)
            trial=root/entry['trial_id'];trial.mkdir();dump(trial/'request.json',request)
            snapshot=dict(snapshot_id='s0',path_xy=[[1,1],[3,1]],elapsed_s=.1,native_objective_value=2.)
            worker=dict(status='EXACT_SOLUTION',snapshots=[snapshot],phase_timings=dict(solve_s=15.,setup_s=.1),
                native_model=dict(cpu_flush_denormal=True,numeric_policy=copy.deepcopy(SGP_NUMERIC_POLICY)))
            dump(trial/'worker.json',worker);(trial/'path_xy.csv').write_text('x,y\n1,1\n3,1\n')
            task=(root,[entry],validation_code_identity())
            first=_validate_group(task)[0]
            self.assertTrue(first['success'],first['errors']);self.assertFalse(first['validation_cache_hit'])
            self.assertTrue(_validate_group(task)[0]['validation_cache_hit'])
            dump(root/'manifest.json',dict(entries=[entry]))
            dump(root/'validation_summary.json',dict(manifest_sha256=sha256(root/'manifest.json'),raw_hashes=raw_hashes(root),
                validation_code_identity=validation_code_identity(),rows=[first],correctness_failures=0))
            summary=analyze_campaign(root,root/'report')
            self.assertEqual(summary['tables'][0]['sti'],0.)
            self.assertTrue((root/'report/aggregate.csv').is_file())
            self.assertTrue((root/'report/paired_comparisons.csv').is_file())
            self.assertFalse(list((root/'report').glob('*.png')))
            worker['snapshots'][0]['native_objective_value']=1.;dump(trial/'worker.json',worker)
            changed=_validate_group(task)[0]
            self.assertFalse(changed['success']);self.assertIn('native_objective_mismatch',changed['errors'])
            self.assertFalse(changed['validation_cache_hit'])
            dump(root/'validation_summary.json',dict(manifest_sha256=sha256(root/'manifest.json'),raw_hashes=raw_hashes(root),
                validation_code_identity=validation_code_identity(),rows=[changed],correctness_failures=1))
            self.assertEqual(analyze_campaign(root,root/'report')['completion'],'FAILED_CORRECTNESS')

    def test_sgp_policy_required_for_model_but_not_construction_failure(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_sgp_policy_',dir=self.fixtures) as d:
            root=Path(d);mp=root/'fixture.t25d';write_map(mp,[[0.]*5 for _ in range(5)])
            request=dict(map_path=str(mp),start=[1,1],goal=[3,1],planning_budget_s=15.)
            entry=dict(trial_id='synthetic_sgp_policy',backend='sgp',family='fixture',map_id='fixture',planner_seed=1,
                execution_status='COMPLETED',map_sha256=sha256(mp),request=request)
            trial=root/entry['trial_id'];trial.mkdir();dump(trial/'request.json',request)
            model=dict(cpu_flush_denormal=True,numeric_policy=copy.deepcopy(SGP_NUMERIC_POLICY))
            for status,metadata,expect_error in [('ERROR',{},False),('MODEL_ERROR',{},False),('TIMEOUT',{},True),
                ('TIMEOUT',model,False),('INVALID_NATIVE_ENDPOINTS',{**model,'cpu_flush_denormal':False},True)]:
                dump(trial/'worker.json',dict(status=status,snapshots=[],native_model=metadata))
                verdict=_validate_group((root,[entry],validation_code_identity()))[0]
                self.assertEqual('sgp_cpu_numeric_policy_mismatch' in verdict['errors'],expect_error,(status,metadata,verdict))
                self.assertFalse(verdict['success'])
            bad=copy.deepcopy(model);bad['numeric_policy']['enabled']=1
            self.assertFalse(sgp_numeric_policy_valid(bad))


if __name__=='__main__':unittest.main()
