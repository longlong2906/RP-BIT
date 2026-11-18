"""Real OMPL integration on synthetic correctness fixtures, not benchmark data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import test_safe_length_v2_worker as fixture

WORKER = None


class CorridorWorkerTests(unittest.TestCase):
    save_field = fixture.V2WorkerTests.save_field

    def setUp(self):
        fixture.V2WorkerTests.setUp(self)
        self.scorer.risk[:] = 1.
        self.scorer.risk[28:32, 26:34] = 12.
        self.save_field()
        self.request['variant_id'] = 'synthetic_corridor_v3'
        self.request['work_iterations'] = 1600
        policy = self.request['safe_length']
        policy.pop('risk_slack')
        policy.update(policy_id='a3t_corridor_v3', risk_cap=40., corridor_policy=dict(
            alpha=1.10,label_bins=16,dilation_cells=1,max_labels=2000000,max_memory_bytes=536870912))

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, name, request=None, expected=0):
        directory = self.folder/name
        directory.mkdir()
        path = directory/'request.json'
        path.write_text(json.dumps(request or self.request), encoding='utf-8')
        process = subprocess.run([WORKER,str(path),str(directory)],capture_output=True,text=True,timeout=120)
        (directory/'stdout.log').write_text(process.stdout,encoding='utf-8')
        (directory/'stderr.log').write_text(process.stderr,encoding='utf-8')
        self.assertEqual(process.returncode,expected,process.stderr)
        if expected:
            self.assertFalse((directory/'path_xy.csv').exists())
            return directory,{}
        return directory,json.loads((directory/'worker.json').read_text())

    def assert_corridor(self, worker):
        c = worker['corridor']
        self.assertEqual(c['status'],'GUIDANCE_READY')
        self.assertFalse(c['guide_used_as_incumbent'])
        self.assertEqual(worker['effective_safe_length']['domain'],'immutable_centerline_corridor')
        self.assertTrue(worker['effective_safe_length']['thresholds_immutable_during_solve'])
        self.assertLessEqual(c['guide_length'],c['alpha']*c['L0']+1e-8)
        self.assertEqual(len(c['mask']),3600)
        self.assertTrue(set(c['mask']) <= {0,1})
        score = self.scorer.score_path(c['guide_xy'])
        self.assertTrue(score['valid'])
        self.assertAlmostEqual(score['pri'],c['guide_pri'],delta=1e-10)
        self.assertTrue(self.oracle.validate_path(c['guide_xy'],self.request['start'],self.request['goal'])['valid'])
        for i,j in self.scorer.path_cells(c['guide_xy']):
            self.assertEqual(c['mask'][j*60+i],1)

    def test_bitstar_hooks_snapshots_and_fixed_work_reproducibility(self):
        directory,worker = self.launch('serial')
        self.assert_corridor(worker)
        self.assertEqual(worker['ompl_version'],'1.7.0')
        self.assertTrue(worker['planner_solve_called'])
        self.assertEqual(worker['planner_status'],'EXACT_SOLUTION')
        policy = worker['effective_policy']
        self.assertEqual(policy['motion_objective'],'surface_length')
        self.assertEqual(policy['nearest_neighbor_metric'],'euclidean_xy')
        self.assertTrue(policy['k_nearest'])
        self.assertFalse(policy['jit'])
        for key in ('motion_cost_calls','motion_heuristic_calls','cost_to_go_calls','atlas_sample_proposals'):
            self.assertGreater(worker['counters'][key],0,key)
        self.assertEqual(worker['batch_transition_count'],1)
        previous = math.inf
        for event in map(json.loads,(directory/'progress.jsonl').read_text().splitlines()):
            if event['event_type'] != 'improvement':
                continue
            path = fixture.read_path(directory/'snapshots'/event['snapshot_id']/'path_xy.csv')
            self.assertTrue(self.oracle.validate_path(path,self.request['start'],self.request['goal'],
                claimed_cost=event['best_cost_terrain_m'])['valid'])
            for i,j in self.scorer.path_cells(path):
                self.assertEqual(worker['corridor']['mask'][j*60+i],1)
            self.assertLessEqual(event['best_cost_terrain_m'],previous+1e-8)
            previous = event['best_cost_terrain_m']
        with ThreadPoolExecutor(max_workers=2) as pool:
            repetitions = list(pool.map(self.launch,('parallel_a','parallel_b')))
        for other_dir,other in repetitions:
            self.assertEqual(other['corridor']['guide_xy'],worker['corridor']['guide_xy'])
            self.assertEqual(other['corridor']['mask'],worker['corridor']['mask'])
            self.assertEqual(other['counters'],worker['counters'])
            self.assertEqual(fixture.read_path(other_dir/'path_xy.csv'),fixture.read_path(directory/'path_xy.csv'))

    def test_zero_budget_never_uses_guide_as_incumbent(self):
        request = copy.deepcopy(self.request)
        request.update(planning_budget_s=0.,work_iterations=0)
        directory,worker = self.launch('zero',request)
        self.assert_corridor(worker)
        self.assertFalse(worker['path_available'])
        self.assertIsNone(worker['best_cost_terrain_m'])
        self.assertFalse((directory/'path_xy.csv').exists())
        self.assertEqual(worker['improvement_count'],0)

    def test_resource_limit_and_disconnection_are_recorded(self):
        request = copy.deepcopy(self.request)
        request['safe_length']['corridor_policy']['max_labels'] = 1
        directory,worker = self.launch('resource',request)
        self.assertEqual(worker['termination_reason'],'GUIDANCE_RESOURCE_LIMIT')
        self.assertFalse(worker['planner_solve_called'])
        self.assertFalse(worker['path_available'])
        self.scorer.valid[:,29:31] = False
        self.save_field()
        directory,worker = self.launch('disconnected')
        self.assertEqual(worker['termination_reason'],'GUIDANCE_NO_CONNECTION')
        self.assertFalse(worker['planner_solve_called'])

    def test_matched_common_uses_identical_domain(self):
        _,worker = self.launch('a3')
        request = copy.deepcopy(self.request)
        request.update(planner_id='sem_rrtstar_common_v1',variant_id='synthetic_common_corridor_v3',
            sampler_query_cache=False,batch_after_first_solution=0,samples_per_batch=200,
            sem_range_m=1.,sem_neighbor_radius_m=3.)
        request['safe_length']['atlas_sampling'] = False
        _,common = self.launch('common',request)
        self.assert_corridor(common)
        self.assertEqual(worker['corridor']['guide_xy'],common['corridor']['guide_xy'])
        self.assertEqual(worker['corridor']['mask'],common['corridor']['mask'])
        self.assertEqual(common['effective_policy']['motion_objective'],'surface_length')

    def test_unknown_fields_and_ceiling_changes_fail_closed(self):
        for index,extra in enumerate(({'risk_cap':41.},{'slope_cap_deg':34.},
                                     {'risk_slack':0.},{'unregistered_weight':1.})):
            request = copy.deepcopy(self.request)
            request['safe_length'].update(extra)
            self.launch('rejected_'+str(index),request,expected=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',required=True)
    args,remaining = parser.parse_known_args()
    WORKER = str(Path(args.worker).resolve())
    os.environ['PATH'] = os.pathsep.join((str(Path(WORKER).parent),str(ROOT/'.deps/install/lib'),
                                         'C:/mingw64/bin',os.environ.get('PATH','')))
    unittest.main(argv=[sys.argv[0],*remaining])
