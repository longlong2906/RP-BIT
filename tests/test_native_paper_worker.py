"""Worker contract fixtures, not benchmark observations or planner substitutes."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import native_paper_worker as worker


class PutnA4OptionsTests(unittest.TestCase):
    """A4 PUTN options (tau slope cap, convergence stop) only for the PUTN backend, in registered form."""

    def request(self, **extra):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        cloud = Path(folder.name)/'cloud.npy'
        np.save(cloud, np.zeros((2, 3)), allow_pickle=False)
        request = dict(backend='putn', track='native', planner_id='putn_native_paper_v2',
                       native_objective_id=worker.OBJECTIVES['putn'], map_path='unused.t25d', cloud_path=str(cloud),
                       cloud_sha256=worker.file_hash(cloud), start=[1., 1.], goal=[2., 2.], planner_seed=1,
                       planning_budget_s=5., bridge_library='unused.dll', representation_config={})
        request.update(extra)
        return request

    def test_registered_putn_options(self):
        stop = dict(relative_eps=1e-3, window_iterations=4000)
        worker.validate_request(self.request())
        worker.validate_request(self.request(putn_tau_slope_cap_deg=35., putn_convergence_stop=stop))
        sem = dict(backend='sem', planner_id='sem_native_paper_v2', native_objective_id=worker.OBJECTIVES['sem'])
        for extra in (dict(putn_tau_slope_cap_deg=35), dict(putn_tau_slope_cap_deg=0.), dict(putn_tau_slope_cap_deg=90.5),
                      dict(putn_convergence_stop=None), dict(putn_convergence_stop=dict(stop, relative_eps=1.)),
                      dict(putn_convergence_stop=dict(stop, extra=1)), dict(sem, putn_tau_slope_cap_deg=35.),
                      dict(sem, putn_convergence_stop=stop), dict(sem_convergence_stop=stop)):
            with self.assertRaises(ValueError, msg=str(extra)):
                worker.validate_request(self.request(**extra))


class NativeWorkerTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tests/fixtures/native_paper'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_worker_contract_', dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.map = self.root/'synthetic.t25d'
        self.map.write_bytes(b'synthetic mocked geometry, never loaded by a real planner')
        self.library = self.root/'synthetic_bridge.dll'
        self.library.write_bytes(b'synthetic mocked library, never executed')
        self.cloud = self.root/'synthetic.cloud.npy'
        np.save(self.cloud, np.array([[.05,.05,0.],[.15,.05,0.]]), allow_pickle=False)
        self.active_solve = False
        self.solver_finished = False
        self.terrain_calls = []

    def run_fixture(self, events):
        case = self
        class Terrain:
            bounds = (0.,0.,4.,4.)
            timings = dict(load_s=0.,risk_preprocessing_s=0.,collision_index_s=0.)
            def __init__(self, library, map_path, paths, *, risk_enabled):
                case.assertFalse(risk_enabled)
            def evaluate_path(self, path):
                case.assertFalse(case.active_solve, 'geometry measurement entered timed callback')
                case.terrain_calls.append([list(p) for p in path])
                return {'cost_terrain_m':2.,'length_xy_m':2.}
            def close(self):pass
        class Model:
            def __init__(self, points, bounds, output_dir=None):
                self.metadata = {'synthetic_test_fixture':True, 'version':'synthetic_sem_model'}
                if output_dir is not None:self.save(output_dir)
            def save(self, directory):
                case.assertTrue(case.solver_finished, 'native raster serialization entered setup/solve')
                directory = Path(directory)
                directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/'raster.npz',risk=np.zeros((2,2)),valid=np.ones((2,2),dtype=bool))
                (directory/'raster.json').write_text(json.dumps({'synthetic_test_fixture':True}),encoding='utf8')
        class Planner:
            def __init__(self, terrain, model):
                self.counters={'synthetic_test_fixture':True}
                self.last_components=None
            def solve(self,start,goal,seed,budget,callback,work_iterations=0):
                case.active_solve=True
                try:
                    for elapsed,cost in events:
                        path=[list(start),list(goal)]
                        raw=[[p[0],p[1],0.] for p in path]
                        self.last_components={'total':cost,'synthetic_test_fixture':True}
                        callback(elapsed,path,raw,cost)
                        # Worker must own its snapshot and native component dict.
                        path[0][0]=-999.;raw[0][0]=-999.;self.last_components['total']=-999.
                finally:
                    case.active_solve=False
                    case.solver_finished=True
                return 'EXACT_SOLUTION'
        bridge=types.ModuleType('terrain_compare_bridge')
        bridge.TerrainBridge=Terrain;bridge.PutnBridge=object
        native=types.ModuleType('sem_native_planner')
        native.SEMNativeModel=Model;native.SEMNativePlanner=Planner
        request=dict(backend='sem',track='native',planner_id='sem_native_paper_v2',
            native_objective_id='sem_length_turn_risk_paper_v2',map_path=str(self.map),
            cloud_path=str(self.cloud),cloud_sha256=worker.file_hash(self.cloud),
            map_sha256=worker.file_hash(self.map),start=[1.,1.],goal=[3.,1.],planner_seed=11,
            planning_budget_s=15.,bridge_library=str(self.library),representation_config={
                'master_cloud_spacing_m':.05,'obstacle_height_m':2.,'sem_resolution_m':.1,
                'putn_voxel_m':.1,'sgp_voxel_m':.2,'observation_noise':'none'},
            work_iterations=0,synthetic_test_fixture=True)
        trial=self.root/'trial';trial.mkdir()
        request_path=trial/'request.json';request_path.write_text(json.dumps(request),encoding='utf8')
        with patch.dict(sys.modules,{'terrain_compare_bridge':bridge,'sem_native_planner':native}):
            code=worker.run(request_path,trial)
        result=json.loads((trial/'worker.json').read_text(encoding='utf8'))
        self.assertEqual(code,0,result)
        self.assertEqual(json.loads((trial/'phase.json').read_text())['phase'],'FINISHED')
        return trial,result

    def test_only_in_budget_incumbents_and_copy_only_callback(self):
        trial,result=self.run_fixture([(.2,10.),(15.,8.),(15.001,7.),(16.,6.)])
        self.assertEqual(result['status'],'EXACT_SOLUTION')
        self.assertTrue(result['exact_goal'])
        self.assertEqual(result['counters']['candidates_after_deadline'],2)
        self.assertEqual([s['elapsed_s'] for s in result['snapshots']],[.2,15.])
        self.assertEqual([s['native_components']['total'] for s in result['snapshots']],[10.,8.])
        self.assertTrue(all(s['path_xy']==[[1.,1.],[3.,1.]] for s in result['snapshots']))
        self.assertEqual(len(self.terrain_calls),2)
        self.assertEqual(len(list((trial/'snapshots').iterdir())),2)
        self.assertEqual(result['final_snapshot_id'],'incumbent_000001')

    def test_first_candidate_late_is_timeout_and_no_fabricated_path(self):
        trial,result=self.run_fixture([(15.01,10.),(16.,9.)])
        self.assertEqual(result['native_reported_status'],'EXACT_SOLUTION')
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertFalse(result['exact_goal'])
        self.assertEqual(result['snapshots'],[])
        self.assertEqual(result['counters']['candidates_after_deadline'],2)
        self.assertFalse((trial/'path_xy.csv').exists())
        self.assertFalse((trial/'snapshots').exists())
        self.assertEqual((trial/'progress.jsonl').read_text(),'')
        self.assertEqual(self.terrain_calls,[])

    def test_native_artifacts_inventory_hashes_persisted_bytes(self):
        trial,result=self.run_fixture([(.1,10.)])
        metadata=result['native_model']
        self.assertFalse(metadata['common_a3_risk_field_enabled'])
        self.assertEqual(metadata['artifact_directory'],'native_model')
        self.assertEqual(set(metadata['raster_artifacts']),{'raster.npz','raster.json'})
        for name,checksum in metadata['raster_artifacts'].items():
            self.assertEqual(checksum,hashlib.sha256((trial/'native_model'/name).read_bytes()).hexdigest())
        self.assertEqual(result['provenance']['bridge_sha256'],worker.file_hash(self.library))
        self.assertEqual(result['provenance']['cloud_sha256'],worker.file_hash(self.cloud))

    def test_hash_works_without_python311_file_digest(self):
        with patch.object(hashlib,'file_digest',side_effect=AssertionError('unavailable in Python3.10'),create=True):
            self.assertEqual(worker.file_hash(self.map),hashlib.sha256(self.map.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
