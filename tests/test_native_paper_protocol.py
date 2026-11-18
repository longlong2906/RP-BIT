"""Planning metadata tests use tiny synthetic files, never campaign results."""
import collections
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from experiment_common import digest,dump
from native_paper_protocol import *


class NativeProtocol(unittest.TestCase):
    def setUp(self):
        self.fixtures=ROOT/'tests/fixtures/native_paper'
        self.fixtures.mkdir(parents=True,exist_ok=True)

    def test_frozen_configuration(self):
        c=load_config(ROOT/'configs/native_paper_kaggle_v2.yaml')
        self.assertEqual(c['profiles']['main']['planning_budget_s'],15.)
        self.assertEqual(c['model']['robot_radius_m']+c['model']['safety_margin_m'],.4)
        self.assertFalse(c['analysis']['shared_sem_evaluator'])
        self.assertFalse(c['analysis']['plot'])
        self.assertEqual(c['execution']['max_workers'],4)
        self.assertEqual(c['execution']['setup_timeout_s'],600.)
        self.assertEqual(c['schema_version'],11)
        self.assertEqual(json.loads((ROOT/'configs/terrain_compare_kaggle_v1.yaml').read_text())['schema_version'],10)

    def test_2000_paired_trials_without_generating_evaluation_clouds(self):
        c=default_config()
        with tempfile.TemporaryDirectory(prefix='synthetic_protocol_',dir=self.fixtures) as d:
            root=Path(d);maps=[]
            # Explicit parser-only bytes: no fabricated planner result is emitted.
            for family in FAMILIES:
                for seed in range(8001,8011):
                    mid=f'{family}_{seed}';path=root/(mid+'.t25d');path.write_text('PARSER_FIXTURE_ONLY')
                    meta=path.with_suffix('.json');dump(meta,{'synthetic_parser_fixture':True})
                    cloud=path.with_suffix('.cloud.npy');cloud.write_bytes(b'PARSER_FIXTURE_ONLY')
                    maps.append(dict(map_id=mid,family=family,map_seed=seed,path=path.name,sha256=sha256(path),metadata_sha256=sha256(meta),
                        cloud_path=cloud.name,cloud_sha256=sha256(cloud),queries=[dict(query_id='q0',start=[2,2],goal=[38,38])]))
            dataset=dict(config_hash=digest(c),profile='main',maps=maps)
            plan=plan_trials(c,'main',dataset,root)
            self.assertEqual(plan['planned_slots'],2000)
            self.assertEqual(plan['phase_timeouts'],dict(setup_s=600.,serialize_s=60.,solve_grace_s=5.))
            self.assertEqual(len({e['map_id'] for e in plan['entries']}),50)
            self.assertEqual(collections.Counter(e['backend'] for e in plan['entries']),dict.fromkeys(BACKENDS,500))
            paired=collections.defaultdict(set)
            for e in plan['entries']:
                paired[e['map_id'],e['planner_seed']].add(e['backend'])
                self.assertEqual(e['request']['planning_budget_s'],15.)
                self.assertEqual(e['request']['native_objective_id'],OBJECTIVES[e['backend']])
                self.assertEqual(e['execution_status'],'PLANNED')
                if e['backend']!='a3':self.assertNotIn('traversability',e['request'])
            self.assertEqual(len(paired),500)
            self.assertTrue(all(s==set(BACKENDS) for s in paired.values()))
            self.assertEqual({e['planner_seed'] for e in plan['entries']},set(range(96001,96011)))
            self.assertEqual(plan,plan_trials(c,'main',dataset,root))
            maps[0]['cloud_sha256']='changed'
            with self.assertRaises(ValueError):plan_trials(c,'main',dataset,root)

    def test_a3_safety_objective_enabled(self):
        r=request_for('a3',dict(start=[1,1],goal=[2,2]),'map','cloud','hash',11,15.,default_config())
        self.assertEqual(r['planner_id'],'bitstar_a3')
        self.assertEqual(r['objective_type'],'surface_length_traversability')
        self.assertEqual(r['traversability'],DEFAULT_TRAVERSABILITY)
        self.assertIn('flat_40m',SMOKE_FAMILIES)
        self.assertEqual(len(SMOKE_FAMILIES)*len(BACKENDS),28)

    def test_sgp_declared_cpu_floating_point_policy_is_frozen(self):
        config=default_config()
        self.assertIs(config['native_parameters']['sgp']['cpu_flush_denormal'],True)
        self.assertIn('FP32_and_FP64',config['native_parameters']['sgp']['subnormal_policy_scope'])
        with tempfile.TemporaryDirectory(prefix='synthetic_fp_policy_',dir=self.fixtures) as d:
            path=Path(d)/'synthetic_config.json'
            dump(path,config)
            self.assertEqual(load_config(path),config)
            config['native_parameters']['sgp']['cpu_flush_denormal']=False
            dump(path,config)
            with self.assertRaises(ValueError):load_config(path)


if __name__=='__main__':unittest.main()
