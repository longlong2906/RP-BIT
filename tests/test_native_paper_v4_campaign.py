"""Synthetic controller unit tests; these are not benchmark trials."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from native_paper_v4_campaign import plan_summary,wave_groups
from terrain_compare_scheduler import campaign_runtime_identity,phase_timeout_policy

class V4CampaignTests(unittest.TestCase):
    def test_cardinality(self):
        self.assertEqual({k:plan_summary()[k] for k in ('maps','paired_blocks','slots','planning_budget_s')},
                         dict(maps=50,paired_blocks=500,slots=1500,planning_budget_s=20.))
    def test_waves_cover_every_map_once(self):
        p={'entries':[dict(map_id='synthetic_'+str(i)) for i in range(50)]}
        waves=wave_groups(p)
        self.assertEqual([len(w) for w in waves],[4]*12+[2])
        self.assertEqual(len({m for w in waves for m in w}),50)
        self.assertEqual(waves,wave_groups(p))
    def test_registered_session_policy(self):
        plan=dict(protocol_id='a3t_old_safe_sem_kaggle_v4',schema_version=13,runtime_policy='native_paper_sessions_v4',
                  phase_timeouts=dict(setup_s=600.,serialize_s=60.,solve_grace_s=5.))
        a=dict(platform='linux',machine='x86_64',python='3.12.7',cpu_model='a')
        b=dict(a,cpu_model='b')
        self.assertEqual(campaign_runtime_identity(plan,a),campaign_runtime_identity(plan,b))
        self.assertEqual(phase_timeout_policy(plan)['setup_watchdog_s'],600.)
        plan['schema_version']=12
        with self.assertRaises(ValueError):phase_timeout_policy(plan)

if __name__=='__main__':unittest.main()
