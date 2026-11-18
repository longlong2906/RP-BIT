"""Gate/CLI/notebook contract tests; real planner smoke is a separate gate."""
import argparse
import json
from pathlib import Path
import sys
import time
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import native_paper_campaign as campaign
import kaggle_native_paper as kaggle
import kaggle_native_paper_bundle as bundle
from experiment_common import digest
from native_paper_analysis import validate_campaign
from terrain_compare_scheduler import _lock,IntegrityError


class NativePipeline(unittest.TestCase):
    def setUp(self):
        self.fixtures=ROOT/'tests/fixtures/native_paper'
        self.fixtures.mkdir(parents=True,exist_ok=True)

    def test_explicit_main_gate(self):
        identity={'synthetic_fixture':True}
        with patch.object(campaign,'check_frozen',return_value={'identity':identity}),patch.object(campaign,'read_json',return_value={'status':'PASS','identity_sha256':digest(identity)}):
            with self.assertRaisesRegex(ValueError,'allow-main'):campaign.collect(ROOT,'main')
            with self.assertRaisesRegex(ValueError,'allow-main'):campaign.collect(ROOT,'main',allow_main=True,force_local=True)

    def test_commands_split_collection_and_validation(self):
        rt={'controller':Path('controller'),'sgp':Path('sgp')}
        cmd=kaggle.campaign_command('collect',Path('workspace'),rt,mode='main',resume=True,allow_main=True,started=time.monotonic())
        self.assertIn('--allow-main',cmd);self.assertIn('--resume',cmd);self.assertNotIn('validate',cmd)
        self.assertIn('native_paper_campaign.py',str(cmd[1]))
        validation=kaggle.campaign_command('validate',Path('workspace'),rt,mode='main')
        self.assertIn('--jobs',validation);self.assertNotIn('--allow-main',validation)

    def test_notebook_safe_defaults_and_runnable_python(self):
        nb=json.loads((ROOT/'notebooks/native_paper_kaggle_v2.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        compile(code,'native-paper-notebook','exec')
        self.assertIn('ALLOW_MAIN = False',code);self.assertIn("STAGES = ['prepare', 'smoke']",code)
        self.assertIn(bundle.KIND,code)
        self.assertNotIn('matplotlib',code)
        self.assertEqual(bundle.KIND,'TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA11_SOURCE')

    def test_validation_cannot_overlap_even_between_blocks(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_campaign_lock_',dir=self.fixtures) as d:
            root=Path(d);lock=_lock(root)
            try:
                with self.assertRaises(IntegrityError):validate_campaign(root,1)
            finally:lock.unlink()


if __name__=='__main__':unittest.main()
