"""Schema-12 control plane contract fixtures, never benchmark observations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import native_paper_v3_campaign as campaign
from experiment_common import dump
from native_paper_v3_protocol import PROTOCOL_ID, SCHEMA_VERSION, sha256


class CampaignContract(unittest.TestCase):
    def setUp(self):
        parent=ROOT/"tests/fixtures/native_paper_v3"
        parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="synthetic_campaign_",dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_unprepared_dry_run_has_no_side_effects(self):
        before=set(self.root.iterdir())
        result=campaign.plan_summary()
        self.assertEqual(result["maps"],100)
        self.assertEqual(result["paired_blocks"],1000)
        self.assertEqual(result["slots"],3000)
        self.assertEqual(result["planning_budget_s"],20.)
        self.assertEqual(result["backends"],["a3","sem","putn"])
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["dataset_verified"])
        self.assertEqual(set(self.root.iterdir()),before)

    def test_gate_declares_all_tests_without_gp_dependency(self):
        listed=types.SimpleNamespace(stdout=json.dumps({"tests":[{"name":n} for n in campaign.REQUIRED_CTESTS]}))
        prepared=dict(build_dir="fixture_build",controller_python=sys.executable,sem_python=sys.executable)
        with patch.object(campaign.subprocess,"run",return_value=listed):
            commands=campaign.correctness_commands(prepared,Path("fixture_worker"),Path("fixture_library"))
        text=json.dumps(commands)
        self.assertNotIn("sgp",text.lower())
        self.assertNotIn("gpytorch",text.lower())
        self.assertIn("test_native_paper_v3_fixed_work.py",text)
        self.assertIn("test_shared_sem_evaluator.py",text)
        self.assertIn("test_native_paper_worker.py",text)
        bad=types.SimpleNamespace(stdout=json.dumps({"tests":[]}))
        with patch.object(campaign.subprocess,"run",return_value=bad):
            with self.assertRaisesRegex(ValueError,"missing required"):
                campaign.correctness_commands(prepared,Path("worker"),Path("library"))

    def test_main_still_requires_explicit_linux_opt_in(self):
        dump(self.root/"correctness.json",dict(status="PASS",identity_sha256=campaign.digest({})))
        with patch.object(campaign,"check_frozen",return_value={"identity":{}}),patch.object(campaign,"run_campaign") as run:
            for allow,force in ((False,False),(True,True)):
                with self.assertRaisesRegex(ValueError,"main requires"):
                    campaign.collect(self.root,"main",allow_main=allow,force_local=force)
            with patch.object(campaign.sys,"platform","win32"):
                with self.assertRaisesRegex(ValueError,"main requires"):
                    campaign.collect(self.root,"main",allow_main=True)
            run.assert_not_called()

    def test_frozen_guard_rejects_registry_and_input_changes(self):
        binary=self.root/"fixture_binary";binary.write_bytes(b"synthetic binary hash fixture")
        dump(self.root/"registry.json",{"synthetic_fixture":True})
        dump(self.root/"plans/smoke.json",{"synthetic_fixture":True})
        identity=dict(source_sha256="fixture_source",binary_hashes={str(binary):sha256(binary)},
            native_roots=[],native_linkage=[],environments={"controller":{},"sem":{}})
        prepared=dict(schema_version=SCHEMA_VERSION,protocol_id=PROTOCOL_ID,identity=identity,
            controller_python=sys.executable,sem_python=sys.executable,
            registry_sha256=sha256(self.root/"registry.json"),
            frozen_inputs={"plans/smoke.json":sha256(self.root/"plans/smoke.json")})
        dump(self.root/"prepared.json",prepared)
        with patch.object(campaign,"source_identity",return_value={"source_sha256":"fixture_source"}), \
             patch.object(campaign,"capture_native_dependencies",return_value={"linkage":[]}), \
             patch.object(campaign,"environment_identity",return_value={}):
            self.assertEqual(campaign.check_frozen(self.root),prepared)
            dump(self.root/"registry.json",{"tampered_fixture":True})
            with self.assertRaisesRegex(ValueError,"registry changed"):
                campaign.check_frozen(self.root)
            prepared["registry_sha256"]=sha256(self.root/"registry.json");dump(self.root/"prepared.json",prepared)
            dump(self.root/"plans/smoke.json",{"tampered_fixture":True})
            with self.assertRaisesRegex(ValueError,"plan/map/metadata"):
                campaign.check_frozen(self.root)
            binary.write_bytes(b"tampered fixture binary")
            with self.assertRaisesRegex(ValueError,"binary/interpreter"):
                campaign.check_frozen(self.root)

    def test_export_cli_requires_and_binds_source_archive(self):
        from kaggle_a3_bundle import archive
        from kaggle_native_paper_v3_bundle import KIND
        source=self.root/"source.zip"
        archive([("configs/native_paper_kaggle_v3.yaml",ROOT/"configs/native_paper_kaggle_v3.yaml")],
                source,KIND,{"protocol_id":PROTOCOL_ID,"config_schema":SCHEMA_VERSION})
        arguments=["campaign","export","--workspace",str(self.root),"--output",str(self.root/"checkpoint.zip")]
        with patch.object(sys,"argv",arguments),patch.object(campaign,"check_frozen") as frozen:
            with self.assertRaises(SystemExit) as error:
                campaign.main()
            self.assertEqual(error.exception.code,2)
            frozen.assert_not_called()
        with patch.object(sys,"argv",[*arguments,"--input-archive",str(source)]), \
             patch.object(campaign,"check_frozen"), \
             patch("kaggle_native_paper_v3_bundle.export_checkpoint",return_value={"status":"EXPORTED"}) as export:
            campaign.main()
        self.assertEqual(export.call_args.kwargs["input_archive"],source)


if __name__=="__main__":unittest.main()
