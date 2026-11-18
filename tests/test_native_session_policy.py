"""Registered native-session resume and geometry-only safety regression tests."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import terrain_compare_scheduler as scheduler
from independent_verifier import Oracle
from terrain_io import write_map


class NativeSessionTests(unittest.TestCase):
    def test_hardware_change_only_for_registered_protocol(self):
        old={"platform":"linux", "machine":"x86_64", "python":"3.12.7",
             "cpu_model":"CPU A", "eligible_cpus":[0,1,2,3], "memory_limit_bytes":8*1024**3}
        new={**old,"cpu_model":"CPU B", "eligible_cpus":[4,5,6,7],"memory_limit_bytes":16*1024**3}
        plan={"protocol_id":"a3t_native_paper_kaggle_v2", "runtime_policy":"native_paper_sessions_v2"}
        self.assertEqual(scheduler.campaign_runtime_identity(plan,old),scheduler.campaign_runtime_identity(plan,new))
        self.assertNotEqual(scheduler.campaign_runtime_identity({},old),scheduler.campaign_runtime_identity({},new))
        self.assertNotEqual(scheduler.campaign_runtime_identity(plan,old),
                            scheduler.campaign_runtime_identity(plan,{**new,"python":"3.13.0"}))
        self.assertNotEqual(scheduler.campaign_runtime_identity({"runtime_policy":"native_paper_sessions_v2"},old),
                            scheduler.campaign_runtime_identity({"runtime_policy":"native_paper_sessions_v2"},new))

    def test_resume_new_hardware_retains_attempt_and_records_stratum(self):
        # Synthetic worker is used only inside the explicitly labeled fixtures directory.
        with tempfile.TemporaryDirectory(prefix="native-session-fixture-",dir=ROOT/"tests/fixtures") as tmp:
            root=Path(tmp)
            cpus=scheduler.psutil.Process().cpu_affinity()[:1]
            runtime={"platform":sys.platform,"machine":"test-machine","python":"3.12.7",
                     "cpu_model":"first", "eligible_cpus":cpus,"cpu_capacity":1,
                     "memory_limit_bytes":16*scheduler.GIB,"memory_sources":[]}
            registry={"fake":{"argv":[sys.executable,str(ROOT/"tests/fixtures/terrain_compare_fake_worker.py")]}}
            plan={"schema_version":11,"protocol_id":"a3t_native_paper_kaggle_v2", "runtime_policy":"native_paper_sessions_v2",
                  "phase_timeouts":{"setup_s":600.,"serialize_s":60.,"solve_grace_s":5.},
                  "profile":"smoke", "synthetic_test_fixture":True,"entries":[
                      {"trial_id":"synthetic_one","backend":"fake","block_order":0,"planned_order":0,
                       "execution_status":"PLANNED","request":{"planning_budget_s":.1,"seed":11}}]}
            scheduler.atomic_json(root/"plan.json",plan)
            def execute(resume=False):
                with patch.object(scheduler,"detect_runtime",return_value=runtime), \
                     patch.object(scheduler,"memory_pressure",return_value=.1):
                    return scheduler.run_campaign(root/"plan.json",registry,root/"run",max_workers=1,
                        force_local=True,reservations={"fake":512*scheduler.MIB},resume=resume)
            first=execute()
            journal=(root/"run/start_journal.jsonl").read_bytes()
            runtime["cpu_model"]="second"
            second=execute(True)
            self.assertEqual(second["counts"],{"COMPLETED":1})
            self.assertEqual(journal,(root/"run/start_journal.jsonl").read_bytes())
            self.assertNotEqual(second["sessions"][0]["timing_stratum"],second["sessions"][1]["timing_stratum"])
            self.assertEqual(first["entries"][0]["runtime_sha256"],second["entries"][0]["runtime_sha256"])
            with patch.object(scheduler,"phase_timeout_policy",return_value={"setup_watchdog_s":180.,"serialization_watchdog_s":60.,"solve_grace_s":5.}):
                with self.assertRaisesRegex(scheduler.IntegrityError,"provenance/policy/runtime"):
                    execute(True)

    def test_registered_phase_limits_and_malformed_native_metadata(self):
        native={"schema_version":11,"protocol_id":"a3t_native_paper_kaggle_v2","runtime_policy":"native_paper_sessions_v2",
                "phase_timeouts":{"setup_s":600.,"serialize_s":60.,"solve_grace_s":5.}}
        policy=scheduler.phase_timeout_policy(native)
        self.assertEqual(policy,{"setup_watchdog_s":600.,"serialization_watchdog_s":60.,"solve_grace_s":5.})
        self.assertEqual(scheduler.phase_timeout_policy({"phase_timeouts":native["phase_timeouts"]})["setup_watchdog_s"],180.)
        for change in ({"phase_timeouts":None},{"schema_version":10},{"runtime_policy":"unregistered"},
                       {"phase_timeouts":{"setup_s":600.,"serialize_s":60.}},
                       {"phase_timeouts":{**native["phase_timeouts"],"unregistered":1.}},
                       *[{"phase_timeouts":{**native["phase_timeouts"],"setup_s":bad}} for bad in (180.,False,float('nan'),float('inf'),'600')]):
            with self.assertRaises(scheduler.IntegrityError):scheduler.phase_timeout_policy({**native,**change})

    def test_effective_setup_age_501_survives_only_native_and_601_fails(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-native-watchdog-",dir=ROOT/"tests/fixtures") as tmp:
            root=Path(tmp);cpus=scheduler.psutil.Process().cpu_affinity()[:1]
            runtime={"platform":sys.platform,"machine":"fixture","python":"3.12.7","cpu_model":"fixture",
                     "eligible_cpus":cpus,"cpu_capacity":1,"memory_limit_bytes":16*scheduler.GIB,"memory_sources":[]}
            registry={"fake":{"argv":[sys.executable,str(ROOT/"tests/fixtures/terrain_compare_fake_worker.py")]}}
            update=scheduler._phase_update
            for name,native,age,expected in (("native_501",True,501.,"COMPLETED"),("legacy_501",False,501.,"WATCHDOG"),("native_601",True,601.,"WATCHDOG")):
                plan={"profile":"smoke","synthetic_test_fixture":True,"entries":[dict(trial_id="synthetic",backend="fake",block_order=0,
                      planned_order=0,execution_status="PLANNED",request=dict(planning_budget_s=.1,setup_s=.1))]}
                if native:plan.update(schema_version=11,protocol_id="a3t_native_paper_kaggle_v2",runtime_policy="native_paper_sessions_v2",
                                     phase_timeouts={"setup_s":600.,"serialize_s":60.,"solve_grace_s":5.})
                scheduler.atomic_json(root/(name+".json"),plan)
                def aged(active,now):
                    update(active,now)
                    if active["phase"]=="SETUP":active["phase_began"]=now-age
                with patch.object(scheduler,"detect_runtime",return_value=runtime),patch.object(scheduler,"memory_pressure",return_value=.1),patch.object(scheduler,"_phase_update",side_effect=aged):
                    result=scheduler.run_campaign(root/(name+".json"),registry,root/name,max_workers=1,force_local=True,reservations={"fake":512*scheduler.MIB})
                self.assertEqual(result["entries"][0]["execution_status"],expected)
                self.assertEqual(result["scheduler_policy"]["setup_watchdog_s"],600. if native else 180.)

    def test_same_setup_phase_timestamp_does_not_refresh_guard(self):
        active={"directory":Path('unused'),"phase":"SETUP","phase_began":5.,"began":5.,"phase_history":[]}
        with patch.object(scheduler,"_read_phase",return_value={"phase":"SETUP","updated_utc":"2099-01-01T00:00:00+00:00"}):
            scheduler._phase_update(active,506.)
        self.assertEqual(active["phase_began"],5.)

    def test_geometry_safety_without_risk_field(self):
        with tempfile.TemporaryDirectory(prefix="native-oracle-fixture-",dir=ROOT/"tests/fixtures") as tmp:
            path=Path(tmp)/"plane.t25d"
            spacing=.2
            write_map(path,[[.1*i*spacing for i in range(21)] for j in range(21)],spacing=(spacing,spacing))
            oracle=Oracle(path)
            with patch.object(oracle,"traversability_edge",side_effect=AssertionError("unexpected risk evaluator")):
                result=oracle.validate_path([[1.,2.],[3.,2.]],[1.,2.],[3.,2.],geometric_safety_metrics=True)
            self.assertTrue(result["valid"],result)
            self.assertAlmostEqual(result["maximum_swept_footprint_slope_deg"],math.degrees(math.atan(.1)),places=8)
            self.assertAlmostEqual(result["minimum_footprint_clearance_m"],.6,places=8)
            self.assertNotIn("risk_integral_m",result)
            self.assertNotIn("objective_value",result)


if __name__=="__main__":unittest.main()
