"""Labeled synthetic runner tests; artifacts stay under tests/fixtures.

Optional --worker adds a real 13-planner worker/independent-validation probe.
It uses a tiny analytic map and fixed work, never the 780-trial pilot.
"""
import argparse
import copy
import contextlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import a3_pilot_campaign as campaign
from a3_pilot_protocol import PLANNERS, expected_config
from experiment_common import ROOT, digest, dump, file_hash, resolved_planner_options
from generate_scenarios import write_map

WORKER = None


def fixture_entry(label="bitstar_a3", index=0):
    entry = dict(trial_id="synthetic_"+label, map_id="synthetic_map", map_path="synthetic_map.t25d",
        map_hash="synthetic-map", planner_id=label, planner_seed=11, start=[1., 1.], goal=[4., 4.],
        block_order=0, shuffle_order=index)
    if label in expected_config()["planner_variants"]:
        options = resolved_planner_options(expected_config(), label)
        entry.update(resolved_optimizations=options, resolved_optimizations_sha256=digest(options))
    return entry


def fixture_record(label="bitstar_a3"):
    config = expected_config()
    options = resolved_planner_options(config, label)
    return dict(planner_id=label, backend_planner_id=options["planner_id"], variant_id=label,
        execution_status="TIMEOUT", planner_status="NO_SOLUTION", path_available=False,
        effective_policy={**{k: options[k] for k in ("state_cache", "scalar_cost", "collision_index", "samples_per_batch", "batch_after_first_solution")},
            "batch_policy": "after_first_solution" if options["batch_after_first_solution"] else "fixed"})


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = expected_config()
        self.entry = fixture_entry()
        self.identity = dict(config_hash=digest(self.config), code_sha256="synthetic-source", worker_sha256="synthetic-worker", ompl_library_sha256="synthetic-library")
        self.manifest = dict(profile="smoke", experiment_id="synthetic_test_fixture", planning_budget_s=2., identity=self.identity, trials=[self.entry])
        self.session = dict(session_id="synthetic-session", stratum_id="synthetic-cpu-stratum")
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-a3-pilot-campaign-", dir=ROOT / "tests/fixtures")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_pilot_needs_explicit_flag_and_linux_no_accidental_execution(self):
        with self.assertRaisesRegex(campaign.IntegrityError, "explicit"):
            campaign.run_campaign("unused", "unused", self.root, self.root, "unused", profile="a3_pilot")
        with patch.object(campaign.sys, "platform", "win32"), self.assertRaisesRegex(campaign.IntegrityError, "Linux"):
            campaign.run_campaign("unused", "unused", self.root, self.root, "unused", profile="a3_pilot", allow_a3_pilot=True)
        with self.assertRaisesRegex(campaign.IntegrityError, "full prohibited"):
            campaign.run_campaign("unused", "unused", self.root, self.root, "unused", profile="full")

    def test_all_thirteen_effective_policies_and_baseline_mismatch(self):
        for label in PLANNERS:
            with self.subTest(label=label):
                campaign.verify_policy(fixture_entry(label), self.config, fixture_record(label))
                wrong = fixture_record(label)
                wrong["effective_policy"]["state_cache"] = not wrong["effective_policy"]["state_cache"]
                with self.assertRaises(ValueError):
                    campaign.verify_policy(fixture_entry(label), self.config, wrong)
        wrong = fixture_record()
        wrong["effective_policy"]["samples_per_batch"] = 50
        with self.assertRaisesRegex(campaign.IntegrityError, "200"):
            campaign.verify_policy(self.entry, self.config, wrong)

    def test_build_gate_bound_to_source_worker_and_library(self):
        path = self.root / "gate.json"
        gate = dict(status="PASS", ctest_passed=True, **self.identity)
        dump(path, gate)
        campaign.check_build_gate(path, self.identity)
        for key in campaign.IDENTITY_GATE_KEYS:
            wrong = {**gate, key: "changed"}
            dump(path, wrong)
            with self.assertRaisesRegex(campaign.IntegrityError, "stale"):
                campaign.check_build_gate(path, self.identity)

    def test_runtime_identity_ignores_mount_cpu_but_not_versions(self):
        environment = {"os": "Linux", "cpu": "cpu-a", "python": "3.12.7", "python_executable": "/mount/a/python",
            "packages": {"numpy": "1"}, "compiler": "GCC", "build_flags": {"type": "Release"},
            "dependency_version_evidence": {"ompl": {"version": "1.7.0", "sha256": "header", "header": "/mount/a/header"}}}
        other = copy.deepcopy(environment)
        other.update(cpu="cpu-b", python_executable="/mount/b/python")
        other["dependency_version_evidence"]["ompl"]["header"] = "/mount/b/header"
        self.assertEqual(campaign.runtime_identity(environment), campaign.runtime_identity(other))
        other["packages"]["numpy"] = "2"
        self.assertNotEqual(campaign.runtime_identity(environment), campaign.runtime_identity(other))

    def test_phase_limits_exclude_setup_and_serialization_from_solve(self):
        self.assertEqual(campaign.phase_limit("SETUP", 10., self.config), 60.)
        self.assertEqual(campaign.phase_limit("SOLVING", 10., self.config), 15.)
        self.assertEqual(campaign.phase_limit("SERIALIZING", 10., self.config), 60.)

    def test_phase_control_file_atomic_replace_race_keeps_previous_deadline(self):
        path = self.root / "phase.json"
        dump(path, {"phase": "SOLVING"})
        for error in (FileNotFoundError("replace gap"), PermissionError("sharing lock")):
            with patch.object(campaign, "read_json", side_effect=error):
                self.assertIsNone(campaign.phase_read(path))
        dump(path, {"phase": "WRONG"})
        with self.assertRaises(campaign.IntegrityError):
            campaign.phase_read(path)

    def test_mock_process_setup_solve_and_serialization_watchdogs(self):
        for phase, ticks, reason in [(None, [0., 61., 62.], "SETUP_WATCHDOG"),
                ({"phase": "SOLVING"}, [0., 0., 8., 9.], "SOLVING_WATCHDOG"),
                ({"phase": "SERIALIZING"}, [0., 0., 61., 62.], "SERIALIZING_WATCHDOG")]:
            with self.subTest(reason=reason):
                proc = Mock()
                proc.pid, proc.poll.return_value, proc.wait.return_value = 123, None, -9
                with patch.object(campaign.subprocess, "Popen", return_value=proc), patch.object(campaign.psutil, "Process", side_effect=campaign.psutil.NoSuchProcess(123)), patch.object(campaign, "phase_read", return_value=phase), patch.object(campaign.time, "perf_counter", side_effect=ticks), patch.object(campaign.time, "sleep"):
                    trial = campaign.launch_trial(self.entry, self.config, self.root / reason, self.root / "worker", {}, self.manifest, self.root, self.session)
                proc.kill.assert_called()
                self.assertEqual(trial["execution_status"], "WATCHDOG")
                self.assertEqual(trial["termination_reason"], reason)
                self.assertFalse(trial["success"])

    def test_interrupted_attempt_retained_and_never_rewritten(self):
        directory = self.root / self.entry["trial_id"]
        directory.mkdir()
        dump(directory / "attempt.json", {"runtime_session_id": "original-session", "synthetic_test_fixture": True})
        self.assertTrue(campaign.retain_interrupted(directory, self.entry, self.manifest))
        initial = file_hash(directory / "trial.json")
        self.assertFalse(campaign.retain_interrupted(directory, self.entry, self.manifest))
        self.assertEqual(file_hash(directory / "trial.json"), initial)
        trial = campaign.read_json(directory / "trial.json")
        self.assertEqual(trial["execution_status"], "CRASH")
        self.assertEqual(trial["runtime_session_id"], "original-session")

    def test_interrupted_serialized_final_path_still_gets_validation(self):
        directory = self.root / self.entry["trial_id"]
        directory.mkdir()
        dump(directory / "attempt.json", {"runtime_session_id": "original-session"})
        record = fixture_record()
        record.update(path_available=True, planner_status="EXACT_SOLUTION", best_cost_terrain_m=5.)
        dump(directory / "worker.json", record)
        campaign.retain_interrupted(directory, self.entry, self.manifest, self.config)
        recovered = campaign.read_json(directory / "trial.json")
        self.assertTrue(recovered["path_available"])
        self.assertEqual(recovered["planner_status"], "EXACT_SOLUTION")
        self.assertEqual(recovered["execution_status"], "CRASH")
        self.assertFalse(recovered["success"])

    def test_frozen_trial_detects_changes_and_additional_files(self):
        directory = self.root / self.entry["trial_id"]
        directory.mkdir()
        dump(directory / "trial.json", {"synthetic_test_fixture": True})
        campaign.freeze_trial(directory)
        campaign.verify_completed(self.root, [self.entry])
        (directory / "added.txt").write_text("synthetic tamper", encoding="utf-8")
        with self.assertRaises(campaign.IntegrityError):
            campaign.verify_completed(self.root, [self.entry])

    def test_timeout_is_executed_failure_without_zero_cost_or_first_time(self):
        directory = self.root / self.entry["trial_id"]
        directory.mkdir()
        trial = {**fixture_record(), "best_cost_terrain_m": None, "first_solution_time_s": None,
                 "validation_wall_s": .01, "success": False, "synthetic_test_fixture": True}
        dump(directory / "trial.json", trial)
        dump(directory / "validation.json", {"valid": False, "errors": ["no_path"], "synthetic_test_fixture": True})
        counts = campaign.summarize(self.root, [self.entry])
        self.assertEqual(counts["executed"], 1)
        self.assertEqual(counts["timeout"], 1)
        self.assertEqual(counts["fatal"], 0)
        self.assertEqual(counts["validated_success"], 0)
        self.assertEqual(counts["invalid"], 0)
        self.assertIsNone(campaign.read_json(directory / "trial.json")["first_solution_time_s"])
        self.assertIsNone(campaign.read_json(directory / "trial.json")["best_cost_terrain_m"])
        self.assertEqual(campaign.checkpoint(self.root, self.manifest, "VALIDATED")["status"], "COMPLETE_VALIDATED")

    def test_smoke_gate_cannot_be_bare_forged_or_different_runtime(self):
        gate = dict(protocol_id=campaign.PROTOCOL_ID, status="PASS", planned_trials=52, validated_trials=52,
                    identity=self.identity, runtime_identity_sha256=digest({"python": "3.12"}), manifest_sha256="missing")
        dump(self.root / "smoke_gate.json", gate)
        with self.assertRaisesRegex(campaign.IntegrityError, "runtime/config"):
            campaign.check_smoke_gate(self.root / "smoke_gate.json", self.identity, {"python": "3.13"})
        with self.assertRaises(OSError):
            campaign.check_smoke_gate(self.root / "smoke_gate.json", self.identity, {"python": "3.12"})

    def test_disk_guard_checks_free_space_and_output_size(self):
        (self.root / "synthetic.bin").write_bytes(b"fixture")
        with patch.object(campaign.shutil, "disk_usage", return_value=Mock(free=100)):
            self.assertTrue(campaign.disk_ready(self.root, 50, 100))
            self.assertFalse(campaign.disk_ready(self.root, 101, 100))
            self.assertFalse(campaign.disk_ready(self.root, 50, 6))

    @contextlib.contextmanager
    def fixture_campaign_context(self):
        """A labeled one-entry orchestration fixture, never an experiment."""
        import a3_pilot_protocol
        manifest = self.root / "synthetic_maps.json"
        dump(manifest, {"maps": [], "synthetic_test_fixture": True})
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(campaign, "load_config", return_value=self.config))
            stack.enter_context(patch.object(campaign, "validate_dataset"))
            stack.enter_context(patch.object(a3_pilot_protocol, "plan_trials", return_value=[self.entry]))
            stack.enter_context(patch.object(campaign.common, "cloud_identity", return_value=dict(self.identity)))
            stack.enter_context(patch.object(campaign.common, "cloud_environment", return_value={"os": "Linux", "cpu": "synthetic_cpu"}))
            stack.enter_context(patch.object(campaign, "check_build_gate"))
            stack.enter_context(patch.object(campaign, "verify_live_code"))
            yield dict(config_path="synthetic", map_manifest=manifest, dataset_root=self.root,
                output=self.root / "synthetic_campaign", worker=self.root / "synthetic_worker")

    def test_resume_does_not_relaunch_an_interrupted_attempt(self):
        with self.fixture_campaign_context() as args:
            campaign.run_campaign(**args, dry_run=True)
            directory = args["output"] / self.entry["trial_id"]
            directory.mkdir()
            dump(directory / "attempt.json", {"runtime_session_id": "synthetic-session", "synthetic_test_fixture": True})
            def validate_interrupted(output, entries, dataset):
                trial = campaign.read_json(directory / "trial.json")
                trial["validation_status"] = "NO_REQUEST_INTERRUPTED_OR_BLOCKED"
                dump(directory / "trial.json", trial)
                campaign.freeze_trial(directory)
            with patch.object(campaign, "launch_trial") as launch, patch.object(campaign, "validate_block", side_effect=validate_interrupted):
                result = campaign.run_campaign(**args, resume=True)
            launch.assert_not_called()
            self.assertEqual(result["counts"]["attempted"], 1)
            self.assertEqual(result["counts"]["fatal"], 1)

    def test_resume_rejects_changed_source_identity(self):
        with self.fixture_campaign_context() as args:
            campaign.run_campaign(**args, dry_run=True)
            with patch.object(campaign.common, "cloud_identity", return_value={**self.identity, "code_sha256": "changed"}), self.assertRaisesRegex(campaign.IntegrityError, "resume"):
                campaign.run_campaign(**args, resume=True)

    def test_soft_deadline_does_not_start_new_block(self):
        with self.fixture_campaign_context() as args, patch.object(campaign, "launch_trial") as launch:
            result = campaign.run_campaign(**args, session_elapsed_s=28800)
            self.assertEqual(result["phase"], "SOFT_DEADLINE")
            self.assertEqual(result["counts"]["executed"], 0)
            launch.assert_not_called()

    def test_actual_thirteen_worker_variants_and_independent_paths(self):
        if WORKER is None:
            self.skipTest("--worker absent; real integration probe not requested")
        path = self.root / "synthetic_map.t25d"
        meta = write_map(path, [[.1*x+.05*y for x in range(11)] for y in range(11)], spacing=(.5,.5),
            radius=.3, margin=.1, slope=35., metadata={"map_id": "synthetic_map", "synthetic_test_fixture": True})
        entries = []
        for index, label in enumerate(PLANNERS):
            entry = fixture_entry(label, index)
            entry.update(map_hash=file_hash(path), work_iterations=200)
            entries.append(entry)
            trial = campaign.launch_trial(entry, self.config, self.root / entry["trial_id"], WORKER, {}, self.manifest, self.root, self.session)
            self.assertEqual(trial["execution_status"], "COMPLETED", (label, trial))
            self.assertTrue(trial["policy_verified"])
            self.assertEqual(trial["runtime_session_id"], self.session["session_id"])
        campaign.validate_block(self.root, entries, self.root)
        campaign.verify_completed(self.root, entries)
        counts = campaign.summarize(self.root, entries)
        self.assertEqual(counts["offline_validated"], 13)
        self.assertEqual(counts["invalid"], 0)
        self.assertEqual(counts["fatal"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    arguments, remaining = parser.parse_known_args()
    WORKER = arguments.worker.resolve() if arguments.worker else None
    unittest.main(argv=[sys.argv[0], *remaining])
