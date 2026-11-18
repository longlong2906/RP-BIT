"""Fault-injected controller and smoke-readiness fixtures; no planner runs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import native_paper_v4_campaign as campaign
import native_paper_v4_storage as storage
import terrain_compare_scheduler as scheduler
from experiment_common import dump, digest
from native_paper_v4_analysis import dump as verdict_dump, validation_code_identity


class WaveResumeTests(unittest.TestCase):
    def fixture(self, workspace, map_count=2):
        prepared = dict(identity={"source_sha256":"synthetic_source"})
        dump(workspace/"prepared.json", prepared)
        dump(workspace/"correctness.json", dict(status="PASS", identity_sha256=digest(prepared["identity"])))
        entries, maps = [], []
        for k in range(map_count):
            mid = "map_"+str(k)
            maps.append(dict(map_id=mid, family="flat", map_path="synthetic", map_sha256="a"*64,
                             cloud_path="synthetic", cloud_sha256="b"*64, bounds=[0,0,20,20], start=[1,1], goal=[19,19]))
            for backend in campaign.BACKENDS:
                tid = mid+"__"+backend
                entries.append(dict(trial_id=tid, map_id=mid, family="flat", planner_seed=11,
                                    backend=backend, request=dict(backend=backend, planner_seed=11),
                                    execution_status="PLANNED"))
        plan = dict(entries=entries, planned_slots=len(entries), planned_executable=len(entries))
        dump(workspace/"plans/smoke.json", plan)
        dump(workspace/"maps.json", dict(smoke=maps))
        dump(workspace/"registry.json", {})
        return prepared

    def fake_collect(self, calls):
        def collect(plan_path, registry, output, **kwargs):
            calls.append(dict(plan=str(plan_path), resume=kwargs["resume"]))
            plan = json.loads(Path(plan_path).read_text())
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            for entry in plan["entries"]:
                directory = output/entry["trial_id"]
                directory.mkdir()
                dump(directory/"request.json", entry["request"])
                dump(directory/"execution.json", dict(execution_status="COMPLETED"))
                dump(directory/"worker.json", dict(synthetic=True))
                entry["execution_status"] = "COMPLETED"
            result = dict(plan, status="COLLECTED_VALIDATION_PENDING", concurrency_timeline=[dict(concurrency=1)])
            dump(output/"manifest.json", result)
            return result
        return collect

    def fake_validate(self, calls):
        def validate(args):
            workspace, info, entries, _ = args
            calls.append(info["map_id"])
            rows = []
            for entry in entries:
                row = dict(trial_id=entry["trial_id"], backend=entry["backend"],
                           errors=[], correctness_errors=[], validation_code_identity=validation_code_identity())
                directory = Path(entry["directory"])
                verdict_dump(directory/"validation.json", row)
                verdict_hash = storage.file_hash(directory/"validation.json")
                receipt = dict(verdict_sha256=verdict_hash)
                dump(directory/"validation_receipt.json", receipt)
                dump(Path(workspace)/"verdicts"/info["mode"]/(entry["trial_id"]+".json"),
                     dict(row=row, receipt=receipt, verdict_sha256=verdict_hash))
                rows.append(row)
            return rows
        return validate

    def run_patches(self, prepared, collect, validate, compact):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(campaign, "check_frozen", return_value=prepared))
        stack.enter_context(patch.object(campaign, "detect_runtime", return_value=dict(cpu_capacity=4, memory_limit_bytes=64*storage.GIB,
                                                                                   memory_sources=[])))
        stack.enter_context(patch("psutil.virtual_memory", return_value=SimpleNamespace(total=64*storage.GIB, available=48*storage.GIB)))
        stack.enter_context(patch.object(campaign, "run_campaign", side_effect=collect))
        stack.enter_context(patch.object(campaign, "_validate_map_task", side_effect=validate))
        stack.enter_context(patch.object(campaign, "smoke_evidence", return_value=dict(synthetic_fault_fixture=True)))
        stack.enter_context(patch("native_paper_v4_storage.preflight", return_value=dict(allowed=True, reason="SYNTHETIC")))
        stack.enter_context(patch("native_paper_v4_worker.compact_trial", side_effect=compact))
        return stack

    def test_drained_move_and_compaction_interruption_does_not_recollect(self):
        with tempfile.TemporaryDirectory(prefix="v4_resume_synthetic_") as tmp:
            workspace = Path(tmp); prepared = self.fixture(workspace)
            collects, validations, compactions = [], [], []
            failed = False
            def compact(directory, request, shared):
                nonlocal failed
                compactions.append(Path(directory).name)
                if len(compactions) == 2 and not failed:
                    failed = True
                    raise RuntimeError("synthetic interruption after one moved/compacted trial")
                dump(Path(directory)/"compaction.json", dict(synthetic=True))
            with self.run_patches(prepared, self.fake_collect(collects), self.fake_validate(validations), compact):
                with self.assertRaises(RuntimeError):
                    campaign.run(workspace, "smoke-serial", force_local=True)
                manifest = json.loads((workspace/"manifests/smoke-serial.json").read_text())
                self.assertEqual(manifest["waves"][0]["status"], "DRAINED")
                self.assertEqual(campaign.run(workspace, "smoke-serial", force_local=True, resume=True)["status"], "COMPLETE")
            self.assertEqual(len(collects), 1)
            self.assertEqual(len(validations), 2)
            self.assertEqual(len(json.loads((workspace/"shards/index.json").read_text())["entries"]), 2)

    def test_validated_partial_pack_resumes_without_collection_or_validation(self):
        with tempfile.TemporaryDirectory(prefix="v4_packed_synthetic_") as tmp:
            workspace = Path(tmp); prepared = self.fixture(workspace)
            collects, validations = [], []
            original = storage.ArtifactStore.pack_map
            packed, fail_once = [], True
            def compact(directory, request, shared):
                dump(Path(directory)/"compaction.json", dict(synthetic=True))
            def interrupted_pack(store, wave, mid, cleanup=True):
                nonlocal fail_once
                if packed and mid not in packed and fail_once:
                    fail_once = False
                    raise RuntimeError("synthetic interruption after first map shard cleanup")
                result = original(store, wave, mid, cleanup)
                if mid not in packed: packed.append(mid)
                return result
            with self.run_patches(prepared, self.fake_collect(collects), self.fake_validate(validations), compact):
                with patch.object(storage.ArtifactStore, "pack_map", interrupted_pack):
                    with self.assertRaises(RuntimeError):
                        campaign.run(workspace, "smoke-serial", force_local=True)
                manifest = json.loads((workspace/"manifests/smoke-serial.json").read_text())
                self.assertEqual(manifest["waves"][0]["status"], "VALIDATED")
                first = json.loads((workspace/"shards/index.json").read_text())["entries"]
                count = len(validations)
                self.assertEqual(campaign.run(workspace, "smoke-serial", force_local=True, resume=True)["status"], "COMPLETE")
                self.assertEqual(len(validations), count)
                final = json.loads((workspace/"shards/index.json").read_text())["entries"]
                for key, receipt in first.items(): self.assertEqual(final[key], receipt)
            self.assertEqual(len(collects), 1)
            self.assertEqual(len(final), 2)

    def test_real_scheduler_never_retries_a_journaled_start(self):
        with tempfile.TemporaryDirectory(prefix="v4_started_synthetic_") as tmp:
            output = Path(tmp)
            identity = dict(synthetic=True)
            entries = [dict(trial_id="started_once", backend="a3t_safe", execution_status="RUNNING", request=dict(planner_seed=11)),
                       dict(trial_id="already_done", backend="a3t_old", execution_status="COMPLETED", request=dict(planner_seed=11))]
            plan = dict(entries=entries, protocol_id="a3t_old_safe_sem_kaggle_v4", schema_version=13)
            manifest = dict(plan, identity=identity, concurrency_timeline=[], sessions=[], status="RUNNING")
            dump(output/"manifest.json", manifest)
            (output/"started_once").mkdir()
            journal = output/"start_journal.jsonl"
            journal.write_text(json.dumps(dict(trial_id="started_once", identity_sha256=digest(identity)))+"\n")
            before = journal.read_bytes()
            runtime = dict(eligible_cpus=[0], memory_limit_bytes=16*storage.GIB)
            policy = dict(worker_count=1, thread_environment={})
            with patch.object(scheduler, "_process_alive", return_value=False), patch.object(
                    scheduler.subprocess, "Popen", side_effect=AssertionError("started trial must not launch again")):
                result = scheduler._run_locked(plan, {}, output, True, runtime, policy, identity, 0., 28800.)
            self.assertEqual(result["entries"][0]["execution_status"], "INTERRUPTED")
            self.assertEqual(result["entries"][1]["execution_status"], "COMPLETED")
            self.assertEqual(journal.read_bytes(), before)
            self.assertEqual(result["entries"][0]["request"]["planner_seed"], 11)

    def test_observed_large_wave_disk_budget_survives_resume(self):
        with tempfile.TemporaryDirectory(prefix="v4_disk_resume_synthetic_") as tmp:
            workspace = Path(tmp); prepared = self.fixture(workspace, map_count=5)
            collects, validations, admissions = [], [], []
            def compact(directory, request, shared):
                dump(Path(directory)/"compaction.json", dict(synthetic=True))
            def admission(*args, **kwargs):
                admissions.append(kwargs)
                return dict(allowed=len(admissions) == 1, reason="SYNTHETIC_DISK_LIMIT")
            def packed(store, wave, mid, cleanup=True):
                # Synthetic byte counters simulate a large wave without writing
                # gigabytes or launching any planner.
                return dict(archive="shards/"+wave+"__"+mid+".zip",
                            uncompressed_bytes=2*storage.GIB, archive_bytes=storage.GIB)
            with self.run_patches(prepared, self.fake_collect(collects), self.fake_validate(validations), compact):
                with patch("native_paper_v4_storage.preflight", side_effect=admission), patch.object(
                        storage.ArtifactStore, "pack_map", packed):
                    first = campaign.run(workspace, "smoke-serial", force_local=True)
                    self.assertEqual(first["status"], "PARTIAL_DISK")
                    self.assertGreaterEqual(admissions[-1]["raw_budget_bytes"], 10*storage.GIB)
                    expected = admissions[-1]["raw_budget_bytes"]
                    second = campaign.run(workspace, "smoke-serial", force_local=True, resume=True)
                    self.assertEqual(second["status"], "PARTIAL_DISK")
                    self.assertGreaterEqual(admissions[-1]["raw_budget_bytes"], expected)
            self.assertEqual(len(collects), 1)

    def test_ram_hold_preserves_drained_data_and_resumes_validation_only(self):
        with tempfile.TemporaryDirectory(prefix="v4_ram_resume_synthetic_") as tmp:
            workspace = Path(tmp); prepared = self.fixture(workspace)
            collects, validations = [], []
            def compact(directory, request, shared):
                dump(Path(directory)/"compaction.json", dict(synthetic=True))
            with self.run_patches(prepared, self.fake_collect(collects), self.fake_validate(validations), compact):
                with patch("native_paper_v3_memory.admission_snapshot", side_effect=[
                        dict(absolute_headroom_bytes=storage.GIB), dict(absolute_headroom_bytes=48*storage.GIB)]):
                    first = campaign.run(workspace, "smoke-serial", force_local=True)
                    self.assertEqual(first["status"], "PARTIAL_RAM")
                    self.assertEqual(first["waves"][0]["status"], "DRAINED")
                    self.assertEqual(validations, [])
                    self.assertTrue(list((workspace/"active").rglob("worker.json")))
                    second = campaign.run(workspace, "smoke-serial", force_local=True, resume=True)
                    self.assertEqual(second["status"], "COMPLETE")
            self.assertEqual(len(collects), 1)
            self.assertEqual(len(validations), 2)

    def test_four_planner_lanes_fail_before_collection_when_reservations_do_not_fit(self):
        with tempfile.TemporaryDirectory(prefix="v4_four_lanes_synthetic_") as tmp:
            workspace = Path(tmp); prepared = self.fixture(workspace)
            collects, validations = [], []
            def compact(*args):
                raise AssertionError("no trial may be compacted on rejected resource admission")
            with self.run_patches(prepared, self.fake_collect(collects), self.fake_validate(validations), compact):
                with patch.object(campaign, "detect_runtime", return_value=dict(cpu_capacity=4,
                            memory_limit_bytes=4*storage.GIB, memory_sources=[])), patch.object(
                            campaign, "require_smoke", return_value=dict(synthetic=True)), patch.object(
                            campaign, "collect_reservations", return_value={b:2*storage.GIB for b in campaign.BACKENDS}):
                    with self.assertRaisesRegex(RuntimeError, "four planner lanes"):
                        campaign.run(workspace, "smoke-parallel", force_local=True)
            self.assertEqual(collects, [])
            self.assertEqual(validations, [])


class ReadinessTests(unittest.TestCase):
    def fixture(self, workspace):
        mode, wave = "smoke-serial", "smoke-serial-w000"
        store = storage.ArtifactStore(workspace)
        entries = []
        code = validation_code_identity()
        for k in range(7):
            mid = "synthetic_"+str(k)
            for backend in campaign.BACKENDS:
                tid = mid+"__"+backend
                entry = dict(trial_id=tid, backend=backend, map_id=mid, wave_id=wave,
                             execution_status="COMPLETED", request=dict(backend=backend, planner_seed=11))
                entries.append(entry)
                directory = workspace/"active"/wave/mid/tid
                row = dict(trial_id=tid, backend=backend, errors=[], correctness_errors=[], validation_code_identity=code)
                verdict_dump(directory/"validation.json", row)
                digest_value = storage.file_hash(directory/"validation.json")
                dump(workspace/"verdicts"/mode/(tid+".json"),
                     dict(row=row, receipt=dict(verdict_sha256=digest_value), verdict_sha256=digest_value))
            store.pack_map(wave, mid)
        dump(workspace/"prepared.json", dict(identity=dict(synthetic=True)))
        dump(workspace/"plans/smoke.json", dict(entries=copy.deepcopy(entries)))
        manifest = dict(status="COMPLETE", entries=entries, max_concurrency=1)
        dump(workspace/"manifests"/(mode+".json"), manifest)
        dump(workspace/"readiness"/(mode+".json"), campaign.smoke_evidence(workspace, mode))
        return mode, manifest

    def test_valid_bound_readiness_then_modified_ledger_rejected(self):
        with tempfile.TemporaryDirectory(prefix="v4_readiness_synthetic_") as tmp:
            workspace = Path(tmp); mode, manifest = self.fixture(workspace)
            self.assertEqual(campaign.require_smoke(workspace, mode)["status"], "COMPLETE")
            tid = manifest["entries"][0]["trial_id"]
            path = workspace/"verdicts"/mode/(tid+".json")
            ledger = json.loads(path.read_text()); ledger["row"]["unexpected_edit"] = True
            dump(path, ledger)
            with self.assertRaisesRegex(ValueError, "hash"):
                campaign.require_smoke(workspace, mode)

    def test_empty_complete_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="v4_empty_smoke_synthetic_") as tmp:
            workspace = Path(tmp)
            dump(workspace/"plans/smoke.json", dict(entries=[]))
            dump(workspace/"manifests/smoke-serial.json", dict(status="COMPLETE", entries=[]))
            with self.assertRaises(ValueError):
                campaign.require_smoke(workspace, "smoke-serial")

    def test_stale_verifier_and_changed_shard_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="v4_stale_smoke_synthetic_") as tmp:
            workspace = Path(tmp); mode, _ = self.fixture(workspace)
            with patch("native_paper_v4_analysis.validation_code_identity", return_value="changed_synthetic_verifier"):
                with self.assertRaisesRegex(ValueError, "stale smoke verifier"):
                    campaign.require_smoke(workspace, mode)
            shard = next((workspace/"shards").glob("*.zip"))
            with shard.open("ab") as stream: stream.write(b"synthetic_corruption")
            with self.assertRaisesRegex(ValueError, "shard changed"):
                campaign.require_smoke(workspace, mode)

    def test_changed_frozen_request_or_prepared_identity_invalidates_readiness(self):
        with tempfile.TemporaryDirectory(prefix="v4_identity_smoke_synthetic_") as tmp:
            workspace = Path(tmp); mode, manifest = self.fixture(workspace)
            dump(workspace/"prepared.json", dict(identity=dict(synthetic="changed")))
            with self.assertRaisesRegex(ValueError, "stale or missing"):
                campaign.require_smoke(workspace, mode)
            manifest["entries"][0]["request"]["planner_seed"] = 99
            dump(workspace/"manifests"/(mode+".json"), manifest)
            with self.assertRaisesRegex(ValueError, "request drift"):
                campaign.require_smoke(workspace, mode)


if __name__ == "__main__":
    unittest.main()
