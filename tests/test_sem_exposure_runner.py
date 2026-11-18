"""Coordinator tests with explicitly synthetic failures, never real campaign data."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_sem_exposure as runner
import sem_exposure_protocol as protocol
from corridor_v4_execution import atomic_json as dump, digest, read


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="__sem_exposure_synthetic_", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "synthetic_campaign"
        self.output.mkdir()
        self.config = protocol.default_config()
        self.manifest = protocol.inventory(self.config)
        self.manifest.update(status="PREPARED", synthetic_fixture=True, sources={"fixture": "synthetic"}, sessions=[],
            maps=[dict(map_id=family+"_101", family=family) for family in self.config["profiles"]["smoke"]["families"]])
        dump(self.output / "config.json", self.config)
        dump(self.output / "manifest.json", self.manifest)
        self.worker, self.bridge = self.root / "synthetic_worker.exe", self.root / "synthetic_bridge.dll"
        for path in (self.worker, self.bridge):
            path.write_text("SYNTHETIC TEST PLACEHOLDER; NEVER EXECUTE", encoding="utf-8")
        self.calls, self.collected = [], 0

    def fake_spawn(self, lanes, target, control, *, cores, on_start=None, on_complete=None):
        tasks = [task for lane in lanes for task in lane]
        self.calls.append((target.__name__, len(tasks)))
        if target is runner._collect_trial:
            for task in tasks:
                on_start(task, dict(started_utc="2026-11-12T00:00:00+00:00", lane=0, core=0, queue_wait_s=0.))
                directory = Path(task["directory"])
                directory.mkdir(parents=True)
                execution = dict(status="CRASH", exit_code=1, launched=True, synthetic_fixture=True)
                dump(directory / "execution.json", execution)
                raw = runner.raw_inventory(directory)
                dump(directory / "raw_inventory.json", raw)
                on_complete(task, dict(status="COMPLETED", payload=dict(execution=execution,
                    raw_inventory_sha256=digest(raw))))
                self.collected += 1
        else:
            self.assertIs(target, runner._validate_map)
            self.assertEqual(self.collected, 42, "validation started before complete collection")
            for task in tasks:
                for item in task["items"]:
                    entry = item["entry"]
                    row = dict(**{key: entry[key] for key in ("trial_id", "method", "map_id", "family", "planner_seed", "alpha", "sem_lambda")},
                        synthetic_fixture=True, success=False, attained_safe=False, execution_status="CRASH",
                        execution_launched=True, correctness_errors=[], events=[])
                    identity = dict(trial_id=entry["trial_id"], raw_inventory_sha256=item["raw_inventory_sha256"],
                        sources_sha256=task["sources_sha256"], info_sha256=digest(task["info"]))
                    dump(Path(task["spool"]) / (entry["trial_id"] + ".json"), dict(identity=identity, row=row))
                on_complete(task, dict(status="COMPLETED"))
        return dict(status="COMPLETE", wall_s=.01, max_concurrency=4 if tasks else 0)

    def execute(self, **kwargs):
        with patch.object(protocol, "source_hashes", return_value={"fixture": "synthetic"}), \
             patch.object(runner, "resource_preflight", return_value=dict(admitted=True, cores=[0, 1, 2, 3])), \
             patch.object(runner, "run_spawn_lanes", side_effect=self.fake_spawn):
            return runner.run(self.output, self.worker, self.bridge, **kwargs)

    def test_all_collection_precedes_validation_and_resume_does_not_retry(self):
        summary = self.execute()
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["attempted"], 42)
        self.assertEqual(summary["success"], 0)
        self.assertEqual(self.calls, [("_collect_trial", 42), ("_validate_map", 7)])
        self.calls.clear()
        again = self.execute(resume=True)
        self.assertEqual(again["status"], "COMPLETE")
        self.assertFalse(any(name == "_collect_trial" for name, _ in self.calls))
        lines = (self.output / "execution_control/start_journal.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 42)

    def test_raw_mutation_blocks_resume_without_new_collection(self):
        self.execute()
        entry = self.manifest["entries"][0]
        path = self.output / "trials" / entry["trial_id"] / "execution.json"
        path.write_text('{"synthetic_fixture":true,"tampered":true}', encoding="utf-8")
        self.calls.clear()
        summary = self.execute(resume=True)
        self.assertEqual(summary["status"], "FAILED_INFRASTRUCTURE")
        self.assertEqual(self.calls, [])
        self.assertIn("committed trial changed", read(self.output / "manifest.json")["error"])

    def test_inventory_does_not_generate_inputs_or_accept_overwrite(self):
        output = self.root / "synthetic_inventory"
        with patch.object(protocol, "source_hashes", return_value={"fixture": "synthetic"}):
            result = runner.prepare(output, dry_run=True)
            self.assertEqual(result["planned_trials"], 42)
            self.assertEqual(result["planner_execution_count"], 0)
            self.assertFalse((output / "inputs").exists())
            with self.assertRaises(FileExistsError):
                runner.prepare(output, dry_run=True)

    def test_source_drift_rejected_before_any_launch(self):
        with patch.object(protocol, "source_hashes", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "frozen source"):
                runner.run(self.output, self.worker, self.bridge)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
