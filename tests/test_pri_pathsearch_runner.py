"""Synthetic controller faults under tests/fixtures; never experimental results."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pri_pathsearch_protocol as protocol
import run_pri_pathsearch as runner
from corridor_v4_execution import append_journal, atomic_json as dump, digest, file_hash, read


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="__pri_pathsearch_synthetic_", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "synthetic_campaign"
        self.directory = self.output / "smoke_serial"
        self.directory.mkdir(parents=True)
        self.config = protocol.default_config()
        self.manifest = protocol.inventory(self.config, "smoke_serial")
        map_ids = list(dict.fromkeys(entry["map_id"] for entry in self.manifest["entries"]))
        self.manifest.update(status="PREPARED", synthetic_fixture=True, sources={"fixture": "synthetic"},
            sessions=[], workers=1, validation_workers=4,
            maps=[dict(map_id=map_id, start=[1., 1.], goal=[2., 2.]) for map_id in map_ids])
        dump(self.directory / "config.json", self.config)
        dump(self.directory / "manifest.json", self.manifest)
        self.worker, self.bridge = self.root / "synthetic_worker.exe", self.root / "synthetic_bridge.dll"
        for path in (self.worker, self.bridge):
            path.write_text("SYNTHETIC TEST PLACEHOLDER; NEVER EXECUTE", encoding="utf-8")
        self.calls, self.collected = [], set()

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
                self.collected.add(task["task_id"])
        else:
            self.assertIs(target, runner._validate_map)
            for task in tasks:
                for item in task["items"]:
                    entry = item["entry"]
                    self.assertIn(entry["trial_id"], self.collected, "oracle ran before raw collection joined")
                    row = dict(**{key: entry[key] for key in ("trial_id", "method", "map_id", "family", "planner_seed", "alpha", "phase")},
                        synthetic_fixture=True, success=False, attained_safe=False, execution_status="CRASH",
                        execution_launched=True, correctness_errors=[], events=[])
                    identity = dict(trial_id=entry["trial_id"], raw_inventory_sha256=item["raw_inventory_sha256"],
                        sources_sha256=task["sources_sha256"], info_sha256=digest(task["info"]))
                    dump(Path(task["spool"]) / (entry["trial_id"] + ".json"), dict(identity=identity, row=row))
                on_complete(task, dict(status="COMPLETED"))
        return dict(status="COMPLETE", wall_s=.01, max_concurrency=1)

    def execute(self, **kwargs):
        with patch.object(protocol, "source_hashes", return_value={"fixture": "synthetic"}), \
             patch.object(runner, "resource_preflight", return_value=dict(admitted=True, cores=[0, 1, 2, 3])), \
             patch.object(runner, "_runtime_identity", return_value={"synthetic_fixture": True}), \
             patch.object(runner, "_verify_map"), \
             patch.object(runner, "run_spawn_lanes", side_effect=self.fake_spawn):
            return runner.run(self.output, "smoke_serial", self.worker, self.bridge, **kwargs)

    def test_wave_order_all_crashes_count_and_resume_never_retries(self):
        summary = self.execute()
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["attempted"], 35)
        self.assertEqual(summary["success"], 0)
        self.assertEqual(self.calls, [("_collect_trial", 20), ("_validate_map", 4),
                                     ("_collect_trial", 15), ("_validate_map", 3)])
        self.calls.clear()
        again = self.execute(resume=True)
        self.assertEqual(again["status"], "COMPLETE")
        self.assertEqual(self.calls, [])
        starts = (self.directory / "execution_control/start_journal.jsonl").read_text().splitlines()
        self.assertEqual(len(starts), 35)

    def test_raw_mutation_blocks_resume_without_dispatch(self):
        self.execute()
        entry = self.manifest["entries"][0]
        path = self.directory / "trials" / entry["trial_id"] / "execution.json"
        path.write_text('{"synthetic_fixture":true,"tampered":true}', encoding="utf-8")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "raw trial changed"):
            self.execute(resume=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(read(self.directory / "manifest.json")["status"], "FAILED_INFRASTRUCTURE")

    def test_manifest_query_tamper_breaks_spool_binding(self):
        self.execute()
        manifest = read(self.directory / "manifest.json")
        manifest["maps"][0]["goal"] = [8., 8.]
        dump(self.directory / "manifest.json", manifest)
        with patch.object(runner, "_verify_map"), self.assertRaisesRegex(ValueError, "raw/map/query/source"):
            runner._audit(self.directory)

    def test_duplicate_journal_is_detected(self):
        self.execute()
        journal = self.directory / "execution_control/start_journal.jsonl"
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(journal.read_text().splitlines()[0] + "\n")
        with patch.object(runner, "_verify_map"), self.assertRaisesRegex(ValueError, "attempt journal"):
            runner._audit(self.directory)

    def test_spool_tamper_is_detected(self):
        self.execute()
        entry = self.manifest["entries"][0]
        spool = self.directory / "execution_control/validation_spool" / (entry["trial_id"] + ".json")
        changed = read(spool)
        changed["row"]["success"] = True
        dump(spool, changed)
        with patch.object(runner, "_verify_map"), self.assertRaisesRegex(ValueError, "spool changed"):
            runner._audit(self.directory)

    def test_synthetic_smoke_cannot_promote_to_pilot(self):
        self.execute()
        with patch.object(runner, "_verify_map"), self.assertRaisesRegex(ValueError, "synthetic fixtures"):
            runner._smoke_gate(self.output)

    def test_inventory_never_generates_maps_and_refuses_overwrite(self):
        output = self.root / "synthetic_inventory"
        with patch.object(protocol, "source_hashes", return_value={"fixture": "synthetic"}):
            result = runner.prepare(output, dry_run=True)
            self.assertEqual(result["planned_trials"], 35)
            self.assertEqual(result["planner_execution_count"], 0)
            self.assertFalse((output / "smoke_serial/inputs").exists())
            with self.assertRaises(FileExistsError):
                runner.prepare(output, dry_run=True)

    def test_cli_rejects_confirmation_full_and_implicit_pilot(self):
        for command in ("pilot", "confirm", "full"):
            with self.assertRaises(SystemExit) as caught:
                runner.main([command, "--output", str(self.output), "--worker", str(self.worker), "--bridge", str(self.bridge)])
            self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit):
            runner.main(["pilot", "--output", str(self.output), "--allow-pilot", "--allow-full"])

    def test_scientific_source_drift_blocks_dispatch(self):
        with patch.object(protocol, "source_hashes", return_value={"fixture":"changed"}):
            with self.assertRaisesRegex(ValueError, "scientific sources changed"):
                runner.run(self.output,"smoke_serial",self.worker,self.bridge)
        self.assertEqual(self.calls, [])

    def test_selector_outputs_do_not_create_dispatches(self):
        self.execute()
        self.assertEqual(len(self.collected), 35)
        self.assertEqual({e["method"] for e in self.manifest["entries"]}, set(protocol.METHODS))
        self.assertEqual(sum(n for name,n in self.calls if name == "_collect_trial"), 35)


if __name__ == "__main__":
    unittest.main()
