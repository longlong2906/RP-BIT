"""Synthetic controller faults under tests/fixtures; never experimental results."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pri_portfolio_protocol as protocol
import run_pri_portfolio as runner
from corridor_v4_execution import append_journal, atomic_json as dump, digest, file_hash, read


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="__pri_portfolio_synthetic_", dir=ROOT / "tests/fixtures")
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

    def test_cli_requires_explicit_pilot_and_full_flags(self):
        for command in ("pilot", "confirm"):
            with self.assertRaises(SystemExit) as caught:
                runner.main([command, "--output", str(self.output), "--worker", str(self.worker), "--bridge", str(self.bridge)])
            self.assertEqual(caught.exception.code, 2)

    def synthetic_predecessor(self):
        """200 frozen pilot attempts and 70 smoke attempts; no physical results."""
        source = self.root / "synthetic_predecessor"
        old_sources = {"fixture": "synthetic", "tools/run_pri_portfolio.py": "old_controller",
                       "tools/pri_portfolio_validation.py": "old_oracle"}
        current = dict(old_sources, **{"tools/run_pri_portfolio.py": "recovery_controller",
                                     "tools/pri_portfolio_validation.py": "corrected_oracle"})
        for phase, count in (("smoke_serial", 35), ("smoke_parallel", 35), ("pilot", 200)):
            directory = source / phase
            manifest = protocol.inventory(self.config, phase)
            manifest.update(status="FAILED_CORRECTNESS" if phase == "pilot" else "COMPLETE",
                synthetic_fixture=True, sources=old_sources, sessions=[], workers=1 if phase == "smoke_serial" else 4,
                validation_workers=4, binary_sha256=file_hash(self.worker), bridge_sha256=file_hash(self.bridge),
                # A virtualenv's launch path is part of its runtime identity;
                # resolving its symlink would silently name the base Python.
                sem_python=os.path.abspath(sys.executable), sem_python_sha256=file_hash(sys.executable),
                runtime_sha256=digest({"synthetic_fixture": True}),
                maps=[dict(map_id=tid, start=[1.,1.], goal=[2.,2.]) for tid in dict.fromkeys(e["map_id"] for e in manifest["entries"])])
            for i, entry in enumerate(manifest["entries"][:count]):
                tid = entry["trial_id"]
                trial = directory / "trials" / tid
                dump(trial / "execution.json", dict(status="CRASH", exit_code=1, launched=True, synthetic_fixture=True))
                inventory = runner.raw_inventory(trial)
                dump(trial / "raw_inventory.json", inventory)
                entry.update(status="CRASH", raw_inventory_sha256=digest(inventory), started_utc="2026-11-12T00:00:00+00:00",
                             lane=entry["block_index"]%manifest["workers"], core=entry["block_index"]%manifest["workers"], queue_wait_s=0.)
                row = dict(**{k:entry[k] for k in ("trial_id","method","map_id","family","planner_seed","alpha","phase")},
                    success=False, attained_safe=False, correctness_errors=["synthetic_old_oracle_error"] if i == 0 and phase == "pilot" else [],
                    events=[], execution_status="CRASH", synthetic_fixture=True)
                dump(trial / "validation.json", row)
                identity = dict(trial_id=tid, raw_inventory_sha256=digest(inventory), sources_sha256=digest(old_sources),
                    info_sha256=digest(next(info for info in manifest["maps"] if info["map_id"]==entry["map_id"])))
                spool = directory / "execution_control/validation_spool" / (tid+".json")
                dump(spool, dict(identity=identity,row=row))
                entry.update(validation_sha256=file_hash(trial/"validation.json"), validation_spool_sha256=file_hash(spool))
                append_journal(directory/"execution_control/start_journal.jsonl", dict(trial_id=tid,
                    config_sha256=manifest["config_sha256"], sources_sha256=digest(old_sources), started_utc=entry["started_utc"],
                    lane=entry["lane"], core=entry["core"], session_index=0))
                self.collected.add(tid)
            dump(directory/"config.json",self.config)
            dump(directory/"manifest.json",manifest)
        return source, current

    def test_venv_launch_identity_is_preserved_and_base_interpreter_is_rejected(self):
        launch = self.root / "synthetic_venv" / "bin" / "python"
        base = self.root / "synthetic_base_python"
        launch.parent.mkdir(parents=True)
        for path in (launch, base):
            path.write_text("SYNTHETIC INTERPRETER; NEVER EXECUTE", encoding="utf-8")
        self.assertEqual(file_hash(launch), file_hash(base))
        original_resolve = Path.resolve

        def resolve_alias(path, *args, **kwargs):
            return base if path == launch else original_resolve(path, *args, **kwargs)

        # Model a venv symlink without requiring Windows symlink privileges.
        # Before the fix, synthetic_predecessor incorrectly recorded `base`.
        with patch.object(sys, "executable", str(launch)), \
             patch.object(Path, "resolve", autospec=True, side_effect=resolve_alias):
            self.assertEqual(Path(sys.executable).resolve(), base)
            source, _ = self.synthetic_predecessor()
            for phase in ("smoke_serial", "smoke_parallel", "pilot"):
                self.assertEqual(read(source / phase / "manifest.json")["sem_python"], str(launch))
            self.assertEqual(self.execute(sem_python=launch)["status"], "COMPLETE")
            self.calls.clear()
            self.assertEqual(self.execute(sem_python=launch, resume=True)["status"], "COMPLETE")
            with self.assertRaisesRegex(ValueError, "resume binary/interpreter identity mismatch"):
                self.execute(sem_python=base, resume=True)
            self.assertEqual(self.calls, [])

    def test_recovery_revalidates_all_and_dispatches_only_175_unstarted(self):
        source, current = self.synthetic_predecessor()
        output = self.root / "synthetic_recovery"
        previous = {str(p.relative_to(source)):file_hash(p) for p in source.rglob("*") if p.is_file()}
        original_starts = (source/"pilot/execution_control/start_journal.jsonl").read_bytes()
        with patch.object(protocol,"source_hashes",return_value=current), patch.object(runner,"_verify_map"), \
             patch.object(runner,"_runtime_identity",return_value={"synthetic_fixture":True}), \
             patch.object(runner,"resource_preflight",return_value=dict(admitted=True,cores=[0,1,2,3])), \
             patch.object(runner,"run_spawn_lanes",side_effect=self.fake_spawn), patch.object(runner,"_smoke_gate",return_value={}), \
             patch.object(runner,"analyze",return_value={"status":"DONE"}):
            runner.migrate(source,output)
            with self.assertRaisesRegex(ValueError,"must pass revalidation"):
                runner.run(output,"pilot",self.worker,self.bridge,resume=True,allow_pilot=True)
            result=runner.recover(source,output,self.worker,self.bridge,allow_pilot=True,resume=True)
            self.assertEqual(read(output/"pilot/manifest.json")["migration"]["revalidation_complete"],True)
            first_collection=next(i for i,(name,_) in enumerate(self.calls) if name=="_collect_trial")
            # Seven maps per smoke plus eight maps (200 trials) in the pilot
            # must all finish validation before any of the 175 remaining solves.
            self.assertEqual(sum(count for _,count in self.calls[:first_collection]),22)
            self.assertEqual(result["attempted"],375)
            self.assertEqual(result["status"],"COMPLETE")
            self.assertEqual(sum(count for name,count in self.calls if name=="_collect_trial"),175)
            starts=(output/"pilot/execution_control/start_journal.jsonl").read_bytes()
            self.assertTrue(starts.startswith(original_starts))
            records=[json.loads(line) for line in starts.splitlines()]
            self.assertEqual(len(records),375)
            self.assertEqual(len({item["trial_id"] for item in records}),375)
            self.assertNotEqual(records[0]["sources_sha256"],records[-1]["sources_sha256"])
            runner._audit(output/"pilot",require_complete=True)
            # Simulate interruption after committing a newly collected verdict,
            # before the controller records its wave's correctness failure.
            manifest=read(output/"pilot/manifest.json")
            entry=manifest["entries"][-1]
            verdict=output/"pilot/trials"/entry["trial_id"]/"validation.json"
            row=read(verdict)
            row["correctness_errors"]=["synthetic_new_attempt_failure_before_checkpoint"]
            dump(verdict,row)
            spool_path=output/"pilot/execution_control/validation_spool"/(entry["trial_id"]+".json")
            spool=read(spool_path)
            spool["row"]=row
            dump(spool_path,spool)
            entry.update(validation_sha256=file_hash(verdict),validation_spool_sha256=file_hash(spool_path))
            manifest["status"]="INTERRUPTED"
            dump(output/"pilot/manifest.json",manifest)
            self.calls.clear()
            result=runner.run(output,"pilot",self.worker,self.bridge,resume=True,allow_pilot=True)
            self.assertEqual(result["status"],"FAILED_CORRECTNESS")
            self.assertEqual(self.calls,[])
            manifest["status"]="INTERRUPTED"
            dump(output/"pilot/manifest.json",manifest)
            result=runner.run(output,"pilot",self.worker,self.bridge,resume=True,allow_pilot=True,validation_only=True)
            self.assertEqual(result["status"],"FAILED_CORRECTNESS")
            self.assertFalse(read(output/"pilot/manifest.json")["migration"]["revalidation_complete"])
            self.assertEqual(self.calls,[])
        self.assertEqual(previous,{str(p.relative_to(source)):file_hash(p) for p in source.rglob("*") if p.is_file()})

    def test_recovery_rejects_planner_drift_and_import_reset(self):
        source,current=self.synthetic_predecessor()
        bad=dict(current,fixture="changed_generator")
        with patch.object(runner,"_verify_map"), patch.object(protocol,"source_hashes",return_value=bad):
            with self.assertRaisesRegex(ValueError,"only the named oracle"):
                runner.migrate(source,self.root/"rejected_recovery")
        self.assertFalse((self.root/"rejected_recovery").exists())
        output=self.root/"synthetic_recovery"
        with patch.object(runner,"_verify_map"), patch.object(protocol,"source_hashes",return_value=current):
            runner.migrate(source,output)
            manifest=read(output/"pilot/manifest.json")
            manifest["entries"][0]["status"]="NOT_RUN"
            dump(output/"pilot/manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"cannot be reset"):
                runner._audit(output/"pilot")

    def test_recovery_rejects_original_or_copied_raw_tamper(self):
        source,current=self.synthetic_predecessor()
        output=self.root/"synthetic_recovery"
        with patch.object(runner,"_verify_map"), patch.object(protocol,"source_hashes",return_value=current):
            runner.migrate(source,output)
            tid=read(output/"pilot/manifest.json")["entries"][0]["trial_id"]
            raw=output/"pilot/trials"/tid/"execution.json"
            original=raw.read_bytes()
            raw.write_text('{"synthetic_tamper":true}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"copied raw attempt changed"):
                runner._audit(output/"pilot")
            raw.write_bytes(original)
            predecessor=source/"pilot/trials"/tid/"validation.json"
            predecessor.write_text('{"synthetic_tamper":true}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"original verdict changed"):
                runner._audit(output/"pilot")

    def test_recovery_failure_revalidates_all_imports_but_never_collects(self):
        source,current=self.synthetic_predecessor()
        output=self.root/"synthetic_recovery"
        def faulty_validation(lanes,target,control,**kwargs):
            callback=kwargs.get("on_complete")
            if target is runner._validate_map:
                def commit_fault(task,value):
                    if task["items"][0]["entry"]["phase"]=="pilot":
                        sid=task["items"][0]["entry"]["trial_id"]
                        path=Path(task["spool"])/(sid+".json")
                        spool=read(path)
                        spool["row"]["correctness_errors"]=["synthetic_persistent_oracle_failure"]
                        dump(path,spool)
                    callback(task,value)
                kwargs["on_complete"]=commit_fault
            return self.fake_spawn(lanes,target,control,**kwargs)
        with patch.object(protocol,"source_hashes",return_value=current), patch.object(runner,"_verify_map"), \
             patch.object(runner,"_runtime_identity",return_value={"synthetic_fixture":True}), \
             patch.object(runner,"resource_preflight",return_value=dict(admitted=True,cores=[0,1,2,3])), \
             patch.object(runner,"run_spawn_lanes",side_effect=faulty_validation), patch.object(runner,"_smoke_gate",return_value={}), \
             patch.object(runner,"analyze",return_value={"status":"DONE"}):
            result=runner.recover(source,output,self.worker,self.bridge,allow_pilot=True)
            self.assertEqual(result["status"],"FAILED_CORRECTNESS")
            self.assertEqual(result["validated"],200)
            self.assertFalse(any(name=="_collect_trial" for name,_ in self.calls))
            self.assertFalse(read(output/"pilot/manifest.json")["migration"]["revalidation_complete"])


if __name__ == "__main__":
    unittest.main()
