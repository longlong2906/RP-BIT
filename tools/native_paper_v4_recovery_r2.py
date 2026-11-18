"""External affinity-guard correction; preserve 893 attempts and admit only 607 IDs.

The frozen repository and the R1 oracle stay unchanged. Only the scheduler loop
is overlaid; its base identity is retained for the existing append-only journals.
The effective controller hash is recorded separately before every new launch.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

POLICY_ID = "v4_affinity_recovery_v2"
DERIVED = {"validation.json", "validation_receipt.json"}
LOCATION_FIELDS = {"directory", "wave_id"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def entry_state(entry):
    return {key:value for key,value in entry.items() if key not in LOCATION_FIELDS}


class RecoveryV2:
    def __init__(self, root, workspace, patch_dir, manifest, manifest_sha256, *, _r1=None):
        self.root, self.w, self.patch_dir = map(lambda p:Path(p).resolve(), (root, workspace, patch_dir))
        self.manifest_path = Path(manifest).resolve()
        if self.patch_dir.is_relative_to(self.root) or self.manifest_path.parent != self.patch_dir:
            raise ValueError("R2 patch must remain outside the frozen repository")
        if sha(self.manifest_path) != manifest_sha256:
            raise ValueError("R2 manifest SHA mismatch")
        self.manifest_sha = manifest_sha256
        self.testing = _r1 is not None
        self.m = read(self.manifest_path)
        if self.m.get("policy_id") != POLICY_ID:
            raise ValueError("unknown affinity recovery policy")
        if sha(__file__) != self.m["runner"]["sha256"]:
            raise ValueError("executing R2 runner differs from the bound source")
        for key in ("runner", "scheduler", "r1_runner", "analyzer", "r1_manifest", "test_scheduler", "test_runner"):
            record = self.m[key]
            path = self.patch_dir/record["path"]
            if Path(record["path"]).name != record["path"] or path.is_symlink() or sha(path) != record["sha256"]:
                raise ValueError("bound R2 patch changed: "+key)
        if _r1 is None:
            r1 = load_module("_native_paper_v4_r1_bound", self.patch_dir/self.m["r1_runner"]["path"])
            self.r1 = r1.RecoveryRunner(self.root, self.w,
                self.patch_dir/self.m["analyzer"]["path"], self.patch_dir/self.m["r1_manifest"]["path"],
                self.m["r1_manifest"]["sha256"], analyzer_sha256=self.m["analyzer"]["sha256"])
        else:
            self.r1 = _r1
        self.campaign = self.r1.campaign
        self.policy_dir = self.w/"recovery"/POLICY_ID
        self.r1_module = sys.modules[self.r1.__class__.__module__] if _r1 is None else None
        self.dump = self.r1_module.dump if self.r1_module else _r1.dump
        self.owned = self.r1_module._owned if self.r1_module else _r1.owned
        self.copy = self.r1_module._copy if self.r1_module else _r1.copy
        self.owned(self.policy_dir, self.w)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.fixed = load_module("_native_paper_v4_scheduler_r2", self.patch_dir/self.m["scheduler"]["path"])
        self.fixed.LAUNCHER_PATH = self.root/"tools/terrain_compare_launch.py"
        self.guard()

    def guard(self):
        self.r1.guard()
        if sha(self.manifest_path) != self.manifest_sha:
            raise ValueError("R2 manifest changed")
        if sha(__file__) != self.m["runner"]["sha256"]:
            raise ValueError("executing R2 runner changed")
        for key in ("runner", "scheduler", "r1_runner", "analyzer", "r1_manifest", "test_scheduler", "test_runner"):
            if sha(self.patch_dir/self.m[key]["path"]) != self.m[key]["sha256"]:
                raise ValueError("R2 implementation changed: "+key)
        if sha(self.root/"tools/terrain_compare_scheduler.py") != self.m["original_scheduler_sha256"]:
            raise ValueError("frozen base scheduler changed")

    def main(self):
        return read(self.w/"manifests/main.json")

    def declared_directory(self, entry, manifest=None):
        if entry.get("directory"):
            path = Path(entry["directory"])
        else:
            manifest = manifest or self.main()
            wave = next(v["wave_id"] for v in manifest["waves"] if entry["map_id"] in v["map_ids"])
            path = self.w/"collection"/wave/entry["trial_id"]
        self.owned(path, self.w)
        return path

    def directory(self, entry, manifest=None):
        manifest = manifest or self.main()
        wave = entry.get("wave_id") or next(v["wave_id"] for v in manifest["waves"] if entry["map_id"] in v["map_ids"])
        candidates = {self.declared_directory(entry, manifest), self.w/"collection"/wave/entry["trial_id"],
                      self.w/"active"/wave/entry["map_id"]/entry["trial_id"]}
        existing = []
        for path in candidates:
            self.owned(path, self.w)
            if path.is_dir():
                existing.append(path)
        if len(existing) > 1:
            raise ValueError("ambiguous duplicate locations for an original trial")
        return existing[0] if existing else self.declared_directory(entry, manifest)

    def raw_inventory(self, entries):
        """Hash original bytes at either their current location or immutable shard."""
        main = self.main()
        index_path = self.w/"shards/index.json"
        shards = read(index_path)["entries"] if index_path.exists() else {}
        answer = {}
        checked_shards = set()
        for entry in entries:
            tid = entry["trial_id"]
            directory = self.directory(entry, main)
            if directory.is_dir():
                files = {}
                for path in sorted(directory.rglob("*")):
                    self.owned(path, self.w)
                    if path.is_file() and path.name not in DERIVED and not path.name.endswith(".tmp"):
                        files[path.relative_to(directory).as_posix()] = sha(path)
                answer[tid] = files
            else:
                wave = entry.get("wave_id") or next(v["wave_id"] for v in main["waves"] if entry["map_id"] in v["map_ids"])
                receipt = shards[wave+"/"+entry["map_id"]]
                if receipt["archive"] not in checked_shards:
                    if sha(self.w/receipt["archive"]) != receipt["sha256"]:
                        raise ValueError("original immutable shard changed")
                    checked_shards.add(receipt["archive"])
                prefix = tid+"/"
                answer[tid] = {name[len(prefix):]:r["sha256"] for name,r in receipt["inventory"].items()
                    if name.startswith(prefix) and Path(name).name not in DERIVED and not name.endswith(".tmp")}
            if not answer[tid] or "request.json" not in answer[tid]:
                raise ValueError("original attempt raw is missing: "+tid)
        return answer

    def journal_events(self):
        rows = []
        for path in sorted((self.w/"collection").glob("main*/start_journal.jsonl")):
            self.owned(path, self.w)
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        ids = [r["trial_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("trial has more than one launch attempt")
        return rows

    def capture_origin(self):
        path = self.policy_dir/"origin.json"
        if path.exists():
            origin = read(path)
            if origin["manifest_sha256"] != self.manifest_sha:
                raise ValueError("R2 origin belongs to a different bound patch")
            return origin
        index = self.w/"exports/checkpoint_index.json"
        if sha(index) != self.m["source_checkpoint_index_sha256"]:
            raise ValueError("first R2 preparation requires the exact latest R1 checkpoint")
        checkpoint = read(index)
        if checkpoint["inventory_sha256"] != self.m["source_checkpoint_inventory_sha256"] or checkpoint["payloads"] != self.m["base_payloads"]:
            raise ValueError("R2 baseline checkpoint inventory/chain differs")
        manifest = self.main()
        started = [e for e in manifest["entries"] if e["execution_status"] != "PLANNED"]
        pending = sorted(e["trial_id"] for e in manifest["entries"] if e["execution_status"] == "PLANNED")
        started_ids = sorted(e["trial_id"] for e in started)
        missing = sorted(tid for tid in started_ids if not (self.w/"verdicts/main"/(tid+".json")).exists())
        for stem, values in (("started", started_ids), ("pending", pending), ("missing", missing)):
            if len(values) != self.m["initial_"+stem+"_count"] or digest(values) != self.m["initial_"+stem+"_ids_sha256"]:
                raise ValueError("baseline "+stem+" slots differ from the locked checkpoint")
        if sorted(r["trial_id"] for r in self.journal_events()) != started_ids:
            raise ValueError("baseline journals disagree with started attempts")
        self.copy(index, self.policy_dir/"baseline_checkpoint_index.json", self.m["source_checkpoint_index_sha256"])
        for name in ("manifests/main.json", "shards/index.json"):
            expected = checkpoint["inventory"]["campaign/"+name]["sha256"]
            self.copy(self.w/name, self.policy_dir/"baseline_assets"/name, expected)
        journals = {}
        for source in sorted((self.w/"collection").glob("main*/start_journal.jsonl")):
            relative = source.relative_to(self.w).as_posix()
            payload = source.read_bytes()
            target = self.policy_dir/"original_journals"/relative
            self.copy(source, target, sha(source))
            journals[relative] = dict(size_bytes=len(payload), sha256=sha(source))
        ledgers = {tid:sha(self.w/"verdicts/main"/(tid+".json")) for tid in started_ids if tid not in missing}
        # Only location fields may change when the existing wave is moved/packed.
        origin = dict(policy_id=POLICY_ID, manifest_sha256=self.manifest_sha,
            original_entries={e["trial_id"]:entry_state(e) for e in started}, initial_started_ids=started_ids,
            initial_pending_ids=pending, initial_missing_ids=missing, raw_inventory=self.raw_inventory(started),
            journal_prefixes=journals, original_ledger_hashes=ledgers,
            r1_validation_identity=self.r1.validation_identity)
        self.dump(path, origin)
        return origin

    def verify_baseline_seal(self, origin):
        baseline = read(self.policy_dir/"baseline_checkpoint_index.json")
        assets = {}
        for name in ("manifests/main.json", "shards/index.json"):
            path = self.policy_dir/"baseline_assets"/name
            if sha(path) != baseline["inventory"]["campaign/"+name]["sha256"]:
                raise ValueError("baseline proof asset changed: "+name)
            assets[name] = read(path)
        initial = {e["trial_id"]:e for e in assets["manifests/main.json"]["entries"]}
        if origin["original_entries"] != {tid:entry_state(initial[tid]) for tid in origin["initial_started_ids"]}:
            raise ValueError("origin attempt states differ from the authenticated baseline")
        expected_raw = {}
        shards = assets["shards/index.json"]["entries"]
        for tid in origin["initial_started_ids"]:
            entry = initial[tid]
            wave = entry.get("wave_id") or next(v["wave_id"] for v in assets["manifests/main.json"]["waves"] if entry["map_id"] in v["map_ids"])
            directory = self.declared_directory(entry, assets["manifests/main.json"])
            prefix = "campaign/"+directory.relative_to(self.w).as_posix()+"/"
            raw = {name[len(prefix):]:r["sha256"] for name,r in baseline["inventory"].items()
                   if name.startswith(prefix) and Path(name).name not in DERIVED and not name.endswith(".tmp")}
            if not raw:
                receipt = shards[wave+"/"+entry["map_id"]]
                if baseline["inventory"]["campaign/"+receipt["archive"]]["sha256"] != receipt["sha256"]:
                    raise ValueError("baseline shard receipt differs from checkpoint")
                prefix = tid+"/"
                raw = {name[len(prefix):]:r["sha256"] for name,r in receipt["inventory"].items()
                       if name.startswith(prefix) and Path(name).name not in DERIVED and not name.endswith(".tmp")}
            expected_raw[tid] = raw
        if expected_raw != origin["raw_inventory"]:
            raise ValueError("origin raw seal differs from the authenticated checkpoint")
        expected_journals = {name[len("campaign/"):] for name in baseline["inventory"]
                             if name.startswith("campaign/collection/main") and name.endswith("/start_journal.jsonl")}
        if set(origin["journal_prefixes"]) != expected_journals:
            raise ValueError("origin journal seal coverage changed")
        expected_ledgers = set(origin["initial_started_ids"])-set(origin["initial_missing_ids"])
        if set(origin["original_ledger_hashes"]) != expected_ledgers:
            raise ValueError("origin verdict seal coverage changed")
        for name, record in origin["journal_prefixes"].items():
            indexed = baseline["inventory"]["campaign/"+name]
            if record != {k:indexed[k] for k in ("size_bytes", "sha256")}:
                raise ValueError("origin journal seal differs from the checkpoint")
        for tid, expected in origin["original_ledger_hashes"].items():
            if baseline["inventory"]["campaign/verdicts/main/"+tid+".json"]["sha256"] != expected:
                raise ValueError("origin verdict seal differs from the checkpoint")

    def protect_origin(self, origin):
        self.guard()
        if sha(self.policy_dir/"baseline_checkpoint_index.json") != self.m["source_checkpoint_index_sha256"]:
            raise ValueError("sealed baseline checkpoint changed")
        self.verify_baseline_seal(origin)
        for stem in ("started", "pending", "missing"):
            values = origin["initial_"+stem+"_ids"]
            if len(values) != self.m["initial_"+stem+"_count"] or digest(sorted(values)) != self.m["initial_"+stem+"_ids_sha256"]:
                raise ValueError("sealed eligible/preserved IDs changed")
        main = self.main()
        entries = {e["trial_id"]:e for e in main["entries"]}
        if set(entries) != set(origin["initial_started_ids"]) | set(origin["initial_pending_ids"]):
            raise ValueError("campaign lost or added slots")
        old = [entries[tid] for tid in origin["initial_started_ids"]]
        if any(entry_state(e) != origin["original_entries"][e["trial_id"]] for e in old):
            raise ValueError("an original attempt was relabeled or altered")
        if self.raw_inventory(old) != origin["raw_inventory"]:
            raise ValueError("original attempt raw bytes changed")
        for relative, record in origin["journal_prefixes"].items():
            path = self.w/relative
            with path.open("rb") as stream:
                prefix = stream.read(record["size_bytes"])
            if len(prefix) != record["size_bytes"] or hashlib.sha256(prefix).hexdigest() != record["sha256"]:
                raise ValueError("original start journal prefix changed")
        rows = self.journal_events()
        ids = {row["trial_id"] for row in rows}
        if not set(origin["initial_started_ids"]).issubset(ids) or not ids.issubset(entries):
            raise ValueError("start journal lost or added attempts")
        for row in rows:
            if row["trial_id"] not in origin["initial_started_ids"]:
                if row.get("effective_controller_patch_sha256") != self.m["scheduler"]["sha256"]:
                    raise ValueError("new launch lacks the bound R2 scheduler provenance")
        for entry in main["entries"]:
            if entry["execution_status"] == "PLANNED" and entry["trial_id"] in ids:
                # A drained child manifest may be ahead of its parent checkpoint.
                # The scheduler journal, not this stale parent label, prevents retry.
                continue
            if entry["execution_status"] != "PLANNED" and entry["trial_id"] not in ids:
                raise ValueError("executed slot has no original start journal")
        for tid, expected in origin["original_ledger_hashes"].items():
            if sha(self.w/"verdicts/main"/(tid+".json")) != expected:
                raise ValueError("an original validated verdict changed")

    def verify_ledger(self, tid):
        path = self.w/"verdicts/main"/(tid+".json")
        ledger = read(path)
        row, receipt = ledger["row"], ledger["receipt"]
        actual = hashlib.sha256((json.dumps(row, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()).hexdigest()
        if (actual != ledger["verdict_sha256"] or actual != receipt["verdict_sha256"] or
                row.get("validation_code_identity") != self.r1.validation_identity or
                receipt.get("validation_code_identity") != self.r1.validation_identity or row.get("trial_id") != tid):
            raise ValueError("verdict identity/hash differs: "+tid)
        if row.get("errors") or row.get("correctness_errors"):
            raise ValueError("independent validation found correctness errors: "+tid)
        return row

    def verify_test_gates(self):
        if self.testing:
            return
        gate = read(self.policy_dir/"regression_gates.json")
        if gate.get("manifest_sha256") != self.manifest_sha:
            raise ValueError("regression gates belong to a different R2 patch")
        for stem, key in (("r2_affinity_regression", "test_scheduler"), ("r2_preservation_regression", "test_runner")):
            target = self.patch_dir/self.m[key]["path"]
            proof = gate["receipts"][stem]
            path = self.w/proof["path"]
            self.owned(path, self.w/"logs/bootstrap")
            if sha(path) != proof["sha256"]:
                raise ValueError("regression receipt changed")
            record = read(path)
            argv = record.get("argv", [])
            expected = {"--root":self.root}
            if key == "test_scheduler":
                expected["--scheduler-path"] = self.patch_dir/self.m["scheduler"]["path"]
            else:
                expected.update({"--runner-path":self.patch_dir/self.m["runner"]["path"], "--patch-dir":self.patch_dir})
            valid = (record.get("exit_code") == 0 and len(argv) > 1 and Path(argv[1]).resolve() == target and
                     Path(argv[0]).resolve() == Path(self.r1.prepared["controller_python"]).resolve())
            for flag, expected_path in expected.items():
                valid = valid and argv.count(flag) == 1 and argv.index(flag)+1 < len(argv)
                if valid:
                    valid = Path(argv[argv.index(flag)+1]).resolve() == expected_path
            if not valid:
                raise ValueError("current external regression gate is required: "+stem)

    def checkpoint(self, stage):
        if self.testing:
            return
        self.dump(self.policy_dir/"checkpoint_stage.json", dict(stage=stage, manifest_sha256=self.manifest_sha))
        bootstrap = importlib.import_module("kaggle_native_paper_v4")
        bootstrap.retain_checkpoint(self.w, self.root.parent/"source_input.zip")

    def prepare(self):
        self.guard()
        r1_gate = self.r1.prepare()
        origin = self.capture_origin()
        self.protect_origin(origin)
        entries = {e["trial_id"]:e for e in self.main()["entries"]}
        maps = {m["map_id"]:dict(m, mode="main") for m in read(self.w/"maps.json")["main"]}
        targets = [dict(entries[tid], directory=str(self.directory(entries[tid]))) for tid in origin["initial_missing_ids"]]
        with self.r1.analysis_module(self.patch_dir/self.m["analyzer"]["path"], patched=True) as analysis:
            for mid in sorted({e["map_id"] for e in targets}):
                subset = [e for e in targets if e["map_id"] == mid]
                # A previous R2 session may already have packed this wave.
                if all((self.w/"verdicts/main"/(e["trial_id"]+".json")).exists() for e in subset) and not all(Path(e["directory"]).is_dir() for e in subset):
                    continue
                analysis.validate_map(self.w, maps[mid], subset, 1)
        rows = [self.verify_ledger(tid) for tid in origin["initial_started_ids"]]
        self.protect_origin(origin)
        missing_rows = [row for row in rows if row["trial_id"] in origin["initial_missing_ids"]]
        gate = dict(policy_id=POLICY_ID, status="PASS", manifest_sha256=self.manifest_sha,
            r1_validation_identity=self.r1.validation_identity,
            preserved_attempts=len(rows), newly_validated=len(missing_rows),
            validation_snapshots=sum(len(row.get("events", [])) for row in missing_rows),
            validation_failures_retained=sum(row["execution_status"] != "COMPLETED" for row in missing_rows),
            ledger_hashes={tid:sha(self.w/"verdicts/main"/(tid+".json")) for tid in origin["initial_started_ids"]},
            pending_count=sum(e["execution_status"] == "PLANNED" for e in entries.values()),
            eligible_ids_sha256=digest(origin["initial_pending_ids"]))
        self.dump(self.policy_dir/"preparation.json", gate)
        return origin, r1_gate, gate

    @contextmanager
    def scheduler_overlay(self, origin, r1_gate):
        original = importlib.import_module("terrain_compare_scheduler")
        worker = importlib.import_module("native_paper_v4_worker")
        old_loop, old_collect = original._run_locked, self.campaign.run_campaign
        old_append, old_compact, old_require = self.fixed._append_journal, worker.compact_trial, self.campaign.require_smoke
        preserved = set(origin["initial_started_ids"])
        eligible = set(origin["initial_pending_ids"])
        baseline_smokes = self.r1.baseline_smokes()

        def append(path, event):
            if event["trial_id"] in preserved or event["trial_id"] not in eligible:
                raise ValueError("refusing to launch a preserved or unknown trial")
            return old_append(path, dict(event, effective_controller_patch_sha256=self.m["scheduler"]["sha256"],
                effective_controller_policy_id=POLICY_ID))

        def compact(directory, request, shared):
            if Path(directory).name in preserved:
                return dict(status="PRESERVED_ORIGINAL_ATTEMPT", policy_id=POLICY_ID)
            return old_compact(directory, request, shared)

        def require(workspace, mode):
            if Path(workspace).resolve() != self.w or mode not in ("smoke-serial", "smoke-parallel"):
                raise ValueError("wrong recovery smoke scope")
            for files in baseline_smokes.values():
                self.r1._verify_files(files)
            self.r1.verify_smoke_gate(r1_gate, baseline_smokes)
            return read(self.w/"manifests"/(mode+".json"))

        def collect(plan, registry, output, **kwargs):
            if not kwargs.get("resume"):
                # New waves are allowed; their frozen plans must contain only eligible IDs.
                planned = read(plan)["entries"]
                if any(e["trial_id"] not in eligible for e in planned):
                    raise ValueError("new wave contains an ineligible trial")
            result = original.run_campaign(plan, registry, output, **kwargs)
            result.setdefault("effective_controller_overlays", []).append(dict(policy_id=POLICY_ID,
                base_scheduler_sha256=self.m["original_scheduler_sha256"], scheduler_sha256=self.m["scheduler"]["sha256"]))
            if result.get("sessions"):
                result["sessions"][-1]["effective_controller_patch_sha256"] = self.m["scheduler"]["sha256"]
            original.atomic_json(Path(output)/"manifest.json", result)
            self.checkpoint("collection_barrier_"+Path(output).name)
            return result

        original._run_locked = self.fixed._run_locked
        self.fixed._append_journal = append
        worker.compact_trial = compact
        self.campaign.run_campaign = collect
        self.campaign.require_smoke = require
        try:
            yield
        finally:
            original._run_locked, self.campaign.run_campaign = old_loop, old_collect
            self.fixed._append_journal, worker.compact_trial, self.campaign.require_smoke = old_append, old_compact, old_require

    def run(self, *, allow_main=False, prepare_only=False, session_started_monotonic=None):
        origin, r1_gate, gate = self.prepare()
        if prepare_only:
            return dict(status="RECOVERY_PREPARED", **{k:v for k,v in gate.items() if k != "status"})
        if not allow_main or not sys.platform.startswith("linux"):
            raise ValueError("R2 main requires --allow-main on Linux")
        if multiprocessing.get_context().get_start_method() != "fork":
            raise ValueError("R2 requires the pinned Linux fork context")
        self.verify_test_gates()
        self.checkpoint("prepared_893_before_new_launches")
        started = session_started_monotonic or time.monotonic()
        try:
            with self.r1.analysis_module(self.patch_dir/self.m["analyzer"]["path"], patched=True) as analysis:
                with self.scheduler_overlay(origin, r1_gate):
                    result = self.campaign.run(self.w, "main", allow_main=True, resume=True,
                        session_elapsed_s=max(0., time.monotonic()-started))
                report = analysis.analyze_campaign(self.w, self.w/"reports/main", mode="main")
        finally:
            self.protect_origin(origin)
        receipt = dict(policy_id=POLICY_ID, status=result["status"], preserved_attempts=len(origin["initial_started_ids"]),
            permitted_new_attempts=len(origin["initial_pending_ids"]), report_status=report["completion"],
            original_raw_unchanged=True, original_statuses_unchanged=True,
            effective_scheduler_sha256=self.m["scheduler"]["sha256"])
        self.dump(self.policy_dir/"recovery_receipt.json", receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "patch-dir", "manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--session-started-monotonic", type=float)
    args = parser.parse_args(argv)
    runner = RecoveryV2(args.root, args.workspace, args.patch_dir, args.manifest, args.manifest_sha256)
    result = runner.run(allow_main=args.allow_main, prepare_only=args.prepare_only,
        session_started_monotonic=args.session_started_monotonic)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("COMPLETE", "RECOVERY_PREPARED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
