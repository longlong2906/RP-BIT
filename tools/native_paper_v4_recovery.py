"""External, hash-bound verifier recovery for an unchanged frozen v4 campaign.

This file and the corrected analyzer must live outside the frozen repository.
The original planner, requests, inputs, start journals and source stay intact.
Only derived validation/state and new map shards may be written by recovery.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import zipfile

POLICY_ID = "v4_endpoint_roundoff_recovery_v1"
SMOKES = ("smoke-serial", "smoke-parallel")
DERIVED = {"validation.json", "validation_receipt.json"}
HELPERS = ("independent_verifier.py", "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py",
           "native_paper_v3_analysis.py", "native_paper_analysis.py", "a3_traversability_protocol.py")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _regular_destination(path):
    path = Path(path)
    for current in (path, *path.parents):
        if current.is_symlink() or getattr(current,"is_junction",lambda:False)():
            raise ValueError("recovery write path contains a link")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValueError("recovery write path is not a regular file")


@contextmanager
def _pending_stream(path, *, binary):
    _regular_destination(path)
    flags = os.O_WRONLY|os.O_CREAT|os.O_TRUNC|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    descriptor = os.open(path,flags,0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("recovery temporary is not a regular file")
        stream = os.fdopen(descriptor,"wb" if binary else "w", **({} if binary else dict(encoding="utf-8",newline="\n")))
        descriptor = None
        with stream:
            yield stream
    finally:
        if descriptor is not None: os.close(descriptor)


def dump(path, value):
    path = Path(path);_regular_destination(path)
    temporary = path.with_name(path.name+".pending")
    _regular_destination(temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _pending_stream(temporary,binary=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, path)


def _safe(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or str(PurePosixPath(name)) != name:
        raise ValueError("unsafe recovery artifact path")
    return name


def _owned(path, base):
    path, base = Path(path).absolute(), Path(base).resolve()
    if ".." in path.parts or not path.is_relative_to(base) or not path.resolve().is_relative_to(base):
        raise ValueError("recovery path escapes its owned root")
    for current in (path, *path.parents):
        if current == base: break
        if current.is_symlink() or getattr(current,"is_junction",lambda:False)():
            raise ValueError("recovery path contains a link")
    return path


def _copy(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or sha(source) != expected:
        raise ValueError("recovery source hash mismatch")
    _regular_destination(destination)
    if destination.exists():
        if destination.is_symlink() or sha(destination) != expected:
            raise ValueError("refusing to replace unrelated recovery bytes")
        return
    temporary = destination.with_name(destination.name+".copy-pending")
    _regular_destination(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A partial copy is not scientific data. Replace only after the complete
    # source has been checked; the source itself is never removed.
    with source.open("rb") as original, _pending_stream(temporary,binary=True) as output:
        shutil.copyfileobj(original, output, 1024*1024)
    if sha(temporary) != expected:
        raise ValueError("recovery copy hash mismatch")
    os.replace(temporary, destination)


class RecoveryRunner:
    def __init__(self, root, workspace, analyzer_patch, manifest, manifest_sha256, *, analyzer_sha256=None, _campaign=None):
        self.root, self.workspace = Path(root).resolve(), Path(workspace).resolve()
        self.patch, self.manifest_path = Path(analyzer_patch).resolve(), Path(manifest).resolve()
        if self.patch.is_relative_to(self.root) or Path(__file__).resolve().is_relative_to(self.root):
            raise ValueError("recovery code must remain outside the frozen repository")
        if sha(self.manifest_path) != manifest_sha256:
            raise ValueError("recovery manifest SHA-256 mismatch")
        self.manifest = read(self.manifest_path)
        if self.manifest.get("policy_id") != POLICY_ID:
            raise ValueError("unknown recovery policy")
        for key, path in (("analyzer", self.patch), ("runner", Path(__file__))):
            record = self.manifest[key]
            if path.name != record["path"] or sha(path) != record["sha256"]:
                raise ValueError("bound recovery "+key+" changed")
        if self.patch.parent != self.manifest_path.parent:
            raise ValueError("analyzer must be beside its bound manifest")
        if analyzer_sha256 is not None and analyzer_sha256 != self.manifest["analyzer"]["sha256"]:
            raise ValueError("analyzer CLI SHA differs from bound manifest")
        self.policy_dir = self.workspace/"recovery"/POLICY_ID
        _owned(self.policy_dir,self.workspace)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.stage = self.policy_dir/"smoke_stage"
        if str(self.root/"tools") not in sys.path:
            sys.path.insert(0, str(self.root/"tools"))
        self.campaign = _campaign or importlib.import_module("native_paper_v4_campaign")
        if _campaign is None and Path(self.campaign.__file__).resolve() != self.root/"tools/native_paper_v4_campaign.py":
            raise ValueError("shadowed frozen campaign module")
        self.prepared = self.campaign.check_frozen(self.workspace)
        if self.prepared["identity"]["source_sha256"] != self.manifest["prepared_source_sha256"]:
            raise ValueError("wrong frozen campaign source identity")
        self.source_input = self.root.parent/"source_input.zip"
        if sha(self.source_input) != self.manifest["source_input_sha256"]:
            raise ValueError("wrong original source package")
        self.validation_identity = digest(dict(policy_id=POLICY_ID,
            analyzer_sha256=sha(self.patch), runner_sha256=sha(__file__),
            original_analyzer_sha256=sha(self.root/"tools/native_paper_v4_analysis.py"),
            frozen_helpers={name:sha(self.root/"tools"/name) for name in HELPERS},
            frozen_source_identity=self.prepared["identity"]["source_sha256"]))

    def guard(self):
        self.campaign.check_frozen(self.workspace)
        if sha(self.patch) != self.manifest["analyzer"]["sha256"] or sha(__file__) != self.manifest["runner"]["sha256"]:
            raise ValueError("recovery implementation changed during campaign")
        if sha(self.source_input) != self.manifest["source_input_sha256"]:
            raise ValueError("source package changed during recovery")

    @contextmanager
    def analysis_module(self, path, *, patched=False):
        name = "native_paper_v4_analysis"
        old = sys.modules.get(name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            module.ROOT = self.root
            if patched:
                module.validation_code_identity = lambda:self.validation_identity
            yield module
        finally:
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

    def baseline_smokes(self):
        with self.analysis_module(self.root/"tools/native_paper_v4_analysis.py"):
            for mode in SMOKES: self.campaign.require_smoke(self.workspace, mode)
        result = {}
        for mode in SMOKES:
            readiness = read(self.workspace/"readiness"/(mode+".json"))
            files = {"manifests/"+mode+".json":sha(self.workspace/"manifests"/(mode+".json")),
                     "readiness/"+mode+".json":sha(self.workspace/"readiness"/(mode+".json"))}
            files.update({"verdicts/"+mode+"/"+tid+".json":value for tid,value in readiness["ledgers"].items()})
            files.update(readiness["shards"])
            result[mode] = files
        return result

    def _verify_files(self, records):
        for name, expected in records.items():
            path = self.workspace/_safe(name)
            _owned(path,self.workspace)
            if path.is_symlink() or sha(path) != expected:
                raise ValueError("protected recovery artifact changed: "+name)

    def main_raw_inventory(self, only_ids=None):
        """Logical pre-existing trial files, whether active or already packed."""
        main = read(self.workspace/"manifests/main.json")
        index_path = self.workspace/"shards/index.json"
        index = read(index_path)["entries"] if index_path.exists() else {}
        result = {}
        for entry in main["entries"]:
            if entry["execution_status"] == "PLANNED": continue
            tid = entry["trial_id"]
            if only_ids is not None and tid not in only_ids: continue
            wave_id = entry.get("wave_id") or next(wave["wave_id"] for wave in main["waves"] if entry["map_id"] in wave["map_ids"])
            directory = Path(entry["directory"]) if entry.get("directory") else self.workspace/"collection"/wave_id/tid
            if not directory.resolve().is_relative_to(self.workspace):
                raise ValueError("raw trial directory escapes the frozen workspace")
            if directory.is_dir():
                result[tid] = {p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob("*"))
                               if p.is_file() and p.name not in DERIVED and not p.name.endswith(".tmp")}
            else:
                key = wave_id+"/"+entry["map_id"]
                receipt = index[key]
                if sha(self.workspace/receipt["archive"]) != receipt["sha256"]:
                    raise ValueError("pre-existing raw shard changed")
                prefix = tid+"/"
                result[tid] = {name[len(prefix):]:record["sha256"] for name,record in receipt["inventory"].items()
                               if name.startswith(prefix) and Path(name).name not in DERIVED and not name.endswith(".tmp")}
        return result

    def capture_origin(self):
        target = self.policy_dir/"origin.json"
        if target.exists():
            value = read(target)
            if value["policy_id"] != POLICY_ID or value["validation_identity"] != self.validation_identity:
                raise ValueError("existing recovery policy identity differs")
            if (len(value["original_started_ids"]) != self.manifest["initial_started_count"] or
                    digest(sorted(value["original_started_ids"])) != self.manifest["initial_started_ids_sha256"] or
                    set(value["original_raw_inventory"]) != set(value["original_started_ids"])):
                raise ValueError("recovery origin attempted-slot identities changed")
            return value
        raw = self.main_raw_inventory()
        if len(raw) != self.manifest["initial_started_count"] or digest(sorted(raw)) != self.manifest["initial_started_ids_sha256"]:
            raise ValueError("unexpected initial attempted slots; do not retry or relabel them")
        journals = {p.relative_to(self.workspace).as_posix():sha(p) for p in
                    (self.workspace/"collection").glob("main*/start_journal.jsonl")}
        baseline = self.baseline_smokes()
        value = dict(policy_id=POLICY_ID, validation_identity=self.validation_identity,
                     original_started_ids=sorted(raw), original_raw_inventory=raw,
                     original_journals=journals, baseline_smokes=baseline,
                     original_main_manifest_sha256=sha(self.workspace/"manifests/main.json"))
        dump(target, value)
        return value

    def backup_derived(self):
        manifest_path = self.policy_dir/"backup_manifest.json"
        if manifest_path.exists(): records = read(manifest_path)["files"]
        else:
            candidates = {self.workspace/"manifests/main.json"}
            candidates.update((self.workspace/"verdicts/main").glob("*.json"))
            candidates.update(p for p in (self.workspace/"active").rglob("*") if p.is_file() and p.name in DERIVED)
            records = {p.relative_to(self.workspace).as_posix():sha(p) for p in sorted(candidates)}
            dump(manifest_path, dict(files=records, policy_id=POLICY_ID))
        for name, expected in records.items():
            source = self.workspace/_safe(name)
            backup = self.policy_dir/"original_derived"/name
            _owned(backup,self.policy_dir)
            if backup.exists():
                if sha(backup) != expected: raise ValueError("original-derived backup was changed")
            else: _copy(source, backup, expected)
        return records

    def protect_origin(self, origin):
        self.guard()
        for records in origin["baseline_smokes"].values(): self._verify_files(records)
        self._verify_files(origin["original_journals"])
        current = self.main_raw_inventory(set(origin["original_started_ids"]))
        for tid, expected in origin["original_raw_inventory"].items():
            if current.get(tid) != expected:
                raise ValueError("pre-existing attempted raw bytes changed: "+tid)
        starts = []
        for journal in (self.workspace/"collection").glob("main*/start_journal.jsonl"):
            starts.extend(read_json_line["trial_id"] for line in journal.read_text().splitlines()
                          if line.strip() for read_json_line in [json.loads(line)])
        if len(starts) != len(set(starts)) or not set(origin["original_started_ids"]).issubset(starts):
            raise ValueError("start journal lost or repeated an attempted slot")

    def _stage_map(self, mode, wave_id, map_id, receipt):
        destination = self.stage/"active"/wave_id/map_id
        _owned(destination,self.policy_dir)
        archive_path = self.workspace/receipt["archive"]
        if sha(archive_path) != receipt["sha256"]: raise ValueError("original smoke shard changed")
        with zipfile.ZipFile(archive_path) as archive:
            for name, record in receipt["inventory"].items():
                target = destination/_safe(name)
                _owned(target,self.policy_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.name in DERIVED: continue  # Recovery validation cache is derived.
                    if target.is_symlink() or sha(target) != record["sha256"]: raise ValueError("smoke-stage raw changed")
                else:
                    payload = archive.read(name)
                    if hashlib.sha256(payload).hexdigest() != record["sha256"]: raise ValueError("smoke shard member changed")
                    target.write_bytes(payload)
            if archive.testzip() is not None: raise ValueError("smoke shard CRC mismatch")
        for reference in destination.glob("*/artifact_refs.json"):
            value = read(reference); trial = reference.parent
            original_trial = self.workspace/"active"/wave_id/map_id/trial.name
            source = (original_trial/value["native_raster_path"]).resolve()
            target = (trial/value["native_raster_path"]).resolve()
            if not source.is_relative_to(self.workspace/"shared") or not target.is_relative_to(self.stage/"shared"):
                raise ValueError("smoke raster reference escapes the staged shared roots")
            _owned(source,self.workspace);_owned(target,self.policy_dir)
            _copy(source, target, value["native_raster_sha256"])
        return destination

    def verify_smoke_gate(self, gate, baseline):
        if gate.get("status") != "PASS" or gate.get("policy_id") != POLICY_ID or gate.get("validation_identity") != self.validation_identity or gate.get("baseline_smokes") != baseline:
            raise ValueError("stale recovery smoke gate")
        expected = {}
        for mode in SMOKES:
            entries = read(self.workspace/"manifests"/(mode+".json"))["entries"]
            if len(entries) != 21: raise ValueError("smoke gate must cover 21 trials per mode")
            for entry in entries:
                path = self.stage/"verdicts"/mode/(entry["trial_id"]+".json")
                expected[path.relative_to(self.workspace).as_posix()] = (mode,entry)
        if set(gate.get("ledger_files",{})) != set(expected):
            raise ValueError("recovery smoke gate has missing or extra trial ledgers")
        self._verify_files(gate["ledger_files"])
        measured = {mode:dict(trials=0,success=0) for mode in SMOKES}
        for name,(mode,entry) in expected.items():
            ledger = read(self.workspace/name);row=ledger["row"]
            actual = hashlib.sha256((json.dumps(row,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()).hexdigest()
            if (actual != ledger.get("verdict_sha256") or actual != ledger["receipt"].get("verdict_sha256") or
                    row.get("validation_code_identity") != self.validation_identity or
                    ledger["receipt"].get("validation_code_identity") != self.validation_identity or
                    row.get("trial_id") != entry["trial_id"] or row.get("backend") != entry["backend"] or
                    row.get("errors") or row.get("correctness_errors") or
                    row.get("planner_status") in ("ERROR","MODEL_ERROR","BLOCKED_DEPENDENCY")):
                raise ValueError("recovery smoke verdict identity/hash/correctness mismatch")
            measured[mode]["trials"]+=1;measured[mode]["success"]+=int(bool(row.get("success")))
        if measured != gate.get("counts"): raise ValueError("recovery smoke coverage/counts changed")

    def prepare(self):
        self.guard()
        baseline = self.baseline_smokes()
        origin = self.capture_origin()
        if origin["baseline_smokes"] != baseline: raise ValueError("original smoke readiness changed")
        self.protect_origin(origin)
        self.backup_derived()
        gate_path = self.policy_dir/"smoke_gate.json"
        if gate_path.exists():
            gate = read(gate_path)
            self.verify_smoke_gate(gate,baseline)
            return gate
        shard_index = read(self.workspace/"shards/index.json")["entries"]
        maps = {item["map_id"]:item for item in read(self.workspace/"maps.json")["smoke"]}
        ledger_files, counts = {}, {}
        with self.analysis_module(self.patch, patched=True) as analysis:
            for mode in SMOKES:
                manifest = read(self.workspace/"manifests"/(mode+".json"))
                grouped = {}
                for entry in manifest["entries"]: grouped.setdefault((entry["wave_id"],entry["map_id"]), []).append(dict(entry))
                rows = []
                for (wave_id,map_id), entries in grouped.items():
                    directory = self._stage_map(mode,wave_id,map_id,shard_index[wave_id+"/"+map_id])
                    for entry in entries: entry["directory"] = str(directory/entry["trial_id"])
                    info = dict(maps[map_id], mode=mode)
                    rows.extend(analysis.validate_map(self.stage,info,entries,1))
                if len(rows) != 21 or any(row.get("errors") or row.get("correctness_errors") or
                        row.get("planner_status") in ("ERROR","MODEL_ERROR","BLOCKED_DEPENDENCY") for row in rows):
                    raise ValueError("corrected smoke verification did not pass")
                counts[mode] = dict(trials=len(rows), success=sum(bool(r.get("success")) for r in rows))
                for entry in manifest["entries"]:
                    path = self.stage/"verdicts"/mode/(entry["trial_id"]+".json")
                    ledger_files[path.relative_to(self.workspace).as_posix()] = sha(path)
        self.protect_origin(origin)
        gate = dict(policy_id=POLICY_ID, status="PASS", validation_identity=self.validation_identity,
                    baseline_smokes=baseline, ledger_files=ledger_files, counts=counts)
        self.verify_smoke_gate(gate,baseline)
        dump(gate_path,gate)
        return gate

    def run(self, *, allow_main=False, session_started_monotonic=None, prepare_only=False):
        gate = self.prepare()
        if prepare_only: return dict(status="RECOVERY_PREPARED", smoke_gate=gate)
        if not allow_main or not sys.platform.startswith("linux"):
            raise ValueError("resuming the remaining main slots requires --allow-main on Linux")
        import multiprocessing
        if multiprocessing.get_context().get_start_method() != "fork":
            raise ValueError("recovery requires the inherited verifier in the pinned Linux fork context")
        origin = read(self.policy_dir/"origin.json")
        original_require = self.campaign.require_smoke
        def require(workspace,mode):
            if Path(workspace).resolve() != self.workspace or mode not in SMOKES: raise ValueError("wrong recovery smoke gate")
            self._verify_files(origin["baseline_smokes"][mode])
            self.verify_smoke_gate(gate,origin["baseline_smokes"])
            return read(self.workspace/"manifests"/(mode+".json"))
        self.campaign.require_smoke = require
        try:
            with self.analysis_module(self.patch, patched=True) as analysis:
                result = self.campaign.run(self.workspace,"main",allow_main=True,resume=True,
                    session_elapsed_s=max(0.,time.monotonic()-(session_started_monotonic or time.monotonic())))
                report = analysis.analyze_campaign(self.workspace,self.workspace/"reports/main",mode="main")
        finally:
            self.campaign.require_smoke = original_require
            self.protect_origin(origin)
        receipt = dict(policy_id=POLICY_ID, validation_identity=self.validation_identity,
                       status=result["status"], original_attempted_slots_retained=len(origin["original_started_ids"]),
                       source_unchanged=True, original_raw_unchanged=True, report_status=report.get("completion",report.get("status")),
                       completed_utc=datetime.now(timezone.utc).isoformat())
        dump(self.policy_dir/"recovery_receipt.json",receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--analyzer-patch",type=Path,required=True);parser.add_argument("--analyzer-sha256",required=True)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--allow-main",action="store_true");parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--session-started-monotonic",type=float)
    args=parser.parse_args(argv)
    runner=RecoveryRunner(args.root,args.workspace,args.analyzer_patch,args.manifest,args.manifest_sha256,
                          analyzer_sha256=args.analyzer_sha256)
    result=runner.run(allow_main=args.allow_main,session_started_monotonic=args.session_started_monotonic,
                      prepare_only=args.prepare_only)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"] in ("RECOVERY_PREPARED","COMPLETE") else 2


if __name__=="__main__": raise SystemExit(main())
