"""Stream a completed Kaggle Output ZIP into a verified schema-12 checkpoint.

The original Output, source, requests, paths and native binaries are read-only.
No planner, build, generator, oracle or experiment worker is executed here.
Only incomplete exports and temporary correctness-workspace copies are omitted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile

import kaggle_a3_bundle as bundle

SOURCE_SHA256 = "1a5aca979a42189719ef28cac59145a22ef1cf6809b93e79dbdeaca236d59ffb"
SOURCE_IDENTITY = "1faff39bebe0cff1f8a0b8a7f8fbdaad527b3c98307213168f48814de9b9f511"
PREPARED_SHA256 = "73597a4e753ce03afed5686b024d740d178c9d9012150e51c1391e15f5a8b6f4"
ROOT = "/kaggle/working/native_paper_v3/repo"
WORKSPACE = "/kaggle/working/native_paper_v3/campaign"
PREFIX = "native_paper_v3/"
KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA12_CHECKPOINT"
SOURCE_KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA12_SOURCE"
PROTOCOL = "a3t_sem_putn_kaggle_v3"
RUNTIME_DIRS = ("build-native-paper-v3", ".deps/install", ".venv-native-paper-v3", ".managed-python")
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class _Policy:
    """A private labelled policy allows small fixtures, never a CLI bypass."""
    source_sha256: str = SOURCE_SHA256
    source_identity: str = SOURCE_IDENTITY
    prepared_sha256: str = PREPARED_SHA256
    source_count: int = 717
    native_count: int = 153
    trial_count: int = 3000
    synthetic_fixture: bool = False


PRODUCTION = _Policy()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_archive(archive):
    """Reject ambiguity and unsafe types before writing any destination."""
    files, types = {}, {}
    for info in archive.infolist():
        name = info.orig_filename
        require(name == info.filename, "normalized/truncated ZIP filename")
        plain = name[:-1] if info.is_dir() else name
        require(name and "\x00" not in name and "\\" not in name and ":" not in name, "unsafe ZIP path")
        path = PurePosixPath(plain)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == plain,
                "unsafe/noncanonical ZIP path: " + name)
        require(plain not in types, "duplicate ZIP path: " + name)
        mode = info.external_attr >> 16
        require(info.create_system == 3 and stat.S_IFMT(mode) == (stat.S_IFDIR if info.is_dir() else stat.S_IFREG),
                "ZIP entry is not a regular Unix file/directory: " + name)
        require(not info.flag_bits & 1, "encrypted ZIP entry refused")
        require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "unsupported compression")
        if not info.is_dir():
            require(stat.S_IMODE(mode) in (0o644, 0o755), "unsupported file mode: " + name)
            files[name] = info
        types[plain] = info.is_dir()
    for name in types:
        require(all(types.get(str(p), True) for p in PurePosixPath(name).parents),
                "file is ZIP path ancestor: " + name)
    return files


def selected_name(name):
    if not name.startswith(PREFIX):
        return None
    relative = name[len(PREFIX):]
    parts = PurePosixPath(relative).parts
    if relative == "source_input.zip":
        return relative
    if parts[0] == "campaign":
        # tmp contains only retained fixed-work correctness copies, not trials.
        return None if (len(parts) > 1 and parts[1] in ("exports", "tmp")) or parts[-1] == "controller.lock" else relative
    if parts[0] != "repo":
        return None
    local = relative[len("repo/"):]
    if not any(local.startswith(folder + "/") for folder in RUNTIME_DIRS):
        return None
    if PurePosixPath(local).suffix in (".o", ".obj", ".pyc") or "__pycache__" in parts:
        return None
    return relative


def json_member(archive, entries, name):
    require(name in entries, "missing metadata: " + name)
    require(entries[name].file_size <= 64 * 1024**2, "metadata exceeds bounded size: " + name)
    return json.loads(archive.read(entries[name]))


def archive_path(absolute):
    require(isinstance(absolute, str) and absolute.startswith(ROOT + "/"), "native path outside repository")
    return bundle.safe_name("repo/" + absolute[len(ROOT) + 1:])


def aliases(prepared):
    """Restore only the two alias relations proven by saved native linkage."""
    identity = prepared["identity"]
    roots, hashes = identity["native_linkage"]["roots"], identity["binary_hashes"]
    requested = prepared["controller_python"]
    require(requested == prepared["sem_python"] == ROOT + "/.venv-native-paper-v3/bin/python",
            "unexpected interpreter role/path")
    record = roots[requested]
    target = record["resolved_target"]
    require(record["requested_root"] == requested and target != requested, "missing captured interpreter alias")
    require(hashes[requested] == hashes[target], "captured interpreter alias hash differs")
    result = [dict(path=archive_path(requested), target=archive_path(target),
                   target_is_directory=False, target_sha256=hashes[target])]
    target = roots[ROOT + "/build-native-paper-v3/terrain_benchmark"]["libraries"]["libompl.so.18"]
    alias = str(PurePosixPath(target).with_name("libompl.so.18"))
    require(alias != target, "OMPL alias target is not canonical")
    result.append(dict(path=archive_path(alias), target=archive_path(target),
                       target_is_directory=False, target_sha256=hashes[target]))
    return sorted(result, key=lambda item: item["path"])


def hash_member(archive, info):
    h, size = hashlib.sha256(), 0
    with archive.open(info) as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
            size += len(block)
    require(size == info.file_size, "ZIP member size mismatch")
    return h.hexdigest()


def expected_members(prepared, source_manifest, policy):
    expected = {}
    def add(name, checksum):
        bundle.safe_name(name)
        require(name not in expected or expected[name] == checksum, "conflicting frozen member hash: " + name)
        expected[name] = checksum
    rows = prepared["identity"]["source_files"]
    source_records = {r["path"]: r for r in source_manifest["files"]}
    require(len(rows) == policy.source_count and len({r[0] for r in rows}) == len(rows), "frozen source count mismatch")
    require(bundle.digest(rows) == policy.source_identity == prepared["identity"]["source_sha256"], "frozen source identity mismatch")
    for name, checksum in rows:
        require(source_records.get(name, {}).get("sha256") == checksum, "frozen source differs from released source: " + name)
    for name, record in source_records.items():
        add("repo/" + name, record["sha256"])
    native_count = 0
    for name, checksum in prepared["identity"]["binary_hashes"].items():
        if name.startswith(ROOT + "/"):
            add(archive_path(name), checksum)
            native_count += 1
        else:
            require(name.startswith("/usr/lib/"), "unexpected external native path: " + name)
    require(native_count == policy.native_count, "frozen native file count mismatch")
    for name, checksum in prepared["frozen_inputs"].items():
        add("campaign/" + name, checksum)
    add("campaign/registry.json", prepared["registry_sha256"])
    add("campaign/prepared.json", policy.prepared_sha256)
    add("source_input.zip", policy.source_sha256)
    return expected


def load_scientific_inventory(path):
    result = {}
    if path is None:
        return result
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            destination = selected_name(item["member"])
            require(destination is not None and destination.startswith("campaign/"), "invalid scientific inventory selection")
            require(destination not in result, "duplicate scientific inventory member")
            result[destination] = item
    return result


def repack(input_path, output_path, *, expected_input_sha256, source_archive,
           scientific_inventory=None, _policy=PRODUCTION):
    started = time.monotonic()
    input_path, output_path, source_archive = map(Path, (input_path, output_path, source_archive))
    pending = output_path.with_name(output_path.name + ".partial")
    sidecar = output_path.with_name(output_path.name + ".sha256")
    receipt_path = output_path.with_name(output_path.name + ".recovery.json")
    inventory_path = output_path.with_name(output_path.name + ".scientific.jsonl")
    require(not any(p.exists() for p in (output_path, pending, sidecar, receipt_path, inventory_path)), "output/partial/sidecar exists")
    require(input_path.resolve() != output_path.resolve(), "input must not be overwritten")
    before = input_path.stat()
    input_hash = bundle.sha256(input_path)
    require(input_hash == expected_input_sha256, "input Output ZIP SHA-256 mismatch")
    source_manifest = bundle.verify(source_archive, _policy.source_sha256)
    require(source_manifest["kind"] == SOURCE_KIND, "not the released schema-12 source")
    verifier_record = next(r for r in source_manifest["files"] if r["path"] == "tools/kaggle_a3_bundle.py")
    require(bundle.sha256(Path(bundle.__file__)) == verifier_record["sha256"], "archive verifier differs from frozen source")
    scientific = load_scientific_inventory(scientific_inventory)
    inventory_input_sha = bundle.sha256(scientific_inventory) if scientific_inventory else None
    with zipfile.ZipFile(input_path) as original:
        entries = index_archive(original)
        prepared = json_member(original, entries, PREFIX + "campaign/prepared.json")
        require(prepared["source_root"] == ROOT and prepared["schema_version"] == 12 and
                prepared["protocol_id"] == PROTOCOL, "wrong prepared protocol/root")
        main = json_member(original, entries, PREFIX + "campaign/campaigns/main/manifest.json")
        attempts = main["entries"]
        require(main["status"] == "COLLECTED_VALIDATION_PENDING" and len(attempts) == _policy.trial_count and
                len({e["trial_id"] for e in attempts}) == _policy.trial_count and
                all(e["execution_status"] == "COMPLETED" for e in attempts), "all main attempts must already be completed")
        if not _policy.synthetic_fixture:
            require({b:sum(e["backend"] == b for e in attempts) for b in ("a3", "sem", "putn")} ==
                    dict(a3=1000, sem=1000, putn=1000), "unexpected backend cardinality")
        links = aliases(prepared)
        expected = expected_members(prepared, source_manifest, _policy)
        for link in links:
            for field in ("path", "target"):
                name = link[field]
                require(name not in expected or expected[name] == link["target_sha256"], "alias conflicts with frozen member")
                expected[name] = link["target_sha256"]
        selected = {destination:entries[name] for name in entries if (destination := selected_name(name)) is not None}
        require(all(PREFIX + name in entries for name in expected), "missing frozen source/native/input member")
        require(all(link[k] in selected for link in links for k in ("path", "target")), "missing runtime alias/target")
        for attempt in attempts:
            trial = "campaign/campaigns/main/" + attempt["trial_id"] + "/"
            for filename in ("worker.json", "request.json", "execution.json", "progress.jsonl", "launch.json", "phase.json"):
                require(trial + filename in selected, "missing completed trial member: " + trial + filename)
        for name, item in scientific.items():
            require(name in selected, "missing independently audited scientific member: " + name)
            info = selected[name]
            require(info.file_size == item["bytes"] and f"{info.CRC:08x}" == item["crc32"], "audited scientific size/CRC mismatch")
            require(name not in expected or expected[name] == item["sha256"], "scientific inventory conflicts with frozen member")
            expected[name] = item["sha256"]
        checked = {}
        for name, checksum in expected.items():
            if name not in selected:
                require(hash_member(original, entries[PREFIX + name]) == checksum, "omitted source checksum mismatch: " + name)
                checked[name] = checksum
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records, scientific_count, scientific_bytes = [], 0, 0
        with zipfile.ZipFile(pending, "x", allowZip64=True) as target, inventory_path.open("x", encoding="utf-8") as inventory:
            for number, (name, info) in enumerate(sorted(selected.items()), 1):
                mode = stat.S_IMODE(info.external_attr >> 16)
                destination = zipfile.ZipInfo(name, date_time=info.date_time)
                destination.create_system = 3
                destination.external_attr = (stat.S_IFREG | mode) << 16
                destination.compress_type = zipfile.ZIP_STORED if name.endswith((".zip", ".gz")) else zipfile.ZIP_DEFLATED
                destination._compresslevel = 1
                destination.file_size = info.file_size
                h, size = hashlib.sha256(), 0
                with original.open(info) as reader, target.open(destination, "w", force_zip64=True) as writer:
                    for block in iter(lambda: reader.read(CHUNK), b""):
                        h.update(block); size += len(block); writer.write(block)
                actual = h.hexdigest()
                require(size == info.file_size, "streamed size mismatch: " + name)
                require(name not in expected or expected[name] == actual, "frozen checksum mismatch: " + name)
                if name in expected: checked[name] = actual
                records.append(dict(path=name, size_bytes=size, sha256=actual, mode=mode))
                if name.startswith("campaign/"):
                    inventory.write(json.dumps(dict(member=PREFIX + name, bytes=size, crc32=f"{info.CRC:08x}", sha256=actual), sort_keys=True)+"\n")
                    scientific_count += 1; scientific_bytes += size
                if number % 10000 == 0: print(f"Streamed {number}/{len(selected)} members", flush=True)
            require(set(checked) == set(expected), "incomplete frozen member verification")
            metadata = dict(protocol_id=PROTOCOL, root_path=ROOT, workspace_path=WORKSPACE,
                requires_same_runtime_identity=False, runtime_policy="native_paper_sessions_v3", contains_exact_runtime=True,
                internal_runtime_symlinks=links, source_input_sha256=_policy.source_sha256,
                recovery_origin=dict(output_zip_sha256=input_hash, output_zip_size_bytes=before.st_size,
                    source_identity_sha256=_policy.source_identity, prepared_sha256=_policy.prepared_sha256,
                    completed_attempts=_policy.trial_count, verified_source_files=_policy.source_count,
                    verified_in_repo_native_files=_policy.native_count, verified_scientific_entries=scientific_count,
                    independently_audited_scientific_entries=len(scientific), scientific_inventory_sha256=inventory_input_sha,
                    requires_linux_check_frozen_before_validation=True, original_filesystem_symlinks_fully_reconstructed=False,
                    method="stream_original_bytes; reconstruct_only_two_native_linkage_aliases",
                    synthetic_fixture=_policy.synthetic_fixture))
            manifest = dict(schema_version=1, kind=KIND, files=records, files_sha256=bundle.digest(records),
                            metadata=metadata, path_case_policy=bundle.LINUX_CASE_POLICY)
            target.writestr("bundle.json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    after = input_path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "original Output changed during conversion")
    print("Verifying all checkpoint member hashes", flush=True)
    bundle.verify(pending)
    if os.name == "nt": pending.rename(output_path)
    else:
        os.link(pending, output_path); pending.unlink()
    output_hash = bundle.sha256(output_path)
    with sidecar.open("x", encoding="utf-8") as stream: stream.write(output_hash + "  " + output_path.name + "\n")
    report = dict(status="VERIFIED_RECOVERED_CHECKPOINT", input=str(input_path), input_sha256=input_hash,
        output=str(output_path), output_sha256=output_hash, output_size_bytes=output_path.stat().st_size,
        selected_files=len(records), uncompressed_size_bytes=sum(r["size_bytes"] for r in records),
        scientific_entries=scientific_count, scientific_uncompressed_bytes=scientific_bytes,
        scientific_inventory_sha256=bundle.sha256(inventory_path), independently_audited_entries=len(scientific),
        source_expanded_size_bytes=sum(r["size_bytes"] for r in source_manifest["files"]),
        source_archive_size_bytes=source_archive.stat().st_size,
        frozen_checks=len(checked), main_attempts=_policy.trial_count, trial_executions=0,
        linux_check_frozen="NOT_RUN", path_validation="NOT_RUN", synthetic_fixture=_policy.synthetic_fixture,
        finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-started)
    with receipt_path.open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2); stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--scientific-inventory", type=Path)
    args = parser.parse_args()
    print(json.dumps(repack(args.input, args.output, expected_input_sha256=args.expected_input_sha256,
        source_archive=args.source_archive, scientific_inventory=args.scientific_inventory), indent=2))


if __name__ == "__main__": main()
