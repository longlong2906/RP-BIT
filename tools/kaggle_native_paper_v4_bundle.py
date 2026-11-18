"""Schema-13 source bundle and hash-verified base/delta runtime checkpoints.

Raw map shards stay as single files in campaign/shards. Payloads contain only
new/changed files; their index references those shards without copying them.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile

from kaggle_a3_bundle import (LINUX_CASE_POLICY, PORTABLE_CASE_POLICY, archive, digest,
                             safe_name, sha256, validate_extraction_case_policy, verify)
from kaggle_native_paper_v3_bundle import source_entries as v3_source_entries

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_old_safe_sem_kaggle_v4"
KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_SOURCE"
CHECKPOINT_KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_CHECKPOINT_PAYLOAD"
INDEX_KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_CHECKPOINT_INDEX"
GIB = 1024**3
MAX_OUTPUT_BYTES = 16*GIB
EXPORT_RESERVE_BYTES = 512*1024**2
RUNTIME_DIRS = ("build-native-paper-v4", ".deps/install", ".venv-native-paper-v4", ".managed-python")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def source_entries(root=ROOT):
    root = Path(root)
    return v3_source_entries(root)+[(name, root/name) for name in (
        "requirements-native-paper-v4.in", "requirements-native-paper-v4.txt",
        "docs/A3T_SAFE_LENGTH.md", "docs/NATIVE_PAPER_V4_PROTOCOL.md", "docs/KAGGLE_NATIVE_PAPER_V4.md")]


def build_source(output, root=ROOT):
    output, root = Path(output), Path(root)
    for path in (output, output.with_suffix(".b64"), output.with_suffix(".ipynb")):
        if path.exists():
            raise ValueError("source output already exists: "+str(path))
    result = archive(source_entries(root), output, KIND, dict(protocol_id=PROTOCOL_ID,
        config_schema=13, main_requires_explicit_flag=True, contains_results=False))
    notebook = json.loads((root/"notebooks/native_paper_kaggle_v4.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["source"] = [line.replace("SOURCE_SHA256_PLACEHOLDER", sha256(output)) for line in cell["source"]]
    output.with_suffix(".ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    with output.open("rb") as source, output.with_suffix(".b64").open("xb") as encoded:
        for chunk in iter(lambda: source.read(3*1024*1024), b""):
            encoded.write(base64.b64encode(chunk))
    return result


def tree_size(folder):
    total = 0
    for parent, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent)/name).is_symlink()]
        total += sum((Path(parent)/name).stat().st_size for name in files
                     if not (Path(parent)/name).is_symlink() and (Path(parent)/name).is_file())
    return total


def output_scope(path):
    path = Path(path).resolve()
    kaggle = Path("/kaggle/working").resolve()
    return kaggle if path == kaggle or kaggle in path.parents else path


def _record(name, path):
    return dict(path=name, sha256=sha256(path), size_bytes=path.stat().st_size,
                mode=0o755 if os.name != "nt" and path.stat().st_mode & stat.S_IXUSR else 0o644)


def _inventory(workspace, root, source, *, runtime_dirs=RUNTIME_DIRS, nested_shards=False,
               runtime_excluded_prefixes=()):
    files, external, links = {}, {}, []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if "exports" in relative.parts or path.name in ("controller.lock",):
            continue
        if path.is_symlink():
            raise ValueError("campaign symlink refused: "+str(path))
        if path.is_file():
            name = "campaign/"+relative.as_posix()
            is_shard = ("shards" in relative.parts[:-1] if nested_shards else relative.parts[0] == "shards")
            (external if is_shard and path.suffix == ".zip" else files)[name] = path
    for directory in runtime_dirs:
        for path in (root/directory).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if any(relative == prefix or relative.startswith(prefix+"/") for prefix in runtime_excluded_prefixes):
                continue
            if path.suffix in (".o", ".obj", ".pyc") or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                target = path.resolve(strict=True)
                if root not in target.parents:
                    raise ValueError("runtime link escapes repository")
                links.append(dict(path="repo/"+path.relative_to(root).as_posix(),
                    target="repo/"+target.relative_to(root).as_posix(), target_is_directory=target.is_dir(),
                    target_sha256=None if target.is_dir() else sha256(target)))
            if path.is_file():
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise ValueError("runtime file escapes repository")
                files["repo/"+path.relative_to(root).as_posix()] = resolved
    if source is not None:
        files["source_input.zip"] = Path(source).resolve()
    return files, external, sorted(links, key=lambda x: x["path"])


def _index(path, *, protocol_id=PROTOCOL_ID, index_kind=INDEX_KIND):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("kind") != index_kind or value.get("protocol_id") != protocol_id:
        raise ValueError("wrong checkpoint index protocol")
    if value.get("inventory_sha256") != digest(value["inventory"]):
        raise ValueError("checkpoint inventory hash mismatch")
    return value


def export_checkpoint(workspace, output, *, root=ROOT, input_archive=None,
                      max_output_bytes=MAX_OUTPUT_BYTES, reserve_bytes=EXPORT_RESERVE_BYTES,
                      protocol_id=PROTOCOL_ID, checkpoint_kind=CHECKPOINT_KIND, index_kind=INDEX_KIND,
                      schema_version=13, runtime_policy="native_paper_sessions_v4", runtime_dirs=RUNTIME_DIRS,
                      runtime_binary="build-native-paper-v4/terrain_benchmark", nested_shards=False,
                      runtime_excluded_prefixes=()):
    """Append one verified payload, then atomically publish its chain index.

    The output must be under workspace/exports. Previous immutable payloads and
    raw shards are verified before they are referenced by the next generation.
    """
    workspace, root, output = Path(workspace).resolve(), Path(root).resolve(), Path(output).resolve()
    exports = workspace/"exports"
    exports.mkdir(parents=True, exist_ok=True)
    if output.parent != exports:
        raise ValueError("checkpoint payload must be directly under workspace/exports")
    index_path = exports/"checkpoint_index.json"
    previous = _index(index_path, protocol_id=protocol_id, index_kind=index_kind) if index_path.exists() else None
    source_hash = sha256(input_archive) if input_archive is not None else None
    identity = dict(protocol_id=protocol_id, root_path=str(root), workspace_path=str(workspace),
                    source_input_sha256=source_hash, runtime_policy=runtime_policy)
    if previous:
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("checkpoint identity drift")
        for item in previous["payloads"]:
            if sha256(exports/safe_name(item["path"])) != item["sha256"]:
                raise ValueError("previous checkpoint payload changed")
    files, external, links = _inventory(workspace, root, input_archive, runtime_dirs=runtime_dirs,
                                       nested_shards=nested_shards, runtime_excluded_prefixes=runtime_excluded_prefixes)
    if previous and links != previous["internal_runtime_symlinks"]:
        raise ValueError("immutable runtime symlink graph changed")
    current = {name: _record(name, path) for name, path in sorted({**files, **external}.items())}
    prior = previous["inventory"] if previous else {}
    for name, record in prior.items():
        is_shard = (("shards" in Path(name).parts[:-1]) if nested_shards else name.startswith("campaign/shards/")) and name.endswith(".zip")
        if name.startswith(("repo/", "source_input.zip", "campaign/data/", "campaign/plans/", "campaign/shared/")) or is_shard:
            if name not in current or current[name] != record:
                raise ValueError("immutable checkpoint asset changed: "+name)
    changed = [(name, path) for name, path in files.items() if current[name] != prior.get(name)]
    deleted = [record for name, record in prior.items() if name not in current]
    size = sum(path.stat().st_size for _, path in changed)+1024*1024
    # Streaming ZIP has bounded memory. Reserve its worst-case input size so a
    # partial output cannot exhaust the disk before the verified index commits.
    base = root.parent if root.parent == workspace.parent else workspace
    scope = output_scope(base)
    retained = tree_size(scope)+(0 if scope == root or scope in root.parents else tree_size(root))
    if retained+size > max_output_bytes:
        raise RuntimeError("BLOCKED_DISK: projected retained output exceeds 16 GiB cap")
    if shutil.disk_usage(exports).free < size+reserve_bytes:
        raise RuntimeError("BLOCKED_DISK: insufficient streaming checkpoint reserve")
    metadata = dict(identity, sequence=len(previous["payloads"]) if previous else 0,
        previous_payload_sha256=previous["payloads"][-1]["sha256"] if previous else None,
        payload_type="delta" if previous else "base", removed=deleted,
        internal_runtime_symlinks=links, external_files=[current[name] for name in sorted(external)],
        contains_exact_runtime=(root/runtime_binary).is_file(),
        previous_files={name: prior[name]["sha256"] for name, _ in changed if name in prior})
    policy = LINUX_CASE_POLICY if sys.platform.startswith("linux") else PORTABLE_CASE_POLICY
    result = archive(changed, output, checkpoint_kind, metadata, path_case_policy=policy)
    payloads = (previous["payloads"][:] if previous else [])+[
        dict(path=output.name, sha256=sha256(output), size_bytes=output.stat().st_size)]
    value = dict(identity, kind=index_kind, schema_version=schema_version, payloads=payloads,
        inventory=current, inventory_sha256=digest(current), internal_runtime_symlinks=links)
    write(index_path, value)
    index_path.with_name(index_path.name+".sha256").write_text(sha256(index_path)+"  "+index_path.name+"\n")
    return result


def _destination(target, name):
    path = target/safe_name(name)
    if target not in path.parent.resolve().parents and path.parent.resolve() != target:
        raise ValueError("checkpoint destination escapes target")
    return path


def _restore_stream(source, destination, record):
    """Resume only a byte-identical prefix left by an interrupted extraction."""
    pending = destination.with_name(destination.name+".restore-partial")
    if pending.exists() or pending.is_symlink():
        if pending.is_symlink() or not pending.is_file() or pending.stat().st_size > record["size_bytes"]:
            raise ValueError("unverified interrupted restore file")
        with pending.open("rb") as prefix:
            for chunk in iter(lambda: prefix.read(1024*1024), b""):
                if source.read(len(chunk)) != chunk:
                    raise ValueError("unverified interrupted restore prefix")
        mode = "ab"
    else:
        mode = "xb"
    with pending.open(mode) as output:
        shutil.copyfileobj(source, output)
    if pending.stat().st_size != record["size_bytes"] or sha256(pending) != record["sha256"]:
        raise ValueError("restored file hash/size mismatch")
    pending.chmod(record["mode"])
    pending.replace(destination)


def restore_checkpoint(index_path, target, expected_sha256, *, source_sha256, root=ROOT,
                       workspace=None, external_root=None, max_output_bytes=MAX_OUTPUT_BYTES,
                       reserve_bytes=EXPORT_RESERVE_BYTES, protocol_id=PROTOCOL_ID,
                       checkpoint_kind=CHECKPOINT_KIND, index_kind=INDEX_KIND):
    """Restore the latest indexed state after verifying the complete chain.

    Inputs remain read-only. Earlier versions of managed files may be replaced
    only when their bytes match a recorded generation in this exact chain.
    """
    index_path, target, root = Path(index_path).resolve(), Path(target).resolve(), Path(root).resolve()
    if sha256(index_path) != expected_sha256:
        raise ValueError("checkpoint index SHA-256 mismatch")
    index = _index(index_path, protocol_id=protocol_id, index_kind=index_kind)
    workspace = Path(workspace or target/"campaign").resolve()
    if index.get("source_input_sha256") != source_sha256:
        raise ValueError("checkpoint source identity mismatch")
    if (str(root), str(workspace)) != (index["root_path"], index["workspace_path"]) or (
            root != target/"repo" or workspace != target/"campaign"):
        raise ValueError("restore requires the same absolute repo/workspace paths")
    latest, history, payloads, external = {}, {}, [], {}
    previous = None
    for sequence, item in enumerate(index["payloads"]):
        path = index_path.parent/safe_name(item["path"])
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError("checkpoint payload size mismatch")
        manifest = verify(path, item["sha256"])
        validate_extraction_case_policy(manifest, target)
        metadata = manifest["metadata"]
        if manifest["kind"] != checkpoint_kind or metadata.get("sequence") != sequence or metadata.get("previous_payload_sha256") != previous:
            raise ValueError("checkpoint chain mismatch")
        for key in ("protocol_id", "source_input_sha256", "root_path", "workspace_path", "runtime_policy"):
            if metadata.get(key) != index.get(key):
                raise ValueError("checkpoint chain identity mismatch")
        for record in metadata["removed"]:
            name = safe_name(record["path"])
            if name not in latest or latest[name][1] != record:
                raise ValueError("invalid checkpoint tombstone")
            del latest[name]
        for record in manifest["files"]:
            name = record["path"]
            if name in latest and metadata["previous_files"].get(name) != latest[name][1]["sha256"]:
                raise ValueError("checkpoint predecessor file mismatch")
            history.setdefault(name, set()).add(record["sha256"])
            latest[name] = (path, record)
        external = {record["path"]: record for record in metadata["external_files"]}
        payloads.append((path, item))
        previous = item["sha256"]
    if not payloads or {**{name: pair[1] for name, pair in latest.items()}, **external} != index["inventory"]:
        raise ValueError("checkpoint chain final inventory mismatch")
    links = index["internal_runtime_symlinks"]
    seen = set()
    for link in links:
        name, real = safe_name(link["path"]), safe_name(link["target"])
        if name in seen or not name.startswith("repo/") or not real.startswith("repo/") or name == real:
            raise ValueError("invalid runtime symlink metadata")
        seen.add(name)
        alias, resolved = _destination(target, name), _destination(target, real)
        if alias.is_symlink() and alias.resolve() != resolved.resolve():
            raise ValueError("existing runtime link differs")
        if link["target_is_directory"]:
            if alias.exists() and not alias.is_symlink():
                raise ValueError("existing runtime directory alias differs")
        elif name not in latest or latest[name][1]["sha256"] != link["target_sha256"]:
            raise ValueError("runtime symlink alias hash mismatch")
    external_root = Path(external_root).resolve() if external_root else index_path.parents[2]
    external_sources = {}
    for name, record in external.items():
        source = external_root/safe_name(name)
        if source.is_symlink() or not source.is_file() or source.stat().st_size != record["size_bytes"] or sha256(source) != record["sha256"]:
            raise ValueError("external raw shard missing or changed: "+name)
        external_sources[name] = source
        history.setdefault(name, set()).add(record["sha256"])
    for name, hashes in history.items():
        destination = _destination(target, name)
        if destination.is_symlink():
            if name not in seen:
                raise ValueError("unregistered symlink destination")
        elif destination.exists() and (not destination.is_file() or sha256(destination) not in hashes):
            raise ValueError("refusing to overwrite unverified checkpoint destination: "+name)
    needed = sum(record["size_bytes"] for name, record in index["inventory"].items()
                 if name not in seen and (not (target/name).is_file() or sha256(target/name) != record["sha256"]))
    exports = workspace/"exports"
    for source, item in payloads:
        destination = exports/item["path"]
        if destination.exists() and (destination.is_symlink() or sha256(destination) != item["sha256"]):
            raise ValueError("retained payload destination differs")
        if not destination.exists():
            needed += item["size_bytes"]
    probe = target
    while not probe.exists():
        probe = probe.parent
    if tree_size(output_scope(target))+needed+1024*1024 > max_output_bytes:
        raise RuntimeError("BLOCKED_DISK: restored output would exceed 16 GiB cap")
    if shutil.disk_usage(probe).free < needed+reserve_bytes:
        raise RuntimeError("BLOCKED_DISK: insufficient restore reserve")
    # All bytes and destinations passed preflight. Writes use sibling temporary
    # files and atomic replacement; a retry accepts known earlier generations.
    for name, (payload, record) in latest.items():
        destination = _destination(target, name)
        if name in seen:
            continue
        if destination.is_file() and sha256(destination) == record["sha256"]:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(payload) as contents, contents.open(name) as src:
            _restore_stream(src, destination, record)
    for name, source in external_sources.items():
        destination = _destination(target, name)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src:
            _restore_stream(src, destination, external[name])
    for name in history:
        if name not in index["inventory"]:
            destination = _destination(target, name)
            if destination.is_file():
                destination.unlink()
    for link in links:
        alias, real = target/link["path"], target/link["target"]
        if link["target_is_directory"]:
            if not real.is_dir():
                raise ValueError("runtime directory target missing")
        elif not real.is_file() or sha256(real) != link["target_sha256"]:
            raise ValueError("runtime target hash mismatch")
        if alias.is_symlink():
            continue
        if alias.exists():
            alias.unlink()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(os.path.relpath(real, alias.parent), target_is_directory=link["target_is_directory"])
    # Retain the same payload bytes for the next notebook-output generation.
    exports.mkdir(parents=True, exist_ok=True)
    for source, item in payloads:
        destination = exports/item["path"]
        if destination.resolve() != source.resolve():
            if destination.exists() and sha256(destination) != item["sha256"]:
                raise ValueError("retained payload destination differs")
            if not destination.exists():
                shutil.copyfile(source, destination)
        destination.with_name(destination.name+".sha256").write_text(item["sha256"]+"  "+item["path"]+"\n")
    write(exports/"checkpoint_index.json", index)
    (exports/"checkpoint_index.json.sha256").write_text(sha256(exports/"checkpoint_index.json")+"  checkpoint_index.json\n")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("archive", type=Path)
    export = sub.add_parser("export")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--input-archive", type=Path, required=True)
    args = parser.parse_args()
    result = build_source(args.output) if args.command == "source" else verify(args.archive) if args.command == "verify" else export_checkpoint(args.workspace, args.output, input_archive=args.input_archive)
    print(json.dumps({"kind": result["kind"], "files": len(result["files"]), "files_sha256": result["files_sha256"]}))


if __name__ == "__main__":
    main()
