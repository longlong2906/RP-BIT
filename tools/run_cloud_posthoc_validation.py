"""Portable, fail-closed runner for the interrupted A3-vs-all validation.

This program deliberately cannot launch a planner.  It consumes the immutable
capsule produced by ``build_cloud_validation_capsule.py``, verifies every byte,
loads the independent validator from the frozen ``source_at_run.zip``, checks
five already-known control trials, and then validates exactly the 135 pending
trials.  Its only deliverable is an allowlisted validation patch.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterable
import zipfile


CAPSULE_ARTIFACT = "A3_VS_ALL_CLOUD_VALIDATION_CAPSULE"
PATCH_ARTIFACT = "A3_VS_ALL_CLOUD_VALIDATION_PATCH"
EXPECTED_PENDING_COUNT = 135
EXPECTED_CONTROL_COUNT = 5
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "98a4656c6f46e8102af6b564c221d7cf3d11f5f7e11c838cca3f5377015b0580"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "f164b99807ad475f7bd31286f0d95939c98fb75830971883cc519214b9d95244"
)
EXPECTED_CODE_SHA256 = (
    "43d1354e1985558936356d34b857b408b31d9fb8768c1e9838cab4a19d567440"
)
EXPECTED_BASELINE_HASHES = {
    "manifest.json": "c474c669a510dc4a6feebe99f55b6b892ce131bf4e85d581a20e456742644f14",
    "campaign_state.json": "dbf1f0ac2f60bc78fe8e5e315a0b8e678b0c9791461d45f87ac1d334638c60b9",
    "map_manifest.json": "fc8fb97cf9db0c36a6c191c3f66273469afdf058fb995bd9002886f0f25a6d2a",
    "resolved_config.json": "bbcf830fb34385cdeebfb78bbd9b2c21931d64194578e51c3c3a2ba79210831b",
    "source_manifest.json": EXPECTED_SOURCE_MANIFEST_SHA256,
    "source_at_run.zip": EXPECTED_SOURCE_ARCHIVE_SHA256,
}
EXPECTED_CAPSULE_BASELINE_HASHES = {
    key: EXPECTED_BASELINE_HASHES[key]
    for key in ("manifest.json", "map_manifest.json", "resolved_config.json", "source_at_run.zip")
}
EXPECTED_CONTROL_TRIAL_IDS = [
    "mixed_obstacles_5005_q0_sem_rrtstar_common_v1_s93001",
    "narrow_passage_5009_q0_abitstar_s93001",
    "ridge_pass_5004_q0_aitstar_s93001",
    "rolling_terrain_5008_q0_bitstar_a3_s93001",
    "traversable_hill_5005_q0_lazyprmstar_hauser2015_s93001",
]
EXPECTED_FILE_CONTRACT = {
    "pending_validation_outputs": "excluded_even_if_partially_present_in_primary",
    "pending_input_files": [
        "request.json",
        "trial.json",
        "progress.jsonl",
        "path_xy.csv",
        "path_surface.csv",
    ],
    "snapshot_input_files": ["path_xy.csv", "path_surface.csv"],
    "control_expected_files": ["trial.json", "validation.json", "validated_events.json"],
    "control_snapshot_expected_files": ["validation.json"],
    "map_resolution": "map_id_and_sha256_never_embedded_host_path",
}

CAPSULE_MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "entry_inventory_policy",
    "entries",
    "primary",
    "pending_trial_ids",
    "validation_order",
    "control_trial_ids",
    "maps",
    "file_contract",
}
CAPSULE_PRIMARY_KEYS = {
    "experiment_id",
    "protocol_id",
    "status",
    "phase",
    "identity",
    "manifest_counts",
    "campaign_counts",
    "baseline_hashes",
    "primary_inventory_sha256",
}
CAPSULE_MAP_KEYS = {"map_id", "archive_path", "sha256", "original_paths"}
ENTRY_KEYS = {"path", "size_bytes", "sha256"}
PRIMARY_INVENTORY_KEYS = {"schema_version", "policy", "entries", "inventory_sha256"}
SOURCE_MANIFEST_KEYS = {"code_sha256", "files", "archive_sha256", "hash_policy"}

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9_]+\Z")
WINDOWS_MAP_PATH = re.compile(
    r"D:\\BIT_2\.5D\\data\\a3_vs_all_v1\\(?P<map_id>[A-Za-z0-9_]+)\.t25d\Z"
)
SNAPSHOT_ID = re.compile(r"snapshot_[0-9]+\Z")
NUMERIC_PARITY_FIELDS = (
    "cost_terrain_m",
    "length_xy_m",
    "cost_abs_error_m",
    "minimum_footprint_clearance_m",
    "maximum_slope_deg",
)
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_MEMBER_BYTES = 768 * 1024**2


class IntegrityError(RuntimeError):
    """The capsule/checkpoint violates its immutable integrity contract."""


class ControlMismatch(RuntimeError):
    """The Linux validator does not reproduce a frozen Windows control."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_tree_files(root: Path, label: str) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise IntegrityError(f"{label}: root must be a real directory")
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise IntegrityError(f"{label}: symlink is forbidden: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise IntegrityError(f"{label}: non-regular file is forbidden: {relative}")
        files[relative] = path
    return files


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite JSON constant: " + value)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IntegrityError(f"invalid JSON in {label}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        return _read_json_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise IntegrityError(f"cannot read {path}: {exc}") from exc


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _exclusive_write(path: Path, data: bytes) -> None:
    """Create *path* once; never replace an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            with contextlib.suppress(OSError):
                path.unlink()
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _json_bytes(value))


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise IntegrityError(f"{label}: missing/unknown fields: {sorted(actual ^ keys)}")
    return value


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise IntegrityError(f"{label}: expected lowercase SHA-256")
    return value


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise IntegrityError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise IntegrityError(f"absolute ZIP member path: {name!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise IntegrityError(f"non-canonical ZIP member path: {name!r}")
    canonical = path.as_posix()
    if canonical != name:
        raise IntegrityError(f"non-canonical ZIP member path: {name!r}")
    return canonical


def _zip_members(archive: zipfile.ZipFile, label: str) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise IntegrityError(f"{label}: too many ZIP members")
    members: dict[str, zipfile.ZipInfo] = {}
    collision_keys: set[str] = set()
    total = 0
    for info in infos:
        name = _safe_member_name(info.filename)
        if info.is_dir():
            raise IntegrityError(f"{label}: explicit directory entry is not allowed: {name}")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode):
            raise IntegrityError(f"{label}: symlink is not allowed: {name}")
        if kind not in (0, stat.S_IFREG):
            raise IntegrityError(f"{label}: non-regular member is not allowed: {name}")
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise IntegrityError(f"{label}: unreasonable member size: {name}")
        total += info.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise IntegrityError(f"{label}: uncompressed size limit exceeded")
        collision = name.casefold()
        if name in members or collision in collision_keys:
            raise IntegrityError(f"{label}: duplicate/colliding ZIP member: {name}")
        members[name] = info
        collision_keys.add(collision)
    return members


def _parse_sidecar(path: Path, archive: Path) -> str:
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise IntegrityError(f"missing/unreadable SHA-256 sidecar {path}: {exc}") from exc
    fields = text.split()
    if not fields or not HEX64.fullmatch(fields[0]):
        raise IntegrityError(f"invalid SHA-256 sidecar: {path}")
    if len(fields) > 2:
        raise IntegrityError(f"invalid SHA-256 sidecar field count: {path}")
    if len(fields) == 2:
        recorded_name = fields[1].lstrip("*")
        if recorded_name != archive.name:
            raise IntegrityError(
                f"sidecar filename mismatch: expected {archive.name!r}, got {recorded_name!r}"
            )
    return fields[0]


def _entries_by_path(raw: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise IntegrityError(f"{label}: entries must be a sorted list")
    result: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(raw):
        entry = _exact_keys(item, ENTRY_KEYS, f"{label}[{index}]")
        path = _safe_member_name(entry["path"])
        if path in result:
            raise IntegrityError(f"{label}: duplicate entry {path}")
        if not isinstance(entry["size_bytes"], int) or isinstance(entry["size_bytes"], bool) or entry["size_bytes"] < 0:
            raise IntegrityError(f"{label}: invalid size for {path}")
        _require_hash(entry["sha256"], f"{label}:{path}")
        result[path] = entry
        order.append(path)
    if order != sorted(order):
        raise IntegrityError(f"{label}: entries are not sorted")
    return result


def _hash_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_member_inventory(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    inventory: dict[str, dict[str, Any]],
    manifest_name: str,
    label: str,
) -> None:
    expected = set(inventory) | {manifest_name}
    if set(members) != expected:
        missing = sorted(expected - set(members))
        extra = sorted(set(members) - expected)
        raise IntegrityError(f"{label}: archive allowlist mismatch; missing={missing}, extra={extra}")
    for name, entry in inventory.items():
        info = members[name]
        if info.file_size != entry["size_bytes"]:
            raise IntegrityError(f"{label}: size mismatch for {name}")
        if _hash_zip_member(archive, info) != entry["sha256"]:
            raise IntegrityError(f"{label}: SHA-256 mismatch for {name}")


def _verify_primary_inventory(data: bytes, expected_inventory_digest: str) -> dict[str, Any]:
    value = _exact_keys(
        _read_json_bytes(data, "primary_inventory.json"),
        PRIMARY_INVENTORY_KEYS,
        "primary_inventory.json",
    )
    if value["schema_version"] != 1 or value["policy"] != "all_regular_files_exact":
        raise IntegrityError("unsupported primary inventory schema/policy")
    entries = _entries_by_path(value["entries"], "primary_inventory.entries")
    canonical = [entries[path] for path in sorted(entries)]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if value["inventory_sha256"] != digest or digest != expected_inventory_digest:
        raise IntegrityError("primary inventory digest mismatch")
    return value


def _source_manifest_code_digest(files: dict[str, str]) -> str:
    # The original Windows code hash used insertion order and backslash paths.
    pairs = [(name, digest) for name, digest in files.items() if not name.replace("\\", "/").startswith("configs/")]
    encoded = json.dumps(pairs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_frozen_source(source_manifest_data: bytes, source_archive_data: bytes) -> dict[str, Any]:
    if _sha256_bytes(source_manifest_data) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise IntegrityError("unexpected frozen source_manifest.json identity")
    source_manifest = _exact_keys(
        _read_json_bytes(source_manifest_data, "metadata/source_manifest.json"),
        SOURCE_MANIFEST_KEYS,
        "source_manifest.json",
    )
    if source_manifest["archive_sha256"] != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise IntegrityError("source manifest is not bound to the registered source archive")
    if _sha256_bytes(source_archive_data) != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise IntegrityError("frozen source_at_run.zip hash mismatch")
    if source_manifest["code_sha256"] != EXPECTED_CODE_SHA256:
        raise IntegrityError("unexpected frozen code identity")
    files = source_manifest["files"]
    if not isinstance(files, dict) or not files:
        raise IntegrityError("source_manifest.files must be a nonempty mapping")
    normalized: dict[str, str] = {}
    for original, digest in files.items():
        if not isinstance(original, str):
            raise IntegrityError("non-string path in source manifest")
        name = _safe_member_name(original.replace("\\", "/"))
        if name in normalized:
            raise IntegrityError(f"source manifest path collision: {name}")
        normalized[name] = _require_hash(digest, f"source manifest:{original}")
    if _source_manifest_code_digest(files) != EXPECTED_CODE_SHA256:
        raise IntegrityError("source_manifest code digest is internally inconsistent")
    with zipfile.ZipFile(io.BytesIO(source_archive_data), "r") as source_archive:
        members = _zip_members(source_archive, "source_at_run.zip")
        if set(members) != set(normalized):
            raise IntegrityError("source archive member set does not match source manifest")
        for name, expected in normalized.items():
            if _hash_zip_member(source_archive, members[name]) != expected:
                raise IntegrityError(f"frozen source member hash mismatch: {name}")
    for required in ("tools/validate_results.py", "tools/independent_verifier.py", "tools/experiment_common.py", "tools/run_experiments.py"):
        if required not in normalized:
            raise IntegrityError(f"frozen source is missing {required}")
    return source_manifest


def _validate_id_list(raw: Any, label: str, expected_count: int) -> list[str]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise IntegrityError(f"{label}: expected exactly {expected_count} IDs")
    if any(not isinstance(value, str) or not SAFE_ID.fullmatch(value) for value in raw):
        raise IntegrityError(f"{label}: unsafe trial ID")
    if len(set(raw)) != len(raw):
        raise IntegrityError(f"{label}: duplicate trial ID")
    return list(raw)


def _validate_capsule_manifest(manifest: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _exact_keys(manifest, CAPSULE_MANIFEST_KEYS, "capsule_manifest.json")
    if manifest["schema_version"] != 1 or manifest["artifact_type"] != CAPSULE_ARTIFACT:
        raise IntegrityError("unsupported capsule schema/artifact type")
    if manifest["entry_inventory_policy"] != "all_members_except_capsule_manifest_due_to_self_reference":
        raise IntegrityError("unsupported capsule entry inventory policy")
    entries = _entries_by_path(manifest["entries"], "capsule_manifest.entries")
    primary = _exact_keys(manifest["primary"], CAPSULE_PRIMARY_KEYS, "capsule_manifest.primary")
    if primary["experiment_id"] != "a3_vs_all_v1_20260930" or primary["protocol_id"] != "a3_vs_all_v1":
        raise IntegrityError("capsule targets an unexpected experiment/protocol")
    if primary["status"] != "PARTIAL" or primary["phase"] != "INTERRUPTED":
        raise IntegrityError("capsule is not from the registered pause point")
    baseline_hashes = primary["baseline_hashes"]
    if baseline_hashes != EXPECTED_CAPSULE_BASELINE_HASHES:
        raise IntegrityError("capsule baseline hashes differ from the registered pause point")
    _require_hash(primary["primary_inventory_sha256"], "primary.primary_inventory_sha256")
    if not isinstance(primary["identity"], dict) or not isinstance(primary["manifest_counts"], dict) or not isinstance(primary["campaign_counts"], dict):
        raise IntegrityError("invalid primary identity/counts")
    pending = _validate_id_list(manifest["pending_trial_ids"], "pending_trial_ids", EXPECTED_PENDING_COUNT)
    order = _validate_id_list(manifest["validation_order"], "validation_order", EXPECTED_PENDING_COUNT)
    controls = _validate_id_list(manifest["control_trial_ids"], "control_trial_ids", EXPECTED_CONTROL_COUNT)
    if set(order) != set(pending):
        raise IntegrityError("validation_order is not a permutation of pending_trial_ids")
    if set(controls) & set(pending):
        raise IntegrityError("control and pending trial IDs overlap")
    if controls != EXPECTED_CONTROL_TRIAL_IDS:
        raise IntegrityError("control trial IDs/order differ from the locked parity gate")
    maps = manifest["maps"]
    if not isinstance(maps, list) or not maps:
        raise IntegrityError("capsule maps must be a nonempty list")
    seen_maps: set[str] = set()
    for index, raw in enumerate(maps):
        record = _exact_keys(raw, CAPSULE_MAP_KEYS, f"capsule maps[{index}]")
        map_id = record["map_id"]
        if not isinstance(map_id, str) or not SAFE_ID.fullmatch(map_id) or map_id in seen_maps:
            raise IntegrityError(f"invalid/duplicate map_id: {map_id!r}")
        seen_maps.add(map_id)
        if record["archive_path"] != f"maps/{map_id}.t25d":
            raise IntegrityError(f"noncanonical archive path for map {map_id}")
        _require_hash(record["sha256"], f"map:{map_id}")
        original_paths = record["original_paths"]
        if not isinstance(original_paths, list) or not original_paths or len(set(original_paths)) != len(original_paths):
            raise IntegrityError(f"invalid original_paths for map {map_id}")
        for original in original_paths:
            match = WINDOWS_MAP_PATH.fullmatch(original) if isinstance(original, str) else None
            if not match or match.group("map_id") != map_id:
                raise IntegrityError(f"unexpected original map path for {map_id}: {original!r}")
    if manifest["file_contract"] != EXPECTED_FILE_CONTRACT:
        raise IntegrityError("capsule file_contract differs from schema v1")
    return manifest, entries


def _trial_input_prefix(kind: str, trial_id: str) -> str:
    return f"controls/{trial_id}/inputs/" if kind == "control" else f"pending/{trial_id}/"


def _validate_input_tree(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    prefix: str,
    trial_id: str,
    manifest_entry: dict[str, Any],
    map_records: dict[str, dict[str, Any]],
    *,
    pending: bool,
) -> str:
    names = sorted(name for name in members if name.startswith(prefix))
    if not names:
        raise IntegrityError(f"missing input tree for {trial_id}")
    relatives = [name[len(prefix):] for name in names]
    required = {"request.json", "trial.json", "progress.jsonl", "path_xy.csv", "path_surface.csv"}
    if not required.issubset(relatives):
        raise IntegrityError(f"{trial_id}: missing required inputs {sorted(required - set(relatives))}")
    for relative in relatives:
        if relative in ("validation.json", "validated_events.json") or relative.endswith("/validation.json"):
            raise IntegrityError(f"{trial_id}: validation output leaked into immutable inputs")
        if relative.startswith("snapshots/"):
            parts = PurePosixPath(relative).parts
            if len(parts) != 3 or not SNAPSHOT_ID.fullmatch(parts[1]) or parts[2] not in ("path_xy.csv", "path_surface.csv"):
                raise IntegrityError(f"{trial_id}: unexpected snapshot input {relative}")
        elif relative not in required:
            raise IntegrityError(f"{trial_id}: unexpected trial input {relative}")
    request = _read_json_bytes(archive.read(members[prefix + "request.json"]), prefix + "request.json")
    trial = _read_json_bytes(archive.read(members[prefix + "trial.json"]), prefix + "trial.json")
    if not isinstance(request, dict) or not isinstance(trial, dict):
        raise IntegrityError(f"{trial_id}: request/trial JSON must be objects")
    if trial.get("trial_id") != trial_id or manifest_entry.get("trial_id") != trial_id:
        raise IntegrityError(f"{trial_id}: trial identity mismatch")
    map_id = trial.get("map_id")
    if map_id != manifest_entry.get("map_id") or map_id not in map_records:
        raise IntegrityError(f"{trial_id}: map_id mismatch")
    map_record = map_records[map_id]
    original = request.get("map_path")
    match = WINDOWS_MAP_PATH.fullmatch(original) if isinstance(original, str) else None
    if not match or match.group("map_id") != map_id or original not in map_record["original_paths"]:
        raise IntegrityError(f"{trial_id}: request map_path is not registered")
    if trial.get("map_path") != original or trial.get("map_hash") != map_record["sha256"] or manifest_entry.get("map_hash") != map_record["sha256"]:
        raise IntegrityError(f"{trial_id}: map path/hash mismatch")
    if request.get("planner_id") != trial.get("planner_id") or trial.get("planner_id") != manifest_entry.get("planner_id"):
        raise IntegrityError(f"{trial_id}: planner identity mismatch")
    if request.get("planner_seed") != trial.get("planner_seed") or trial.get("planner_seed") != manifest_entry.get("planner_seed"):
        raise IntegrityError(f"{trial_id}: planner seed mismatch")
    if trial.get("execution_status") != "COMPLETED" or trial.get("planner_status") != "EXACT_SOLUTION" or trial.get("path_available") is not True:
        raise IntegrityError(f"{trial_id}: unexpected planner outcome for pending validation")
    if pending:
        if trial.get("independent_valid") is not None or trial.get("success") is not False:
            raise IntegrityError(f"{trial_id}: input is not at the pending validation state")
    elif trial.get("independent_valid") is not True or trial.get("success") is not True:
        raise IntegrityError(f"{trial_id}: control input is not a validated success")
    progress_lines = archive.read(members[prefix + "progress.jsonl"]).decode("utf-8").splitlines()
    improvements: list[str] = []
    for number, line in enumerate(progress_lines, 1):
        if not line.strip():
            continue
        event = _read_json_bytes(line.encode("utf-8"), f"{prefix}progress.jsonl:{number}")
        if not isinstance(event, dict):
            raise IntegrityError(f"{trial_id}: non-object progress event")
        if event.get("event_type") == "improvement":
            snapshot_id = event.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not SNAPSHOT_ID.fullmatch(snapshot_id):
                raise IntegrityError(f"{trial_id}: unsafe snapshot_id")
            improvements.append(snapshot_id)
    if len(set(improvements)) != len(improvements):
        raise IntegrityError(f"{trial_id}: duplicate improvement snapshot ID")
    expected_snapshot_files = {
        f"snapshots/{snapshot_id}/{filename}"
        for snapshot_id in improvements
        for filename in ("path_xy.csv", "path_surface.csv")
    }
    actual_snapshot_files = {relative for relative in relatives if relative.startswith("snapshots/")}
    if actual_snapshot_files != expected_snapshot_files:
        raise IntegrityError(f"{trial_id}: snapshot paths do not match progress events")
    return map_id


def _validate_expected_control_tree(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    trial_id: str,
) -> None:
    prefix = f"controls/{trial_id}/expected/"
    relatives = sorted(name[len(prefix):] for name in members if name.startswith(prefix))
    required = {"trial.json", "validation.json", "validated_events.json"}
    if not required.issubset(relatives):
        raise IntegrityError(f"{trial_id}: incomplete expected control output")
    allowed = set(required)
    events = _read_json_bytes(archive.read(members[prefix + "validated_events.json"]), prefix + "validated_events.json")
    if not isinstance(events, list):
        raise IntegrityError(f"{trial_id}: expected validated_events must be a list")
    snapshot_ids: list[str] = []
    for event in events:
        snapshot_id = event.get("snapshot_id") if isinstance(event, dict) else None
        if not isinstance(snapshot_id, str) or not SNAPSHOT_ID.fullmatch(snapshot_id):
            raise IntegrityError(f"{trial_id}: unsafe expected snapshot ID")
        snapshot_ids.append(snapshot_id)
        allowed.add(f"snapshots/{snapshot_id}/validation.json")
    if len(set(snapshot_ids)) != len(snapshot_ids) or set(relatives) != allowed:
        raise IntegrityError(f"{trial_id}: expected control allowlist mismatch")


def _verify_capsule_structure(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    manifest: dict[str, Any],
) -> None:
    metadata_expected = {
        "metadata/manifest.json",
        "metadata/campaign_state.json",
        "metadata/map_manifest.json",
        "metadata/resolved_config.json",
        "metadata/source_manifest.json",
        "metadata/source_at_run.zip",
        "primary_inventory.json",
    }
    if not metadata_expected.issubset(members):
        raise IntegrityError(f"missing capsule metadata: {sorted(metadata_expected - set(members))}")
    primary = manifest["primary"]
    _verify_primary_inventory(archive.read(members["primary_inventory.json"]), primary["primary_inventory_sha256"])
    for filename, expected in EXPECTED_BASELINE_HASHES.items():
        member = "metadata/" + filename
        if _hash_zip_member(archive, members[member]) != expected:
            raise IntegrityError(f"registered metadata hash mismatch: {filename}")
    source_manifest_data = archive.read(members["metadata/source_manifest.json"])
    source_archive_data = archive.read(members["metadata/source_at_run.zip"])
    _verify_frozen_source(source_manifest_data, source_archive_data)

    campaign_state = _read_json_bytes(archive.read(members["metadata/campaign_state.json"]), "campaign_state.json")
    manifest_json = _read_json_bytes(archive.read(members["metadata/manifest.json"]), "manifest.json")
    if campaign_state.get("counts", {}).get("offline_validated") != 115 or campaign_state.get("counts", {}).get("validation_pending") != 135:
        raise IntegrityError("campaign state does not contain the 115/135 pause point")
    if manifest_json.get("planned_trials") != 250 or manifest_json.get("identity") != primary["identity"]:
        raise IntegrityError("primary manifest identity/count mismatch")
    if primary["campaign_counts"] != campaign_state.get("counts") or primary["manifest_counts"] != manifest_json.get("counts"):
        raise IntegrityError("capsule primary counts do not match frozen metadata")
    manifest_entries = manifest_json.get("trials")
    if not isinstance(manifest_entries, list) or len(manifest_entries) != 250:
        raise IntegrityError("primary manifest must contain exactly 250 trials")
    by_trial = {entry.get("trial_id"): entry for entry in manifest_entries if isinstance(entry, dict)}
    if len(by_trial) != 250:
        raise IntegrityError("duplicate/invalid trial identity in primary manifest")
    expected_pending_order = [
        entry["trial_id"]
        for entry in sorted(manifest_entries, key=lambda entry: entry.get("shuffle_order", -1))
        if entry.get("shuffle_order", -1) >= 115
    ]
    if expected_pending_order != manifest["pending_trial_ids"] or expected_pending_order != manifest["validation_order"]:
        raise IntegrityError("pending IDs/order do not match shuffle positions 115..249")

    map_records = {record["map_id"]: record for record in manifest["maps"]}
    for map_id, record in map_records.items():
        name = record["archive_path"]
        if name not in members or _hash_zip_member(archive, members[name]) != record["sha256"]:
            raise IntegrityError(f"map archive/hash mismatch: {map_id}")
    used_maps: list[str] = []
    for trial_id in manifest["validation_order"]:
        if trial_id not in by_trial:
            raise IntegrityError(f"pending trial absent from primary manifest: {trial_id}")
        used_maps.append(
            _validate_input_tree(
                archive,
                members,
                _trial_input_prefix("pending", trial_id),
                trial_id,
                by_trial[trial_id],
                map_records,
                pending=True,
            )
        )
    # Each map forms exactly one contiguous group, which permits one Oracle per map.
    closed: set[str] = set()
    previous: str | None = None
    for map_id in used_maps:
        if map_id != previous:
            if map_id in closed:
                raise IntegrityError("validation_order is not grouped deterministically by map")
            if previous is not None:
                closed.add(previous)
            previous = map_id
    control_maps: set[str] = set()
    for trial_id in manifest["control_trial_ids"]:
        if trial_id not in by_trial:
            raise IntegrityError(f"control trial absent from primary manifest: {trial_id}")
        control_maps.add(_validate_input_tree(
            archive,
            members,
            _trial_input_prefix("control", trial_id),
            trial_id,
            by_trial[trial_id],
            map_records,
            pending=False,
        ))
        _validate_expected_control_tree(archive, members, trial_id)
    if set(map_records) != set(used_maps) | control_maps:
        raise IntegrityError("capsule map set is not exactly the pending/control map set")

    permitted_prefixes = ("metadata/", "maps/", "pending/", "controls/")
    for name in members:
        if name in ("capsule_manifest.json", "primary_inventory.json"):
            continue
        if not name.startswith(permitted_prefixes):
            raise IntegrityError(f"unexpected top-level capsule member: {name}")


class VerifiedCapsule:
    def __init__(self, path: Path, sha256: str, manifest_sha256: str, manifest: dict[str, Any]):
        self.path = path
        self.sha256 = sha256
        self.manifest_sha256 = manifest_sha256
        self.manifest = manifest


def verify_capsule(capsule_path: Path, sidecar_path: Path | None = None) -> VerifiedCapsule:
    raw_capsule = Path(capsule_path)
    raw_sidecar = Path(sidecar_path) if sidecar_path is not None else Path(str(raw_capsule) + ".sha256")
    if raw_capsule.is_symlink() or raw_sidecar.is_symlink():
        raise IntegrityError("capsule and sidecar must be regular, non-symlink files")
    capsule_path = raw_capsule.resolve()
    sidecar_path = raw_sidecar.resolve()
    if capsule_path.suffix.lower() != ".zip":
        raise IntegrityError("capsule path must end in .zip")
    expected_archive_hash = _parse_sidecar(sidecar_path, capsule_path)
    try:
        actual_archive_hash = _sha256_file(capsule_path)
    except OSError as exc:
        raise IntegrityError(f"missing/unreadable capsule archive {capsule_path}: {exc}") from exc
    if actual_archive_hash != expected_archive_hash:
        raise IntegrityError("capsule archive does not match its SHA-256 sidecar")
    try:
        with zipfile.ZipFile(capsule_path, "r") as archive:
            members = _zip_members(archive, "capsule")
            if "capsule_manifest.json" not in members:
                raise IntegrityError("capsule_manifest.json is missing")
            manifest_data = archive.read(members["capsule_manifest.json"])
            manifest, inventory = _validate_capsule_manifest(
                _read_json_bytes(manifest_data, "capsule_manifest.json")
            )
            _verify_member_inventory(archive, members, inventory, "capsule_manifest.json", "capsule")
            _verify_capsule_structure(archive, members, manifest)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        if isinstance(exc, IntegrityError):
            raise
        raise IntegrityError(f"cannot verify capsule ZIP: {exc}") from exc
    return VerifiedCapsule(capsule_path, actual_archive_hash, _sha256_bytes(manifest_data), manifest)


def _extract_verified_archive(capsule: VerifiedCapsule, destination: Path) -> None:
    if destination.exists():
        raise IntegrityError(f"refusing to overwrite extraction directory: {destination}")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(capsule.path, "r") as archive:
        members = _zip_members(archive, "capsule")
        for name in sorted(members):
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[name], "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    # Re-open and fully verify the extracted copy, not merely ZIP CRC.
    entries = _entries_by_path(capsule.manifest["entries"], "capsule_manifest.entries")
    for name, entry in entries.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        if target.stat().st_size != entry["size_bytes"] or _sha256_file(target) != entry["sha256"]:
            raise IntegrityError(f"extracted member verification failed: {name}")
    manifest_path = destination / "capsule_manifest.json"
    if _sha256_file(manifest_path) != capsule.manifest_sha256:
        raise IntegrityError("extracted capsule manifest hash mismatch")


def _extract_frozen_source(extracted_capsule: Path, destination: Path) -> None:
    source_archive_path = extracted_capsule / "metadata/source_at_run.zip"
    source_manifest = _read_json(extracted_capsule / "metadata/source_manifest.json")
    normalized = {name.replace("\\", "/"): digest for name, digest in source_manifest["files"].items()}
    if destination.exists():
        actual = {
            relative: _sha256_file(path)
            for relative, path in _regular_tree_files(destination, "frozen source extraction").items()
        }
        if actual != normalized:
            raise IntegrityError("existing frozen source extraction was modified")
        return
    destination.mkdir(parents=True)
    with zipfile.ZipFile(source_archive_path, "r") as archive:
        members = _zip_members(archive, "source_at_run.zip")
        if set(members) != set(normalized):
            raise IntegrityError("source extraction member mismatch")
        for name in sorted(members):
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(members[name], "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if _sha256_file(target) != normalized[name]:
                raise IntegrityError(f"source extraction hash mismatch: {name}")


def _load_frozen_validator(source_root: Path):
    tools_root = (source_root / "tools").resolve()
    for name in ("validate_results", "independent_verifier", "run_experiments", "experiment_common"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(tools_root))
    try:
        spec = importlib.util.spec_from_file_location(
            "validate_results", tools_root / "validate_results.py"
        )
        if spec is None or spec.loader is None:
            raise IntegrityError("cannot load frozen validate_results.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["validate_results"] = module
        spec.loader.exec_module(module)
        oracle_module = sys.modules.get("independent_verifier")
        imported = {
            "validate_results": Path(module.__file__).resolve(),
            "independent_verifier": Path(oracle_module.__file__).resolve() if oracle_module else None,
        }
        for name, path in imported.items():
            if path is None or tools_root not in path.parents:
                raise IntegrityError(f"{name} was not imported from frozen source")
        return module.validate_trial, oracle_module.Oracle
    finally:
        if sys.path and sys.path[0] == str(tools_root):
            sys.path.pop(0)


def _copy_input_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise IntegrityError(f"refusing to overwrite staged trial: {destination}")
    shutil.copytree(source, destination, copy_function=shutil.copyfile)


def _map_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["map_id"]: record for record in manifest["maps"]}


def _prepare_compatibility_maps(extracted: Path, compat: Path, manifest: dict[str, Any]) -> None:
    compat.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        raise IntegrityError("cloud validation compatibility mode requires a POSIX filesystem")
    expected_names: set[str] = set()
    for record in manifest["maps"]:
        source = extracted.joinpath(*PurePosixPath(record["archive_path"]).parts)
        for original in record["original_paths"]:
            match = WINDOWS_MAP_PATH.fullmatch(original)
            if not match or match.group("map_id") != record["map_id"]:
                raise IntegrityError(f"unsafe compatibility map path: {original!r}")
            expected_names.add(original)
            destination = compat / original
            if destination.exists():
                if destination.is_symlink() or not destination.is_file() or not stat.S_ISREG(destination.lstat().st_mode) or _sha256_file(destination) != record["sha256"]:
                    raise IntegrityError(f"compatibility map was modified: {original}")
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            if _sha256_file(destination) != record["sha256"]:
                raise IntegrityError(f"compatibility map copy/hash mismatch: {original}")
    actual_names = set()
    for path in compat.iterdir():
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityError(f"compatibility directory contains non-regular state: {path.name}")
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise IntegrityError("compatibility directory contains unexpected files")


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _numeric_equal(actual: Any, expected: Any, tolerance: float = 1e-8) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return False
    return math.isfinite(float(actual)) and math.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= tolerance


def _compare_validation_record(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("valid", "errors", "method", "snapshot_id", "xy_sha256", "surface_sha256", "exact_goal_reached"):
        if key in actual or key in expected:
            if actual.get(key) != expected.get(key):
                raise ControlMismatch(f"{label}: exact field mismatch for {key}")
    for key in NUMERIC_PARITY_FIELDS:
        if key in actual or key in expected:
            if not _numeric_equal(actual.get(key), expected.get(key)):
                raise ControlMismatch(f"{label}: numeric field mismatch for {key}")


def compare_control_outputs(actual: Path, expected: Path, trial_id: str) -> dict[str, Any]:
    actual_validation = _read_json(actual / "validation.json")
    expected_validation = _read_json(expected / "validation.json")
    _compare_validation_record(actual_validation, expected_validation, f"{trial_id}:final")
    actual_snapshots = actual_validation.get("snapshots")
    expected_snapshots = expected_validation.get("snapshots")
    if not isinstance(actual_snapshots, list) or not isinstance(expected_snapshots, list) or len(actual_snapshots) != len(expected_snapshots):
        raise ControlMismatch(f"{trial_id}: final snapshot list mismatch")
    for index, (got, want) in enumerate(zip(actual_snapshots, expected_snapshots)):
        if not isinstance(got, dict) or not isinstance(want, dict):
            raise ControlMismatch(f"{trial_id}: non-object snapshot validation")
        _compare_validation_record(got, want, f"{trial_id}:snapshot[{index}]")

    actual_events = _read_json(actual / "validated_events.json")
    expected_events = _read_json(expected / "validated_events.json")
    if not isinstance(actual_events, list) or not isinstance(expected_events, list):
        raise ControlMismatch(f"{trial_id}: validated events are not lists")
    if [event.get("snapshot_id") for event in actual_events] != [event.get("snapshot_id") for event in expected_events]:
        raise ControlMismatch(f"{trial_id}: snapshot ID/order mismatch")
    for index, (got, want) in enumerate(zip(actual_events, expected_events)):
        if got != want:
            raise ControlMismatch(f"{trial_id}: validated event mismatch at index {index}")

    for event in actual_events:
        snapshot_id = event["snapshot_id"]
        got = _read_json(actual / "snapshots" / snapshot_id / "validation.json")
        want = _read_json(expected / "snapshots" / snapshot_id / "validation.json")
        _compare_validation_record(got, want, f"{trial_id}:{snapshot_id}")

    actual_trial = _read_json(actual / "trial.json")
    expected_trial = _read_json(expected / "trial.json")
    for key in (
        "independent_valid",
        "independently_recomputed_cost_m",
        "length_xy_m",
        "cost_abs_error_m",
        "minimum_footprint_clearance_m",
        "maximum_slope_deg",
        "success",
        "snapshots_valid",
    ):
        if key in NUMERIC_PARITY_FIELDS or key == "independently_recomputed_cost_m":
            if not _numeric_equal(actual_trial.get(key), expected_trial.get(key)):
                raise ControlMismatch(f"{trial_id}: trial numeric field mismatch for {key}")
        elif actual_trial.get(key) != expected_trial.get(key):
            raise ControlMismatch(f"{trial_id}: trial field mismatch for {key}")
    return {
        "trial_id": trial_id,
        "status": "PASS",
        "snapshots_compared": len(actual_events),
        "validation_wall_s_ignored": True,
        "numeric_absolute_tolerance": 1e-8,
    }


def _input_inventory(directory: Path) -> list[dict[str, Any]]:
    # trial.json is intentionally rewritten by the frozen validator and belongs
    # to the output patch, so its immutable source copy remains in the capsule.
    output_names = {"trial.json", "validation.json", "validated_events.json"}
    records = []
    for relative, path in sorted(_regular_tree_files(directory, f"trial input:{directory.name}").items()):
        if path.name in output_names or relative.endswith("/validation.json"):
            continue
        records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return records


def _validation_output_inventory(directory: Path) -> list[dict[str, Any]]:
    allowed = [directory / "trial.json", directory / "validation.json", directory / "validated_events.json"]
    events = _read_json(directory / "validated_events.json")
    if not isinstance(events, list):
        raise IntegrityError(f"{directory.name}: validated_events.json is not a list")
    seen: set[str] = set()
    for event in events:
        snapshot_id = event.get("snapshot_id") if isinstance(event, dict) else None
        if not isinstance(snapshot_id, str) or not SNAPSHOT_ID.fullmatch(snapshot_id) or snapshot_id in seen:
            raise IntegrityError(f"{directory.name}: invalid validated snapshot ID")
        seen.add(snapshot_id)
        allowed.append(directory / "snapshots" / snapshot_id / "validation.json")
    records = []
    for path in allowed:
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
            raise IntegrityError(f"{directory.name}: missing validator output {path.relative_to(directory)}")
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return sorted(records, key=lambda record: record["path"])


def _verify_inventory(directory: Path, records: Any, label: str) -> None:
    parsed = _entries_by_path(records, label)
    for relative, record in parsed.items():
        path = directory.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size != record["size_bytes"] or _sha256_file(path) != record["sha256"]:
            raise IntegrityError(f"{label}: checkpoint hash mismatch for {relative}")


def _cloud_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "PyYAML", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "execution_role": "posthoc_independent_validation_only",
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "jobs": 1,
        "planner_trials_launched": 0,
        "seed_changes": 0,
    }


def _verify_runtime_packages() -> None:
    expected = {"numpy": "1.26.4", "PyYAML": "6.0.1", "psutil": "5.9.0"}
    actual: dict[str, str | None] = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
    if actual != expected:
        raise IntegrityError(f"Cloud validator dependency mismatch: expected={expected}, actual={actual}")


def _checkpoint_template(capsule: VerifiedCapsule, environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "A3_VS_ALL_CLOUD_VALIDATION_CHECKPOINT",
        "capsule_sha256": capsule.sha256,
        "capsule_manifest_sha256": capsule.manifest_sha256,
        "source_at_run_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "jobs": 1,
        "planner_trials_launched": 0,
        "seed_changes": 0,
        "validation_order": capsule.manifest["validation_order"],
        "completed_trial_ids": [],
        "trials": {},
        "control_results": [],
        "cloud_environment": environment,
        "status": "RUNNING",
    }


def _load_checkpoint(path: Path, capsule: VerifiedCapsule) -> dict[str, Any]:
    checkpoint = _read_json(path)
    required = {
        "schema_version",
        "artifact_type",
        "capsule_sha256",
        "capsule_manifest_sha256",
        "source_at_run_sha256",
        "jobs",
        "planner_trials_launched",
        "seed_changes",
        "validation_order",
        "completed_trial_ids",
        "trials",
        "control_results",
        "cloud_environment",
        "status",
    }
    _exact_keys(checkpoint, required, "checkpoint")
    if (
        checkpoint["schema_version"] != 1
        or checkpoint["artifact_type"] != "A3_VS_ALL_CLOUD_VALIDATION_CHECKPOINT"
        or checkpoint["capsule_sha256"] != capsule.sha256
        or checkpoint["capsule_manifest_sha256"] != capsule.manifest_sha256
        or checkpoint["source_at_run_sha256"] != EXPECTED_SOURCE_ARCHIVE_SHA256
        or checkpoint["jobs"] != 1
        or checkpoint["planner_trials_launched"] != 0
        or checkpoint["seed_changes"] != 0
        or checkpoint["validation_order"] != capsule.manifest["validation_order"]
    ):
        raise IntegrityError("checkpoint identity mismatch")
    completed = checkpoint["completed_trial_ids"]
    order = checkpoint["validation_order"]
    if not isinstance(completed, list) or completed != order[: len(completed)] or len(set(completed)) != len(completed):
        raise IntegrityError("checkpoint completion is not an exact validation-order prefix")
    if not isinstance(checkpoint["trials"], dict) or set(checkpoint["trials"]) != set(completed):
        raise IntegrityError("checkpoint trial records do not match completed IDs")
    return checkpoint


def _verify_completed_trials(work_runs: Path, checkpoint: dict[str, Any]) -> None:
    for trial_id in checkpoint["completed_trial_ids"]:
        record = checkpoint["trials"][trial_id]
        if not isinstance(record, dict) or set(record) != {"input_entries", "output_entries", "invalid"}:
            raise IntegrityError(f"invalid checkpoint record for {trial_id}")
        directory = work_runs / trial_id
        _verify_inventory(directory, record["input_entries"], f"checkpoint input:{trial_id}")
        _verify_inventory(directory, record["output_entries"], f"checkpoint output:{trial_id}")
        expected = {
            entry["path"] for entry in record["input_entries"] + record["output_entries"]
        }
        actual = set(_regular_tree_files(directory, f"checkpoint tree:{trial_id}"))
        if actual != expected:
            raise IntegrityError(f"checkpoint file allowlist mismatch for {trial_id}")


def _verify_resume_runs_layout(work_runs: Path, checkpoint: dict[str, Any]) -> None:
    completed = checkpoint["completed_trial_ids"]
    order = checkpoint["validation_order"]
    allowed = set(completed)
    if len(completed) < len(order):
        next_id = order[len(completed)]
        allowed.update((next_id, next_id + ".staging"))
    actual = {path.name for path in work_runs.iterdir()}
    if actual - allowed:
        raise IntegrityError(
            "work/runs contains state outside the completed prefix/current trial: "
            + repr(sorted(actual - allowed))
        )
    for path in work_runs.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise IntegrityError(f"unexpected non-directory run state: {path.name}")


def _run_controls(
    extracted: Path,
    compat: Path,
    capsule: VerifiedCapsule,
    validate_trial,
    Oracle,
    scratch_parent: Path,
) -> list[dict[str, Any]]:
    map_records = _map_records(capsule.manifest)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="control-gate-", dir=scratch_parent) as temporary:
        temporary_root = Path(temporary)
        oracle_by_map: dict[str, Any] = {}
        with _working_directory(compat):
            for trial_id in capsule.manifest["control_trial_ids"]:
                source = extracted / "controls" / trial_id / "inputs"
                run_directory = temporary_root / trial_id
                _copy_input_tree(source, run_directory)
                trial = _read_json(run_directory / "trial.json")
                map_id = trial["map_id"]
                if map_id not in oracle_by_map:
                    original = map_records[map_id]["original_paths"][0]
                    oracle_by_map[map_id] = Oracle(original)
                validate_trial(run_directory, oracle_by_map[map_id])
                results.append(
                    compare_control_outputs(
                        run_directory,
                        extracted / "controls" / trial_id / "expected",
                        trial_id,
                    )
                )
    return results


def _stage_pending_input(extracted: Path, work_runs: Path, trial_id: str) -> Path:
    source = extracted / "pending" / trial_id
    destination = work_runs / trial_id
    if destination.exists():
        # Only an uncheckpointed validation attempt is discarded.  Completed
        # directories are never passed here and are verified separately.
        if destination.is_symlink() or not destination.is_dir():
            raise IntegrityError("refusing to reset a non-directory/symlinked trial state")
        resolved = destination.resolve()
        runs_resolved = work_runs.resolve()
        if runs_resolved not in resolved.parents:
            raise IntegrityError("refusing to reset a trial outside work/runs")
        shutil.rmtree(destination)
    temporary = work_runs / (trial_id + ".staging")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise IntegrityError("refusing to reset a non-directory/symlinked staging state")
        resolved = temporary.resolve()
        if work_runs.resolve() not in resolved.parents:
            raise IntegrityError("unsafe staging path")
        shutil.rmtree(temporary)
    _copy_input_tree(source, temporary)
    os.replace(temporary, destination)
    return destination


def _trial_invalid(directory: Path) -> bool:
    trial = _read_json(directory / "trial.json")
    validation = _read_json(directory / "validation.json")
    return bool(
        validation.get("valid") is not True
        or trial.get("independent_valid") is not True
        or trial.get("snapshots_valid") is not True
        or trial.get("success") is not True
    )


def _pending_map_id(extracted: Path, trial_id: str) -> str:
    trial = _read_json(extracted / "pending" / trial_id / "trial.json")
    return trial["map_id"]


def _write_deterministic_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(members):
                _safe_member_name(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.flag_bits |= 0x800
                archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        # Hard-link publication is an atomic create-if-absent on the same
        # filesystem.  It deliberately fails rather than replacing a patch.
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _build_patch(
    capsule: VerifiedCapsule,
    extracted: Path,
    work_runs: Path,
    checkpoint: dict[str, Any],
    patch_path: Path,
) -> tuple[dict[str, Any], str]:
    trial_ids = capsule.manifest["pending_trial_ids"]
    if set(checkpoint["completed_trial_ids"]) != set(trial_ids) or len(checkpoint["completed_trial_ids"]) != EXPECTED_PENDING_COUNT:
        raise IntegrityError("cannot build an incomplete validation patch")
    members: dict[str, bytes] = {
        "primary_inventory.json": (extracted / "primary_inventory.json").read_bytes(),
        "cloud_environment.json": _json_bytes(checkpoint["cloud_environment"]),
    }
    invalid_ids: list[str] = []
    for trial_id in trial_ids:
        record = checkpoint["trials"][trial_id]
        if record["invalid"]:
            invalid_ids.append(trial_id)
        directory = work_runs / trial_id
        _verify_inventory(directory, record["input_entries"], f"patch input:{trial_id}")
        _verify_inventory(directory, record["output_entries"], f"patch output:{trial_id}")
        for output in record["output_entries"]:
            relative = output["path"]
            member_name = f"trials/{trial_id}/{relative}"
            members[member_name] = directory.joinpath(*PurePosixPath(relative).parts).read_bytes()

    status = "FAILED_CORRECTNESS" if invalid_ids else "COMPLETE"
    summary = {
        "schema_version": 1,
        "status": status,
        "planned_pending": EXPECTED_PENDING_COUNT,
        "validated_pending": EXPECTED_PENDING_COUNT,
        "valid_pending": EXPECTED_PENDING_COUNT - len(invalid_ids),
        "geometric_or_snapshot_invalid": len(invalid_ids),
        "invalid_trial_ids": invalid_ids,
        "controls_passed": len(checkpoint["control_results"]),
        "jobs": 1,
        "planner_trials_launched": 0,
        "seed_changes": 0,
        "classification": "POSTHOC_EXTENDED_VALIDATION_NOT_REGISTERED_PRIMARY",
    }
    members["validation_summary.json"] = _json_bytes(summary)
    entry_list = [
        {"path": name, "size_bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(members.items())
    ]
    primary = capsule.manifest["primary"]
    patch_manifest = {
        "schema_version": 1,
        "artifact_type": PATCH_ARTIFACT,
        "entries": entry_list,
        "trial_ids": trial_ids,
        "validation_order": capsule.manifest["validation_order"],
        "capsule_sha256": capsule.sha256,
        "capsule_manifest_sha256": capsule.manifest_sha256,
        "source_at_run_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "source_primary": {
            "manifest_sha256": EXPECTED_BASELINE_HASHES["manifest.json"],
            "campaign_state_sha256": EXPECTED_BASELINE_HASHES["campaign_state.json"],
            "primary_inventory_sha256": primary["primary_inventory_sha256"],
            "counts": primary["campaign_counts"],
        },
        "jobs": 1,
        "planner_trials_launched": 0,
        "seed_changes": 0,
        "control_results": checkpoint["control_results"],
        "counts": {
            "trial_ids": EXPECTED_PENDING_COUNT,
            "valid": EXPECTED_PENDING_COUNT - len(invalid_ids),
            "invalid": len(invalid_ids),
            "controls": EXPECTED_CONTROL_COUNT,
        },
        "status": status,
    }
    members["patch_manifest.json"] = _json_bytes(patch_manifest)
    _write_deterministic_zip(patch_path, members)
    patch_hash = _sha256_file(patch_path)
    sidecar = Path(str(patch_path) + ".sha256")
    try:
        _exclusive_write(sidecar, f"{patch_hash}  {patch_path.name}\n".encode("ascii"))
    except BaseException:
        # patch_path was created by this invocation with exclusive semantics.
        with contextlib.suppress(OSError):
            patch_path.unlink()
        raise
    return patch_manifest, patch_hash


def run_cloud_validation(
    capsule_path: Path,
    work_dir: Path,
    patch_path: Path,
    *,
    sidecar_path: Path | None = None,
    jobs: int = 1,
    resume: bool = False,
) -> tuple[dict[str, Any], str]:
    if jobs != 1:
        raise IntegrityError("jobs must equal 1 for the registered posthoc validation topology")
    if platform.system() != "Linux":
        raise IntegrityError("pending validation may run only in the recorded Codex Cloud Linux topology")
    _verify_runtime_packages()
    capsule = verify_capsule(capsule_path, sidecar_path)
    work_dir = work_dir.resolve()
    patch_path = patch_path.resolve()
    if patch_path.suffix.lower() != ".zip":
        raise IntegrityError("--patch must end in .zip")
    patch_sidecar = Path(str(patch_path) + ".sha256")
    if patch_path.exists() or patch_sidecar.exists():
        raise IntegrityError("refusing to overwrite an existing patch/sidecar")
    if not resume and work_dir.exists() and any(work_dir.iterdir()):
        raise IntegrityError("fresh run requires an empty --work-dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / "checkpoint.json"
    extraction = work_dir / ("capsule-" + capsule.sha256[:16])
    source_root = work_dir / ("frozen-source-" + EXPECTED_SOURCE_ARCHIVE_SHA256[:16])
    work_runs = work_dir / "runs"
    compat = work_dir / "compat"
    if extraction.exists():
        if not resume:
            raise IntegrityError("work directory already contains capsule state; use --resume")
        # The original ZIP is fully reverified above; verify the persisted extraction too.
        entries = _entries_by_path(capsule.manifest["entries"], "capsule_manifest.entries")
        actual_files = set(_regular_tree_files(extraction, "persisted capsule extraction"))
        if actual_files != set(entries) | {"capsule_manifest.json"}:
            raise IntegrityError("persisted capsule extraction allowlist mismatch")
        for name, entry in entries.items():
            path = extraction.joinpath(*PurePosixPath(name).parts)
            if path.stat().st_size != entry["size_bytes"] or _sha256_file(path) != entry["sha256"]:
                raise IntegrityError(f"persisted capsule extraction changed: {name}")
        if _sha256_file(extraction / "capsule_manifest.json") != capsule.manifest_sha256:
            raise IntegrityError("persisted capsule manifest changed")
    else:
        if resume:
            raise IntegrityError("--resume requested but no extracted capsule exists")
        _extract_verified_archive(capsule, extraction)
    _extract_frozen_source(extraction, source_root)
    _prepare_compatibility_maps(extraction, compat, capsule.manifest)
    validate_trial, Oracle = _load_frozen_validator(source_root)
    work_runs.mkdir(parents=True, exist_ok=True)

    if resume:
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise IntegrityError("--resume requested but checkpoint.json is missing")
        checkpoint = _load_checkpoint(checkpoint_path, capsule)
        _verify_completed_trials(work_runs, checkpoint)
        _verify_resume_runs_layout(work_runs, checkpoint)
    else:
        if checkpoint_path.exists() or any(work_runs.iterdir()):
            raise IntegrityError("fresh run requires an empty work/runs directory")
        checkpoint = _checkpoint_template(capsule, _cloud_environment())
        _atomic_json(checkpoint_path, checkpoint)

    controls = _run_controls(
        extraction,
        compat,
        capsule,
        validate_trial,
        Oracle,
        work_dir,
    )
    checkpoint["control_results"] = controls
    _atomic_json(checkpoint_path, checkpoint)

    map_records = _map_records(capsule.manifest)
    completed = set(checkpoint["completed_trial_ids"])
    current_map_id: str | None = None
    oracle = None
    with _working_directory(compat):
        for trial_id in checkpoint["validation_order"]:
            map_id = _pending_map_id(extraction, trial_id)
            if map_id != current_map_id:
                current_map_id = map_id
                oracle = Oracle(map_records[map_id]["original_paths"][0])
            if trial_id in completed:
                continue
            directory = _stage_pending_input(extraction, work_runs, trial_id)
            input_entries = _input_inventory(directory)
            validate_trial(directory, oracle)
            output_entries = _validation_output_inventory(directory)
            invalid = _trial_invalid(directory)
            checkpoint["completed_trial_ids"].append(trial_id)
            checkpoint["trials"][trial_id] = {
                "input_entries": input_entries,
                "output_entries": output_entries,
                "invalid": invalid,
            }
            _atomic_json(checkpoint_path, checkpoint)
            print(
                f"validated {len(checkpoint['completed_trial_ids'])}/{EXPECTED_PENDING_COUNT} "
                f"{trial_id}: invalid={invalid}",
                flush=True,
            )

    invalid_count = sum(bool(record["invalid"]) for record in checkpoint["trials"].values())
    checkpoint["status"] = "FAILED_CORRECTNESS" if invalid_count else "COMPLETE"
    _atomic_json(checkpoint_path, checkpoint)
    return _build_patch(capsule, extraction, work_runs, checkpoint, patch_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the 135 pending A3-vs-all trials from an immutable cloud capsule."
    )
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--sha256", type=Path, help="capsule SHA-256 sidecar; default: <capsule>.sha256")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, patch_hash = run_cloud_validation(
            args.capsule,
            args.work_dir,
            args.patch,
            sidecar_path=args.sha256,
            jobs=args.jobs,
            resume=args.resume,
        )
    except (IntegrityError, ControlMismatch, OSError, ValueError, ImportError, zipfile.BadZipFile) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "patch": str(args.patch.resolve()),
                "patch_sha256": patch_hash,
                "validated": manifest["counts"]["trial_ids"],
                "invalid": manifest["counts"]["invalid"],
                "planner_trials_launched": 0,
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 2 if manifest["status"] == "FAILED_CORRECTNESS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
