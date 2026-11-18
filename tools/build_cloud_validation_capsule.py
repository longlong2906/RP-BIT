"""Build the immutable, portable input capsule for Cloud post-hoc validation.

This tool is intentionally specific to the interrupted ``a3_vs_all_v1_20260930``
campaign.  It never edits that campaign and it never starts a planner.  The
archive contract is versioned here so the Cloud runner and local patch applier
can fail closed when the layout changes.

Capsule schema v1
-----------------
``capsule_manifest.json`` describes every other archive member (a manifest
cannot contain its own byte hash without a self-reference).  Its ``entries``
records are sorted and have exactly ``path``, ``size_bytes`` and ``sha256``.
The remaining roots are::

    primary_inventory.json
    metadata/{manifest.json,campaign_state.json,map_manifest.json,
              resolved_config.json,source_manifest.json,source_at_run.zip}
    maps/<map_id>.t25d
    pending/<trial_id>/{request.json,trial.json,progress.jsonl,
                        path_xy.csv,path_surface.csv,
                        snapshots/<snapshot_id>/{path_xy.csv,path_surface.csv}}
    controls/<trial_id>/inputs/<same strict input layout>
    controls/<trial_id>/expected/{trial.json,validation.json,
                                  validated_events.json,
                                  snapshots/<snapshot_id>/validation.json}

Pending validation outputs, including partially written ones, are deliberately
excluded by the input allowlist.  The exact original primary tree is still
committed by ``primary_inventory.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Iterable
import zipfile

from experiment_common import digest, file_hash, read_json


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "A3_VS_ALL_CLOUD_VALIDATION_CAPSULE"
CAMPAIGN_BASENAME = "a3_vs_all_v1_20260930"
PROTOCOL_ID = "a3_vs_all_v1"
PRIMARY_INVENTORY_POLICY = "all_regular_files_exact"
ENTRY_INVENTORY_POLICY = "all_members_except_capsule_manifest_due_to_self_reference"

ENTRY_RECORD_KEYS = frozenset({"path", "size_bytes", "sha256"})
MAP_RECORD_KEYS = frozenset({"map_id", "archive_path", "sha256", "original_paths"})
CAPSULE_MANIFEST_KEYS = frozenset(
    {
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
)

BASELINE_HASHES = {
    "manifest.json": "c474c669a510dc4a6feebe99f55b6b892ce131bf4e85d581a20e456742644f14",
    "map_manifest.json": "fc8fb97cf9db0c36a6c191c3f66273469afdf058fb995bd9002886f0f25a6d2a",
    "resolved_config.json": "bbcf830fb34385cdeebfb78bbd9b2c21931d64194578e51c3c3a2ba79210831b",
    "source_at_run.zip": "98a4656c6f46e8102af6b564c221d7cf3d11f5f7e11c838cca3f5377015b0580",
}

EXPECTED_COUNTS = {
    "planned": 250,
    "executed": 250,
    "not_run": 0,
    "offline_validated": 115,
    "validation_pending": 135,
    "validated_success": 115,
    "geometric_invalid": 0,
}

PLANNER_IDS = frozenset(
    {
        "bitstar_a3",
        "abitstar",
        "aitstar",
        "lazyprmstar_hauser2015",
        "sem_rrtstar_common_v1",
    }
)

CONTROL_TRIAL_IDS = (
    "mixed_obstacles_5005_q0_sem_rrtstar_common_v1_s93001",
    "narrow_passage_5009_q0_abitstar_s93001",
    "ridge_pass_5004_q0_aitstar_s93001",
    "rolling_terrain_5008_q0_bitstar_a3_s93001",
    "traversable_hill_5005_q0_lazyprmstar_hauser2015_s93001",
)

METADATA_FILES = (
    "manifest.json",
    "campaign_state.json",
    "map_manifest.json",
    "resolved_config.json",
    "source_manifest.json",
    "source_at_run.zip",
)
TRIAL_INPUT_FILES = (
    "request.json",
    "trial.json",
    "progress.jsonl",
    "path_xy.csv",
    "path_surface.csv",
)
CONTROL_EXPECTED_FILES = ("trial.json", "validation.json", "validated_events.json")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class CapsuleError(ValueError):
    """The source campaign does not satisfy the locked capsule contract."""


@dataclass(frozen=True)
class ArchiveItem:
    archive_path: str
    sha256: str
    size_bytes: int
    source: Path | None = None
    data: bytes | None = None

    def entry(self) -> dict[str, Any]:
        return {
            "path": self.archive_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_component(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise CapsuleError(f"invalid {label}: {value!r}")
    if "/" in value or "\\" in value or ":" in value or "\0" in value:
        raise CapsuleError(f"unsafe {label}: {value!r}")
    return value


def _safe_archive_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise CapsuleError(f"unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CapsuleError(f"unsafe archive path: {value!r}")
    normalized = path.as_posix()
    if normalized != value or any(":" in part for part in path.parts):
        raise CapsuleError(f"non-canonical archive path: {value!r}")
    return normalized


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise CapsuleError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise CapsuleError(f"missing regular {label}: {path}")
    return path


def _strict_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise CapsuleError(f"invalid JSON {path}: {error}") from error


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_source_roots(campaign: Path, maps: Path, output: Path) -> tuple[Path, Path, Path]:
    if campaign.is_symlink() or maps.is_symlink():
        raise CapsuleError("campaign/maps roots must not be symlinks")
    campaign = campaign.resolve()
    maps = maps.resolve()
    output = output.resolve(strict=False)
    if not campaign.is_dir() or campaign.name != CAMPAIGN_BASENAME:
        raise CapsuleError(f"campaign must be the locked {CAMPAIGN_BASENAME} directory")
    if not maps.is_dir():
        raise CapsuleError("--maps must name the map-data directory")
    if output.suffix.lower() != ".zip":
        raise CapsuleError("--output must end in .zip")
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("output capsule or SHA-256 sidecar already exists")
    if _inside(output, campaign) or _inside(sidecar, campaign):
        raise CapsuleError("output must be outside the immutable primary campaign")
    if _inside(output, maps) or _inside(sidecar, maps):
        raise CapsuleError("output must be outside the map-data directory")
    return campaign, maps, output


def build_primary_inventory(campaign: Path) -> dict[str, Any]:
    """Hash every regular file below *campaign* and reject every symlink."""
    entries: list[dict[str, Any]] = []
    for path in sorted(campaign.rglob("*"), key=lambda item: item.relative_to(campaign).as_posix()):
        relative = path.relative_to(campaign).as_posix()
        if path.is_symlink():
            raise CapsuleError(f"primary inventory forbids symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CapsuleError(f"primary inventory found a non-regular entry: {relative}")
        _safe_archive_path(relative)
        entries.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": file_hash(path)}
        )
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "policy": PRIMARY_INVENTORY_POLICY,
        "entries": entries,
        "inventory_sha256": digest(entries),
    }
    return inventory


def _is_validation_terminal(directory: Path, trial: dict[str, Any]) -> bool:
    if trial.get("validation_status") == "NO_REQUEST_INTERRUPTED_OR_BLOCKED":
        return True
    wall = trial.get("validation_wall_s")
    return (
        (directory / "validation.json").is_file()
        and isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(wall)
        and wall >= 0.0
    )


def _computed_campaign_counts(
    campaign: Path, manifest: dict[str, Any]
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    executed = validated = success = invalid = 0
    pending: list[dict[str, Any]] = []
    trial_records: dict[str, dict[str, Any]] = {}
    trials = manifest.get("trials")
    if not isinstance(trials, list):
        raise CapsuleError("manifest.trials must be a list")
    seen: set[str] = set()
    for entry in trials:
        if not isinstance(entry, dict):
            raise CapsuleError("manifest contains a non-object trial")
        trial_id = _safe_component(entry.get("trial_id"), "trial_id")
        if trial_id in seen:
            raise CapsuleError(f"duplicate trial_id: {trial_id}")
        seen.add(trial_id)
        directory = campaign / trial_id
        trial = _strict_json(_regular_file(directory / "trial.json", "trial.json"))
        if not isinstance(trial, dict) or trial.get("trial_id") != trial_id:
            raise CapsuleError(f"trial identity mismatch: {trial_id}")
        trial_records[trial_id] = trial
        executed_trial = trial.get("execution_status") not in {"BLOCKED", "NOT_RUN", "PLANNED"}
        executed += int(executed_trial)
        terminal = _is_validation_terminal(directory, trial)
        if terminal and executed_trial:
            validated += 1
            validation_path = directory / "validation.json"
            validation = _strict_json(validation_path) if validation_path.is_file() else {}
            invalid += int(
                bool(
                    (trial.get("path_available") and not validation.get("valid"))
                    or trial.get("snapshots_valid") is False
                )
            )
        elif executed_trial:
            pending.append(entry)
        success += int(bool(executed_trial and terminal and trial.get("success")))
    counts = {
        "planned": len(trials),
        "executed": executed,
        "not_run": len(trials) - executed,
        "offline_validated": validated,
        "validation_pending": executed - validated,
        "validated_success": success,
        "geometric_invalid": invalid,
    }
    return counts, pending, trial_records


def _verify_campaign(
    campaign: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    for name, expected in BASELINE_HASHES.items():
        path = _regular_file(campaign / name, name)
        actual = file_hash(path)
        if actual != expected:
            raise CapsuleError(f"baseline hash mismatch for {name}: {actual} != {expected}")

    manifest = _strict_json(campaign / "manifest.json")
    state = _strict_json(campaign / "campaign_state.json")
    map_manifest = _strict_json(campaign / "map_manifest.json")
    source_manifest = _strict_json(_regular_file(campaign / "source_manifest.json", "source_manifest.json"))
    if not all(isinstance(item, dict) for item in (manifest, state, map_manifest, source_manifest)):
        raise CapsuleError("campaign metadata roots must be JSON objects")
    if manifest.get("experiment_id") != CAMPAIGN_BASENAME:
        raise CapsuleError("manifest experiment_id does not identify the locked campaign")
    if manifest.get("profile") != "combined" or manifest.get("planned_trials") != 250:
        raise CapsuleError("manifest is not the locked 250-trial combined campaign")
    if state.get("protocol_id") != PROTOCOL_ID:
        raise CapsuleError("campaign_state protocol mismatch")
    if state.get("status") != "PARTIAL" or state.get("phase") != "INTERRUPTED":
        raise CapsuleError("campaign must remain at the recorded PARTIAL/INTERRUPTED pause point")
    if source_manifest.get("archive_sha256") != BASELINE_HASHES["source_at_run.zip"]:
        raise CapsuleError("source_manifest does not bind the frozen source archive")

    counts, pending, records = _computed_campaign_counts(campaign, manifest)
    if counts != EXPECTED_COUNTS:
        raise CapsuleError(f"computed campaign counts mismatch: {counts!r}")
    if state.get("counts") != EXPECTED_COUNTS:
        raise CapsuleError("campaign_state counts differ from the locked pause point")
    if manifest.get("counts", {}).get("planned") != 250:
        raise CapsuleError("manifest counts no longer describe 250 planned trials")

    pending_by_slot: dict[str, list[dict[str, Any]]] = {}
    for entry in pending:
        slot = _safe_component(entry.get("problem_slot_id"), "problem_slot_id")
        pending_by_slot.setdefault(slot, []).append(entry)
    if len(pending_by_slot) != 27:
        raise CapsuleError("pending trials must form exactly 27 problem blocks")
    for slot, entries in pending_by_slot.items():
        if len(entries) != 5 or {entry.get("planner_id") for entry in entries} != PLANNER_IDS:
            raise CapsuleError(f"pending block is not one trial per five planners: {slot}")
        if len({entry.get("map_id") for entry in entries}) != 1:
            raise CapsuleError(f"pending block mixes maps: {slot}")
    if len({entry.get("map_id") for entry in pending}) != 27:
        raise CapsuleError("pending blocks must correspond to 27 distinct maps")

    known_ids = set(records)
    if any(control not in known_ids for control in CONTROL_TRIAL_IDS):
        raise CapsuleError("one or more fixed control trials are absent")
    for control in CONTROL_TRIAL_IDS:
        if control in {entry["trial_id"] for entry in pending}:
            raise CapsuleError(f"fixed control is not independently validated: {control}")
        if not records[control].get("success") or records[control].get("independent_valid") is not True:
            raise CapsuleError(f"fixed control is not a validated success: {control}")
    return manifest, state, map_manifest, pending, records


def _progress_snapshot_ids(progress_path: Path) -> list[str]:
    snapshot_ids: list[str] = []
    seen: set[str] = set()
    try:
        lines = progress_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CapsuleError(f"cannot read progress file {progress_path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("nonfinite progress value: " + value)
                ),
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise CapsuleError(f"invalid progress event {progress_path}:{line_number}: {error}") from error
        if not isinstance(event, dict):
            raise CapsuleError(f"progress event is not an object: {progress_path}:{line_number}")
        if event.get("event_type") != "improvement":
            continue
        snapshot_id = _safe_component(event.get("snapshot_id"), "snapshot_id")
        if snapshot_id in seen:
            raise CapsuleError(f"duplicate snapshot_id in {progress_path}: {snapshot_id}")
        seen.add(snapshot_id)
        snapshot_ids.append(snapshot_id)
    if not snapshot_ids:
        raise CapsuleError(f"exact-solution trial has no improvement snapshots: {progress_path.parent.name}")
    return snapshot_ids


def _source_item(archive_path: str, source: Path, known_hash: str | None = None) -> ArchiveItem:
    archive_path = _safe_archive_path(archive_path)
    source = _regular_file(source, archive_path)
    actual = file_hash(source)
    if known_hash is not None and actual != known_hash:
        raise CapsuleError(f"source changed while collecting {archive_path}")
    return ArchiveItem(archive_path, actual, source.stat().st_size, source=source)


def _bytes_item(archive_path: str, data: bytes) -> ArchiveItem:
    archive_path = _safe_archive_path(archive_path)
    return ArchiveItem(archive_path, _sha256_bytes(data), len(data), data=data)


def _add_item(items: dict[str, ArchiveItem], item: ArchiveItem) -> None:
    if item.archive_path == "capsule_manifest.json" or item.archive_path in items:
        raise CapsuleError(f"duplicate/reserved archive path: {item.archive_path}")
    items[item.archive_path] = item


def _validate_trial_input_identity(
    entry: dict[str, Any], trial: dict[str, Any], request: dict[str, Any]
) -> None:
    trial_id = entry["trial_id"]
    for field in ("trial_id", "map_id", "planner_id"):
        if trial.get(field) != entry.get(field):
            raise CapsuleError(f"{trial_id}: {field} differs between manifest and trial")
    if request.get("planner_id") != entry.get("planner_id"):
        raise CapsuleError(f"{trial_id}: planner_id differs between request and manifest")
    if trial.get("map_path") != request.get("map_path"):
        raise CapsuleError(f"{trial_id}: map_path differs between trial and request")
    if trial.get("map_hash") != entry.get("map_hash"):
        raise CapsuleError(f"{trial_id}: map hash differs between trial and manifest")
    if trial.get("execution_status") != "COMPLETED" or trial.get("planner_status") != "EXACT_SOLUTION":
        raise CapsuleError(f"{trial_id}: capsule only supports completed exact-solution inputs")
    if trial.get("path_available") is not True:
        raise CapsuleError(f"{trial_id}: exact path is unavailable")


def _collect_trial_inputs(
    items: dict[str, ArchiveItem],
    campaign: Path,
    entry: dict[str, Any],
    prefix: str,
    primary_hashes: dict[str, str],
) -> tuple[list[str], str]:
    trial_id = _safe_component(entry.get("trial_id"), "trial_id")
    directory = campaign / trial_id
    trial = _strict_json(directory / "trial.json")
    request = _strict_json(directory / "request.json")
    if not isinstance(trial, dict) or not isinstance(request, dict):
        raise CapsuleError(f"{trial_id}: request/trial must be JSON objects")
    _validate_trial_input_identity(entry, trial, request)
    for name in TRIAL_INPUT_FILES:
        relative = f"{trial_id}/{name}"
        source = directory / name
        _add_item(
            items,
            _source_item(
                f"{prefix}/{name}", source, primary_hashes.get(relative)
            ),
        )
    snapshot_ids = _progress_snapshot_ids(directory / "progress.jsonl")
    actual_snapshot_dirs = {
        path.name for path in (directory / "snapshots").iterdir() if path.is_dir() and not path.is_symlink()
    }
    if actual_snapshot_dirs != set(snapshot_ids):
        raise CapsuleError(f"{trial_id}: snapshot directories differ from improvement events")
    for snapshot_id in snapshot_ids:
        for name in ("path_xy.csv", "path_surface.csv"):
            relative = f"{trial_id}/snapshots/{snapshot_id}/{name}"
            source = directory / "snapshots" / snapshot_id / name
            _add_item(
                items,
                _source_item(
                    f"{prefix}/snapshots/{snapshot_id}/{name}",
                    source,
                    primary_hashes.get(relative),
                ),
            )
    return snapshot_ids, request["map_path"]


def _collect_control_expected(
    items: dict[str, ArchiveItem],
    campaign: Path,
    trial_id: str,
    snapshot_ids: Iterable[str],
    primary_hashes: dict[str, str],
) -> None:
    directory = campaign / trial_id
    prefix = f"controls/{trial_id}/expected"
    for name in CONTROL_EXPECTED_FILES:
        relative = f"{trial_id}/{name}"
        _add_item(
            items,
            _source_item(f"{prefix}/{name}", directory / name, primary_hashes.get(relative)),
        )
    expected_events = _strict_json(directory / "validated_events.json")
    if not isinstance(expected_events, list):
        raise CapsuleError(f"{trial_id}: validated_events.json must be a list")
    event_snapshot_ids = [
        _safe_component(event.get("snapshot_id"), "validated snapshot_id")
        for event in expected_events
        if isinstance(event, dict) and event.get("event_type") == "improvement"
    ]
    if event_snapshot_ids != list(snapshot_ids):
        raise CapsuleError(f"{trial_id}: expected validated events do not cover input snapshots")
    for snapshot_id in snapshot_ids:
        relative = f"{trial_id}/snapshots/{snapshot_id}/validation.json"
        _add_item(
            items,
            _source_item(
                f"{prefix}/snapshots/{snapshot_id}/validation.json",
                directory / "snapshots" / snapshot_id / "validation.json",
                primary_hashes.get(relative),
            ),
        )


def _map_index(map_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    maps = map_manifest.get("maps")
    if not isinstance(maps, list):
        raise CapsuleError("map_manifest.maps must be a list")
    index: dict[str, dict[str, Any]] = {}
    for entry in maps:
        if not isinstance(entry, dict):
            raise CapsuleError("map_manifest contains a non-object map")
        map_id = _safe_component(entry.get("map_id"), "map_id")
        if map_id in index:
            raise CapsuleError(f"duplicate map_id in map manifest: {map_id}")
        index[map_id] = entry
    return index


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_archive_path(name), date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(path: Path, manifest_bytes: bytes, items: list[ArchiveItem]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = b""
        archive.writestr(_zip_info("capsule_manifest.json"), manifest_bytes, compresslevel=9)
        for item in items:
            if item.data is not None:
                data = item.data
            elif item.source is not None:
                _regular_file(item.source, item.archive_path)
                data = item.source.read_bytes()
            else:
                raise AssertionError("archive item has neither bytes nor source")
            if len(data) != item.size_bytes or _sha256_bytes(data) != item.sha256:
                raise CapsuleError(f"source changed while writing {item.archive_path}")
            archive.writestr(_zip_info(item.archive_path), data, compresslevel=9)


def _verify_zip(path: Path, manifest: dict[str, Any]) -> None:
    expected = ["capsule_manifest.json", *[entry["path"] for entry in manifest["entries"]]]
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)):
            raise CapsuleError("written ZIP member order/layout mismatch")
        actual_manifest = json.loads(archive.read("capsule_manifest.json").decode("utf-8"))
        if actual_manifest != manifest or set(actual_manifest) != CAPSULE_MANIFEST_KEYS:
            raise CapsuleError("written capsule manifest mismatch")
        for record in manifest["entries"]:
            if set(record) != ENTRY_RECORD_KEYS:
                raise CapsuleError("capsule entry record schema mismatch")
            data = archive.read(record["path"])
            if len(data) != record["size_bytes"] or _sha256_bytes(data) != record["sha256"]:
                raise CapsuleError(f"written ZIP member hash mismatch: {record['path']}")


def _exclusive_copy(source: Path, destination: Path) -> None:
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(descriptor, "wb") as output_handle, source.open("rb") as input_handle:
            descriptor = None
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            destination.unlink(missing_ok=True)
        raise


def _publish(temp_zip: Path, output: Path) -> str:
    capsule_sha256 = file_hash(temp_zip)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar_data = f"{capsule_sha256}  {output.name}\n".encode("ascii")
    sidecar_temp: Path | None = None
    published_output = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=sidecar.name + ".", suffix=".tmp", delete=False
        ) as handle:
            handle.write(sidecar_data)
            handle.flush()
            os.fsync(handle.fileno())
            sidecar_temp = Path(handle.name)
        _exclusive_copy(temp_zip, output)
        published_output = True
        _exclusive_copy(sidecar_temp, sidecar)
    except Exception:
        if published_output:
            output.unlink(missing_ok=True)
        raise
    finally:
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)
    return capsule_sha256


def build_capsule(campaign: Path | str, maps: Path | str, output: Path | str) -> dict[str, Any]:
    campaign, maps, output = _assert_source_roots(Path(campaign), Path(maps), Path(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    primary_inventory = build_primary_inventory(campaign)
    primary_hashes = {record["path"]: record["sha256"] for record in primary_inventory["entries"]}
    manifest, state, map_manifest, pending, trial_records = _verify_campaign(campaign)
    trial_index = {entry["trial_id"]: entry for entry in manifest["trials"]}
    map_index = _map_index(map_manifest)

    items: dict[str, ArchiveItem] = {}
    primary_inventory_bytes = _json_bytes(primary_inventory)
    _add_item(items, _bytes_item("primary_inventory.json", primary_inventory_bytes))
    for name in METADATA_FILES:
        _add_item(
            items,
            _source_item(
                f"metadata/{name}", campaign / name, primary_hashes.get(name)
            ),
        )

    original_map_paths: dict[str, set[str]] = {}
    selected_map_ids: set[str] = set()
    pending_trial_ids = [entry["trial_id"] for entry in pending]
    for entry in pending:
        trial_id = entry["trial_id"]
        snapshot_ids, original_path = _collect_trial_inputs(
            items, campaign, entry, f"pending/{trial_id}", primary_hashes
        )
        del snapshot_ids
        map_id = _safe_component(entry.get("map_id"), "map_id")
        selected_map_ids.add(map_id)
        original_map_paths.setdefault(map_id, set()).update(
            path for path in (entry.get("map_path"), original_path) if isinstance(path, str)
        )

    for trial_id in CONTROL_TRIAL_IDS:
        entry = trial_index[trial_id]
        snapshot_ids, original_path = _collect_trial_inputs(
            items, campaign, entry, f"controls/{trial_id}/inputs", primary_hashes
        )
        _collect_control_expected(items, campaign, trial_id, snapshot_ids, primary_hashes)
        map_id = _safe_component(entry.get("map_id"), "map_id")
        selected_map_ids.add(map_id)
        original_map_paths.setdefault(map_id, set()).update(
            path for path in (entry.get("map_path"), original_path) if isinstance(path, str)
        )

    map_records: list[dict[str, Any]] = []
    manifest_identity_hashes = manifest.get("identity", {}).get("map_hashes", {})
    for map_id in sorted(selected_map_ids):
        if map_id not in map_index:
            raise CapsuleError(f"selected map absent from map manifest: {map_id}")
        map_entry = map_index[map_id]
        expected_hash = map_entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise CapsuleError(f"invalid map hash in map manifest: {map_id}")
        if manifest_identity_hashes.get(map_id) != expected_hash:
            raise CapsuleError(f"manifest identity map hash mismatch: {map_id}")
        map_path = maps / f"{map_id}.t25d"
        archive_path = f"maps/{map_id}.t25d"
        item = _source_item(archive_path, map_path, expected_hash)
        _add_item(items, item)
        original_map_paths.setdefault(map_id, set()).add(str(map_entry.get("path")))
        for trial_entry in manifest["trials"]:
            if trial_entry.get("map_id") == map_id:
                if trial_entry.get("map_hash") != expected_hash:
                    raise CapsuleError(f"trial map hash mismatch: {trial_entry['trial_id']}")
                if isinstance(trial_entry.get("map_path"), str):
                    original_map_paths[map_id].add(trial_entry["map_path"])
        map_record = {
            "map_id": map_id,
            "archive_path": archive_path,
            "sha256": expected_hash,
            "original_paths": sorted(path for path in original_map_paths[map_id] if path and path != "None"),
        }
        if set(map_record) != MAP_RECORD_KEYS:
            raise AssertionError("internal map record schema drift")
        map_records.append(map_record)

    ordered_items = [items[name] for name in sorted(items)]
    entries = [item.entry() for item in ordered_items]
    capsule_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "entry_inventory_policy": ENTRY_INVENTORY_POLICY,
        "entries": entries,
        "primary": {
            "experiment_id": manifest["experiment_id"],
            "protocol_id": state["protocol_id"],
            "status": state["status"],
            "phase": state["phase"],
            "identity": manifest.get("identity"),
            "manifest_counts": manifest.get("counts"),
            "campaign_counts": EXPECTED_COUNTS,
            "baseline_hashes": dict(BASELINE_HASHES),
            "primary_inventory_sha256": primary_inventory["inventory_sha256"],
        },
        "pending_trial_ids": pending_trial_ids,
        "validation_order": list(pending_trial_ids),
        "control_trial_ids": list(CONTROL_TRIAL_IDS),
        "maps": map_records,
        "file_contract": {
            "pending_validation_outputs": "excluded_even_if_partially_present_in_primary",
            "pending_input_files": list(TRIAL_INPUT_FILES),
            "snapshot_input_files": ["path_xy.csv", "path_surface.csv"],
            "control_expected_files": list(CONTROL_EXPECTED_FILES),
            "control_snapshot_expected_files": ["validation.json"],
            "map_resolution": "map_id_and_sha256_never_embedded_host_path",
        },
    }
    if set(capsule_manifest) != CAPSULE_MANIFEST_KEYS:
        raise AssertionError("internal capsule manifest schema drift")
    manifest_bytes = _json_bytes(capsule_manifest)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        _write_zip(temporary, manifest_bytes, ordered_items)
        _verify_zip(temporary, capsule_manifest)
        if build_primary_inventory(campaign) != primary_inventory:
            raise CapsuleError("primary campaign changed while capsule was being built")
        capsule_sha256 = _publish(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "output": str(output),
        "sha256_sidecar": str(output.with_name(output.name + ".sha256")),
        "capsule_sha256": capsule_sha256,
        "archive_entries_excluding_manifest": len(entries),
        "pending_trials": len(pending_trial_ids),
        "control_trials": len(CONTROL_TRIAL_IDS),
        "maps": len(map_records),
        "primary_inventory_sha256": primary_inventory["inventory_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_capsule(args.campaign, args.maps, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
