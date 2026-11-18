"""Apply a verified Codex Cloud validation patch to a new derivative campaign.

The paused primary campaign is an immutable input.  This command verifies the
pause-point inventory carried by the patch, copies the primary into a staging
tree, applies only independent-validation outputs, and runs the locked generic
analysis.  It never launches a planner and never writes into the primary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from experiment_common import digest, dump, file_hash, read_json  # noqa: E402
from run_advanced_campaign import (  # noqa: E402
    CampaignClock,
    _report_artifacts,
    campaign_counts,
    campaign_descriptor,
    run_analysis_with_deadline,
    validation_is_terminal,
    verify_raw_artifacts,
)
from run_experiments import refresh_manifest, utc  # noqa: E402


PATCH_TYPE = "A3_VS_ALL_CLOUD_VALIDATION_PATCH"
CLASSIFICATION = "POSTHOC_EXTENDED_VALIDATION_NOT_REGISTERED_PRIMARY"
EXPECTED_TRIALS = 135
EXPECTED_SOURCE_AT_RUN_SHA256 = "98a4656c6f46e8102af6b564c221d7cf3d11f5f7e11c838cca3f5377015b0580"
EXPECTED_CONTROL_TRIAL_IDS = [
    "mixed_obstacles_5005_q0_sem_rrtstar_common_v1_s93001",
    "narrow_passage_5009_q0_abitstar_s93001",
    "ridge_pass_5004_q0_aitstar_s93001",
    "rolling_terrain_5008_q0_bitstar_a3_s93001",
    "traversable_hill_5005_q0_lazyprmstar_hauser2015_s93001",
]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_PATCH_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_PATCH_MEMBER_BYTES = 160 * 1024 * 1024
REQUIRED_ROOT_MEMBERS = {
    "patch_manifest.json",
    "primary_inventory.json",
    "cloud_environment.json",
    "validation_summary.json",
}
TRIAL_VALIDATION_FIELDS = {
    "independent_valid",
    "success",
    "independently_recomputed_cost_m",
    "length_xy_m",
    "cost_abs_error_m",
    "minimum_footprint_clearance_m",
    "maximum_slope_deg",
    "validation_wall_s",
    "snapshots_valid",
}
PATCH_MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "entries",
    "trial_ids",
    "validation_order",
    "capsule_sha256",
    "capsule_manifest_sha256",
    "source_at_run_sha256",
    "source_primary",
    "jobs",
    "planner_trials_launched",
    "seed_changes",
    "control_results",
    "counts",
    "status",
}
SUMMARY_KEYS = {
    "schema_version",
    "status",
    "planned_pending",
    "validated_pending",
    "valid_pending",
    "geometric_or_snapshot_invalid",
    "invalid_trial_ids",
    "controls_passed",
    "jobs",
    "planner_trials_launched",
    "seed_changes",
    "classification",
}
FROZEN_LOCAL_TOOLCHAIN = (
    "tools/experiment_common.py",
    "tools/run_experiments.py",
    "tools/run_advanced_campaign.py",
    "tools/analyze_results.py",
    "tools/independent_verifier.py",
    "tools/validate_results.py",
)


def _strict_json_bytes(payload: bytes, label: str):
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite JSON constant: " + value)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {label}: {error}") from error


def _sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _verify_frozen_local_toolchain(primary: Path):
    source_manifest_path = primary / "source_manifest.json"
    source_manifest = _strict_json_bytes(source_manifest_path.read_bytes(), str(source_manifest_path))
    if not isinstance(source_manifest, dict) or not isinstance(source_manifest.get("files"), dict):
        raise ValueError("primary source manifest is malformed")
    normalized = {name.replace("\\", "/"): value for name, value in source_manifest["files"].items()}
    verified = {}
    for relative in FROZEN_LOCAL_TOOLCHAIN:
        expected = normalized.get(relative)
        path = ROOT.joinpath(*PurePosixPath(relative).parts)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError("frozen source manifest lacks local tool: " + relative)
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("local integration/analysis tool is not a regular file: " + relative)
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError("local integration/analysis tool drifted from source_at_run: " + relative)
        verified[relative] = actual
    return verified


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("unsafe ZIP member name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("unsafe ZIP member path: " + name)
    canonical = path.as_posix()
    if canonical != name or canonical.endswith("/"):
        raise ValueError("non-canonical ZIP member path: " + name)
    return canonical


def _zip_members(archive: zipfile.ZipFile):
    records = {}
    collision_keys = set()
    total = 0
    if len(archive.infolist()) > 100_000:
        raise ValueError("patch contains too many ZIP members")
    for info in archive.infolist():
        name = _safe_member_name(info.filename)
        collision = name.casefold()
        if name in records or collision in collision_keys:
            raise ValueError("duplicate/colliding ZIP member: " + name)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ValueError("ZIP symlink forbidden: " + name)
        kind = stat.S_IFMT(mode)
        if info.is_dir() or kind not in (0, stat.S_IFREG):
            raise ValueError("ZIP member must be a regular file: " + name)
        if info.file_size < 0 or info.file_size > MAX_PATCH_MEMBER_BYTES:
            raise ValueError("oversized ZIP member: " + name)
        total += info.file_size
        if total > MAX_PATCH_UNCOMPRESSED_BYTES:
            raise ValueError("patch exceeds uncompressed size limit")
        records[name] = info
        collision_keys.add(collision)
    return records


def _sidecar_hash(patch: Path, sidecar: Path | None) -> str:
    path = sidecar or Path(str(patch) + ".sha256")
    if not path.is_file():
        raise ValueError("patch SHA-256 sidecar is absent: " + str(path))
    tokens = path.read_text(encoding="ascii").strip().split()
    if not tokens or len(tokens) > 2 or not HEX64.fullmatch(tokens[0]):
        raise ValueError("malformed patch SHA-256 sidecar")
    if len(tokens) == 2 and tokens[1].lstrip("*") != patch.name:
        raise ValueError("patch SHA-256 sidecar filename mismatch")
    expected = tokens[0].lower()
    actual = file_hash(patch)
    if actual != expected:
        raise ValueError("patch archive SHA-256 mismatch")
    return actual


def _validate_entry_table(manifest: dict, members: dict[str, zipfile.ZipInfo]):
    entries = manifest.get("entries")
    if not isinstance(entries, list) or entries != sorted(entries, key=lambda item: item.get("path", "")):
        raise ValueError("patch entries must be a sorted list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("invalid patch entry schema")
        name = _safe_member_name(entry["path"])
        if name == "patch_manifest.json" or name in seen:
            raise ValueError("invalid or duplicate patch entry: " + name)
        if name not in members:
            raise ValueError("patch entry is absent from ZIP: " + name)
        if entry["size_bytes"] != members[name].file_size:
            raise ValueError("patch entry size mismatch: " + name)
        if not isinstance(entry["size_bytes"], int) or isinstance(entry["size_bytes"], bool) or entry["size_bytes"] < 0:
            raise ValueError("patch entry size is malformed: " + name)
        if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
            raise ValueError("patch entry hash is malformed: " + name)
        seen.add(name)
    expected = set(members) - {"patch_manifest.json"}
    if seen != expected:
        raise ValueError("patch entry table does not exactly cover ZIP members")


def _allowed_trial_member(name: str, trial_ids: set[str]) -> bool:
    parts = PurePosixPath(name).parts
    if len(parts) < 3 or parts[0] != "trials" or parts[1] not in trial_ids:
        return False
    relative = parts[2:]
    if relative in (("trial.json",), ("validation.json",), ("validated_events.json",)):
        return True
    return (
        len(relative) == 3
        and relative[0] == "snapshots"
        and relative[1].startswith("snapshot_")
        and relative[2] == "validation.json"
        and relative[1][len("snapshot_"):].isdigit()
    )


def _validate_patch_metadata(manifest: dict, summary: dict, environment: dict):
    if not isinstance(manifest, dict) or set(manifest) != PATCH_MANIFEST_KEYS:
        raise ValueError("patch manifest has missing or unknown fields")
    if manifest["schema_version"] != 1 or manifest["artifact_type"] != PATCH_TYPE:
        raise ValueError("unsupported Cloud validation patch schema")
    if manifest["jobs"] != 1 or manifest["planner_trials_launched"] != 0 or manifest["seed_changes"] != 0:
        raise ValueError("patch is not a jobs=1 validation-only artifact")
    for key in ("capsule_sha256", "capsule_manifest_sha256", "source_at_run_sha256"):
        if not isinstance(manifest[key], str) or not HEX64.fullmatch(manifest[key]):
            raise ValueError("patch has malformed provenance hash: " + key)
    if manifest["source_at_run_sha256"] != EXPECTED_SOURCE_AT_RUN_SHA256:
        raise ValueError("patch used an unexpected frozen source archive")

    trial_ids = manifest["trial_ids"]
    order = manifest["validation_order"]
    if (
        not isinstance(trial_ids, list)
        or len(trial_ids) != EXPECTED_TRIALS
        or len(set(trial_ids)) != EXPECTED_TRIALS
        or any(not isinstance(value, str) or not SAFE_ID.fullmatch(value) for value in trial_ids)
    ):
        raise ValueError("patch must contain exactly 135 unique safe trial IDs")
    if order != trial_ids:
        raise ValueError("patch validation_order must exactly match trial_ids")

    source = manifest["source_primary"]
    source_keys = {"manifest_sha256", "campaign_state_sha256", "primary_inventory_sha256", "counts"}
    if not isinstance(source, dict) or set(source) != source_keys:
        raise ValueError("patch source_primary schema is invalid")
    for key in source_keys - {"counts"}:
        if not isinstance(source[key], str) or not HEX64.fullmatch(source[key]):
            raise ValueError("patch source_primary hash is malformed: " + key)

    controls = manifest["control_results"]
    if not isinstance(controls, list) or [item.get("trial_id") if isinstance(item, dict) else None for item in controls] != EXPECTED_CONTROL_TRIAL_IDS:
        raise ValueError("patch does not contain the five locked controls in order")
    control_keys = {
        "trial_id", "status", "snapshots_compared", "validation_wall_s_ignored",
        "numeric_absolute_tolerance",
    }
    for item in controls:
        if (
            set(item) != control_keys
            or item["status"] != "PASS"
            or item["validation_wall_s_ignored"] is not True
            or not isinstance(item["snapshots_compared"], int)
            or isinstance(item["snapshots_compared"], bool)
            or item["snapshots_compared"] < 1
            or item["numeric_absolute_tolerance"] != 1e-8
        ):
            raise ValueError("patch contains a failed or malformed control result")

    counts = manifest["counts"]
    if not isinstance(counts, dict) or set(counts) != {"trial_ids", "valid", "invalid", "controls"}:
        raise ValueError("patch count schema is invalid")
    if (
        counts["trial_ids"] != EXPECTED_TRIALS
        or counts["controls"] != len(EXPECTED_CONTROL_TRIAL_IDS)
        or not isinstance(counts["valid"], int)
        or not isinstance(counts["invalid"], int)
        or isinstance(counts["valid"], bool)
        or isinstance(counts["invalid"], bool)
        or counts["valid"] < 0
        or counts["invalid"] < 0
        or counts["valid"] + counts["invalid"] != EXPECTED_TRIALS
    ):
        raise ValueError("patch counts are inconsistent")
    expected_status = "COMPLETE" if counts["invalid"] == 0 else "FAILED_CORRECTNESS"
    if manifest["status"] != expected_status:
        raise ValueError("patch status is inconsistent with invalid count")

    if not isinstance(summary, dict) or set(summary) != SUMMARY_KEYS:
        raise ValueError("validation_summary schema is invalid")
    invalid_ids = summary["invalid_trial_ids"]
    if (
        summary["schema_version"] != 1
        or summary["status"] != expected_status
        or summary["planned_pending"] != EXPECTED_TRIALS
        or summary["validated_pending"] != EXPECTED_TRIALS
        or summary["valid_pending"] != counts["valid"]
        or summary["geometric_or_snapshot_invalid"] != counts["invalid"]
        or not isinstance(invalid_ids, list)
        or len(invalid_ids) != counts["invalid"]
        or len(set(invalid_ids)) != len(invalid_ids)
        or not set(invalid_ids).issubset(set(trial_ids))
        or summary["controls_passed"] != len(EXPECTED_CONTROL_TRIAL_IDS)
        or summary["jobs"] != 1
        or summary["planner_trials_launched"] != 0
        or summary["seed_changes"] != 0
        or summary["classification"] != CLASSIFICATION
    ):
        raise ValueError("validation_summary is inconsistent with patch manifest")

    if (
        not isinstance(environment, dict)
        or environment.get("schema_version") != 1
        or environment.get("execution_role") != "posthoc_independent_validation_only"
        or environment.get("system") != "Linux"
        or environment.get("jobs") != 1
        or environment.get("planner_trials_launched") != 0
        or environment.get("seed_changes") != 0
    ):
        raise ValueError("cloud_environment is not a Linux validation-only environment")


def inspect_patch(patch: Path, sidecar: Path | None = None):
    patch = patch.resolve()
    if not patch.is_file():
        raise ValueError("patch archive is absent: " + str(patch))
    archive_sha256 = _sidecar_hash(patch, sidecar)
    with zipfile.ZipFile(patch, "r") as archive:
        members = _zip_members(archive)
        if not REQUIRED_ROOT_MEMBERS.issubset(members):
            raise ValueError("patch is missing required root members")
        manifest = _strict_json_bytes(archive.read("patch_manifest.json"), "patch_manifest.json")
        cloud_environment = _strict_json_bytes(archive.read("cloud_environment.json"), "cloud_environment.json")
        validation_summary = _strict_json_bytes(archive.read("validation_summary.json"), "validation_summary.json")
        _validate_patch_metadata(manifest, validation_summary, cloud_environment)
        trial_ids = manifest["trial_ids"]
        trial_set = set(trial_ids)
        allowed_roots = REQUIRED_ROOT_MEMBERS - {"patch_manifest.json"}
        for name in members:
            if name == "patch_manifest.json" or name in allowed_roots:
                continue
            if not _allowed_trial_member(name, trial_set):
                raise ValueError("patch contains a non-allowlisted member: " + name)
        _validate_entry_table(manifest, members)
        entry_by_name = {entry["path"]: entry for entry in manifest["entries"]}
        for name, entry in entry_by_name.items():
            payload = archive.read(name)
            if digest(payload) != entry["sha256"]:
                raise ValueError("patch member SHA-256 mismatch: " + name)
        primary_inventory = _strict_json_bytes(archive.read("primary_inventory.json"), "primary_inventory.json")
        return {
            "manifest": manifest,
            "primary_inventory": primary_inventory,
            "cloud_environment": cloud_environment,
            "validation_summary": validation_summary,
            "archive_sha256": archive_sha256,
            "members": members,
        }


def _tree_inventory(root: Path):
    root = root.resolve()
    entries = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("primary inventory forbids symlinks: " + str(path))
        if path.is_file():
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_hash(path),
            })
    return entries


def verify_primary_inventory(primary: Path, inventory: dict):
    if not isinstance(inventory, dict) or set(inventory) != {"schema_version", "policy", "entries", "inventory_sha256"}:
        raise ValueError("primary inventory schema is invalid")
    if inventory["schema_version"] != 1 or inventory["policy"] != "all_regular_files_exact":
        raise ValueError("unsupported primary inventory policy")
    entries = inventory["entries"]
    if not isinstance(entries, list) or entries != sorted(entries, key=lambda item: item.get("path", "")):
        raise ValueError("primary inventory entries are malformed or unsorted")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("primary inventory entry schema is invalid")
        name = _safe_member_name(entry["path"])
        if name in seen:
            raise ValueError("duplicate primary inventory path: " + name)
        seen.add(name)
        if (
            not isinstance(entry["size_bytes"], int)
            or isinstance(entry["size_bytes"], bool)
            or entry["size_bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or not HEX64.fullmatch(entry["sha256"])
        ):
            raise ValueError("malformed primary inventory entry: " + name)
    if digest(entries) != inventory.get("inventory_sha256"):
        raise ValueError("primary inventory digest mismatch")
    current = _tree_inventory(primary)
    if current != entries:
        expected = {item["path"]: item for item in entries}
        actual = {item["path"]: item for item in current}
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])[:5]
        raise ValueError(
            "primary differs from pause-point inventory; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return inventory["inventory_sha256"]


def _pending_ids(primary: Path, manifest: dict):
    pending = []
    for entry in manifest["trials"]:
        directory = primary / entry["trial_id"]
        path = directory / "trial.json"
        if not path.is_file():
            continue
        trial = read_json(path)
        executed = trial.get("execution_status") not in ("BLOCKED", "NOT_RUN", "PLANNED")
        if executed and not validation_is_terminal(directory, trial):
            pending.append(entry["trial_id"])
    return pending


def _require_pending_order(patch_trial_ids: list[str], primary_pending_ids: list[str]):
    if patch_trial_ids != primary_pending_ids:
        raise ValueError("patch trial IDs/order do not exactly match current pending trials")


def verify_primary_identity(primary: Path, patch_info: dict):
    manifest = read_json(primary / "manifest.json")
    state = read_json(primary / "campaign_state.json")
    if manifest.get("experiment_id") != "a3_vs_all_v1_20260930" or manifest.get("profile") != "combined":
        raise ValueError("primary is not the locked A3-vs-all campaign")
    if state.get("status") != "PARTIAL" or state.get("phase") != "INTERRUPTED":
        raise ValueError("primary is no longer the paused PARTIAL / INTERRUPTED campaign")
    counts = campaign_counts(primary, manifest)
    expected_counts = {
        "planned": 250,
        "executed": 250,
        "not_run": 0,
        "offline_validated": 115,
        "validation_pending": 135,
        "validated_success": 115,
        "geometric_invalid": 0,
    }
    if counts != expected_counts:
        raise ValueError("primary counts changed since the approved Cloud handoff")
    source = patch_info["manifest"].get("source_primary")
    if not isinstance(source, dict):
        raise ValueError("patch source_primary identity is absent")
    if source.get("manifest_sha256") != file_hash(primary / "manifest.json"):
        raise ValueError("patch was built from a different primary manifest")
    if source.get("campaign_state_sha256") != file_hash(primary / "campaign_state.json"):
        raise ValueError("patch was built from a different primary campaign state")
    if source.get("counts") != counts:
        raise ValueError("patch primary counts do not match current primary")
    source_archive = primary / "source_at_run.zip"
    if not source_archive.is_file() or file_hash(source_archive) != EXPECTED_SOURCE_AT_RUN_SHA256:
        raise ValueError("primary frozen source archive changed")
    if patch_info["manifest"]["source_at_run_sha256"] != file_hash(source_archive):
        raise ValueError("patch frozen source identity differs from primary")
    inventory_sha = verify_primary_inventory(primary, patch_info["primary_inventory"])
    if source.get("primary_inventory_sha256") != inventory_sha:
        raise ValueError("patch primary inventory identity mismatch")
    trial_ids = patch_info["manifest"]["trial_ids"]
    _require_pending_order(trial_ids, _pending_ids(primary, manifest))
    return manifest, state, counts, inventory_sha


def _immutable_trial_projection(trial: dict):
    return {key: value for key, value in trial.items() if key not in TRIAL_VALIDATION_FIELDS}


def _validate_patched_trial(primary_trial: dict, patched_trial: dict, trial_id: str):
    if _immutable_trial_projection(primary_trial) != _immutable_trial_projection(patched_trial):
        raise ValueError("patch changes non-validation trial fields: " + trial_id)
    if patched_trial.get("trial_id") != trial_id:
        raise ValueError("patched trial identity mismatch: " + trial_id)
    wall = patched_trial.get("validation_wall_s")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(wall) or wall < 0:
        raise ValueError("patched trial lacks a finite validation_wall_s: " + trial_id)
    if patched_trial.get("independent_valid") not in (True, False):
        raise ValueError("patched trial lacks terminal independent_valid: " + trial_id)
    if not isinstance(patched_trial.get("success"), bool):
        raise ValueError("patched trial lacks boolean success: " + trial_id)


def _strict_file_json(path: Path):
    if not path.is_file():
        raise ValueError("required patch JSON is absent: " + str(path))
    return _strict_json_bytes(path.read_bytes(), str(path))


def _validate_trial_output_bundle(primary_root: Path, source_root: Path, trial: dict, trial_id: str):
    validation = _strict_file_json(source_root / "validation.json")
    events = _strict_file_json(source_root / "validated_events.json")
    if not isinstance(validation, dict) or not isinstance(events, list):
        raise ValueError("patch validation bundle has invalid JSON roots: " + trial_id)
    if not isinstance(validation.get("valid"), bool) or not isinstance(validation.get("exact_goal_reached"), bool):
        raise ValueError("patch final validation lacks boolean validity/goal fields: " + trial_id)
    snapshots = validation.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or len(snapshots) != len(events):
        raise ValueError("patch snapshot/event count is invalid: " + trial_id)

    progress_path = primary_root / "progress.jsonl"
    if not progress_path.is_file():
        raise ValueError("primary progress log is absent: " + trial_id)
    original_events = []
    for line_number, line in enumerate(progress_path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        event = _strict_json_bytes(line, f"{progress_path}:{line_number}")
        if not isinstance(event, dict):
            raise ValueError("primary progress event is not an object: " + trial_id)
        if event.get("event_type") == "improvement":
            if "independent_valid" in event:
                raise ValueError("pending primary progress already contains validation output: " + trial_id)
            original_events.append(event)
    if len(original_events) != len(events):
        raise ValueError("patch validated events do not match primary improvement count: " + trial_id)

    snapshot_ids = []
    for index, (event, snapshot, original_event) in enumerate(zip(events, snapshots, original_events)):
        if not isinstance(event, dict) or not isinstance(snapshot, dict):
            raise ValueError("patch snapshot/event is not an object: " + trial_id)
        event_without_validation = {key: value for key, value in event.items() if key != "independent_valid"}
        if set(event) != set(original_event) | {"independent_valid"} or event_without_validation != original_event:
            raise ValueError("patch validated event changes immutable progress data: " + trial_id)
        snapshot_id = event.get("snapshot_id")
        if (
            not isinstance(snapshot_id, str)
            or not re.fullmatch(r"snapshot_[0-9]+", snapshot_id)
            or snapshot_id in snapshot_ids
            or snapshot.get("snapshot_id") != snapshot_id
        ):
            raise ValueError("patch snapshot identity/order is invalid: " + trial_id)
        snapshot_ids.append(snapshot_id)
        if not isinstance(snapshot.get("valid"), bool) or event.get("independent_valid") is not snapshot["valid"]:
            raise ValueError("patch event/snapshot validity mismatch: " + trial_id)
        snapshot_file = source_root / "snapshots" / snapshot_id / "validation.json"
        if _strict_file_json(snapshot_file) != snapshot:
            raise ValueError("patch embedded and per-snapshot validation differ: " + trial_id)
        xy = primary_root / "snapshots" / snapshot_id / "path_xy.csv"
        surface = primary_root / "snapshots" / snapshot_id / "path_surface.csv"
        if (
            not xy.is_file()
            or not surface.is_file()
            or snapshot.get("xy_sha256") != file_hash(xy)
            or snapshot.get("surface_sha256") != file_hash(surface)
        ):
            raise ValueError("patch snapshot validation is not bound to primary path bytes: " + trial_id)

    expected_files = {
        "trial.json",
        "validation.json",
        "validated_events.json",
        *{f"snapshots/{snapshot_id}/validation.json" for snapshot_id in snapshot_ids},
    }
    actual_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("patch trial output allowlist does not match validated events: " + trial_id)

    field_mapping = {
        "independent_valid": "valid",
        "independently_recomputed_cost_m": "cost_terrain_m",
        "length_xy_m": "length_xy_m",
        "cost_abs_error_m": "cost_abs_error_m",
        "minimum_footprint_clearance_m": "minimum_footprint_clearance_m",
        "maximum_slope_deg": "maximum_slope_deg",
    }
    for trial_key, validation_key in field_mapping.items():
        if trial.get(trial_key) != validation.get(validation_key):
            raise ValueError("patch trial/final validation field mismatch: " + trial_id + "/" + trial_key)
    snapshots_valid = all(snapshot["valid"] for snapshot in snapshots)
    if trial.get("snapshots_valid") is not snapshots_valid:
        raise ValueError("patch trial snapshots_valid mismatch: " + trial_id)
    first_elapsed = events[0].get("elapsed_s")
    budget = trial.get("time_budget_s")
    if (
        not isinstance(first_elapsed, (int, float))
        or isinstance(first_elapsed, bool)
        or not math.isfinite(first_elapsed)
        or not isinstance(budget, (int, float))
        or isinstance(budget, bool)
        or not math.isfinite(budget)
    ):
        raise ValueError("patch success inputs are malformed: " + trial_id)
    expected_success = bool(
        trial.get("execution_status") == "COMPLETED"
        and trial.get("planner_status") == "EXACT_SOLUTION"
        and validation["valid"]
        and validation["exact_goal_reached"]
        and first_elapsed <= budget
        and all(event.get("independent_valid") is True for event in events)
    )
    if trial.get("success") is not expected_success:
        raise ValueError("patch trial success classification is inconsistent: " + trial_id)
    return not bool(validation["valid"] and snapshots_valid and expected_success)


def extract_patch(patch: Path, destination: Path, inspected: dict):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(patch, "r") as archive:
        for name in sorted(inspected["members"]):
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    for entry in inspected["manifest"]["entries"]:
        path = destination.joinpath(*PurePosixPath(entry["path"]).parts)
        if path.stat().st_size != entry["size_bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError("extracted patch member verification failed: " + entry["path"])


def apply_trial_files(primary: Path, derivative: Path, extracted: Path, trial_ids: list[str]):
    invalid_ids = []
    for trial_id in trial_ids:
        source_root = extracted / "trials" / trial_id
        destination_root = derivative / trial_id
        if not source_root.is_dir() or not destination_root.is_dir():
            raise ValueError("patch or derivative trial directory is absent: " + trial_id)
        primary_trial = read_json(primary / trial_id / "trial.json")
        patched_trial = _strict_file_json(source_root / "trial.json")
        if not isinstance(patched_trial, dict):
            raise ValueError("patched trial JSON is not an object: " + trial_id)
        _validate_patched_trial(primary_trial, patched_trial, trial_id)
        if _validate_trial_output_bundle(primary / trial_id, source_root, patched_trial, trial_id):
            invalid_ids.append(trial_id)
        required = {"trial.json", "validation.json", "validated_events.json"}
        present = {
            path.relative_to(source_root).as_posix()
            for path in source_root.rglob("*")
            if path.is_file()
        }
        if not required.issubset(present):
            raise ValueError("patch trial lacks required validation files: " + trial_id)
        for relative in sorted(present):
            source = source_root.joinpath(*PurePosixPath(relative).parts)
            destination = destination_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative != "trial.json" and destination.exists():
                # The administrative pause interrupted one validation after it
                # atomically wrote some per-snapshot files but before the final
                # trial.json commit.  Such files are explicitly non-terminal
                # under validation_is_terminal().  A complete revalidation may
                # replace them in the derivative only; final validation/events
                # must still be absent from the primary pending trial.
                parts = PurePosixPath(relative).parts
                uncommitted_snapshot = (
                    len(parts) == 3
                    and parts[0] == "snapshots"
                    and parts[1].startswith("snapshot_")
                    and parts[2] == "validation.json"
                )
                if not uncommitted_snapshot:
                    raise ValueError(
                        "primary pending trial already has a committed patch output: "
                        + trial_id + "/" + relative
                    )
            shutil.copy2(source, destination)
    return invalid_ids


def _posthoc_extension(primary: Path, primary_state: dict, primary_counts: dict,
                       inventory_sha: str, patch_info: dict, frozen_toolchain: dict):
    patch_manifest = patch_info["manifest"]
    return {
        "classification": CLASSIFICATION,
        "started_utc": utc(),
        "source_primary": {
            "path": str(primary.resolve()),
            "manifest_sha256": file_hash(primary / "manifest.json"),
            "campaign_state_sha256": file_hash(primary / "campaign_state.json"),
            "primary_inventory_sha256": inventory_sha,
            "state": primary_state,
            "counts": primary_counts,
        },
        "cloud_patch": {
            "archive_sha256": patch_info["archive_sha256"],
            "capsule_sha256": patch_manifest.get("capsule_sha256"),
            "capsule_manifest_sha256": patch_manifest.get("capsule_manifest_sha256"),
            "source_at_run_sha256": patch_manifest.get("source_at_run_sha256"),
            "jobs": patch_manifest.get("jobs"),
            "trial_count": len(patch_manifest["trial_ids"]),
            "status": patch_manifest.get("status"),
            "environment": patch_info["cloud_environment"],
            "validation_summary": patch_info["validation_summary"],
        },
        "planner_trials_launched": 0,
        "seed_changes": 0,
        "local_integration_analysis_toolchain_sha256": frozen_toolchain,
        "validation_wall_s_interpretation": (
            "mixed Windows-local and Codex-Cloud diagnostic only; not a planner-performance metric"
        ),
        "note": "Supplementary posthoc validation; the registered primary remains unchanged and PARTIAL.",
    }


def _annotate_report(report: Path, derivative: Path, final_output: Path, analysis: dict):
    summary_path = report / "summary.json"
    summary = read_json(summary_path)
    summary["posthoc_validation"] = {
        "classification": CLASSIFICATION,
        "validation_environments": ["Windows-local", "Codex-Cloud-Linux"],
        "validation_wall_s_interpretation": (
            "diagnostic only; mixed environments; must not be used to rank planner performance"
        ),
        "registered_primary_status": "PARTIAL",
    }
    dump(summary_path, summary)
    report_path = report / "REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(str(derivative.resolve()), str(final_output.resolve()))
    caveat = (
        "\n\n## Provenance của validation hậu kiểm\n\n"
        "Báo cáo này mang nhãn `POSTHOC_EXTENDED_VALIDATION_NOT_REGISTERED_PRIMARY`. "
        "115 trial được validation trên Windows và 135 trial còn lại được validation tuần tự "
        "trên Codex Cloud Linux bằng cùng frozen independent validator. `validation_wall_s` "
        "chỉ là diagnostic của hai môi trường khác nhau, không được dùng để xếp hạng tốc độ planner. "
        "Primary đã đăng ký vẫn giữ `PARTIAL / INTERRUPTED`.\n"
    )
    report_path.write_text(text.rstrip() + caveat, encoding="utf-8")
    manifest_sha = file_hash(derivative / "manifest.json")
    inventory = verify_raw_artifacts(derivative)
    checked_summary, artifacts = _report_artifacts(report, manifest_sha, inventory["inventory_sha256"])
    analysis.update(
        summary_status=checked_summary["status"],
        report_artifact_inventory_sha256=artifacts["inventory_sha256"],
        report_artifacts=artifacts["entries"],
    )
    dump(derivative / "analysis_attempt.json", analysis)


def run(args):
    primary = args.primary.resolve()
    output = args.output.resolve()
    report_output = args.report_output.resolve()
    patch = args.patch.resolve()
    if not primary.is_dir():
        raise ValueError("primary campaign directory is absent")
    frozen_toolchain = _verify_frozen_local_toolchain(primary)
    if output.exists() or report_output.exists():
        raise ValueError("output and report-output must not exist")
    for first, second, label in (
        (primary, output, "primary/output"),
        (primary, report_output, "primary/report"),
        (output, report_output, "output/report"),
    ):
        if first == second or first in second.parents or second in first.parents:
            raise ValueError(label + " trees must be disjoint")
    if args.analysis_budget_s <= 5:
        raise ValueError("analysis budget must exceed five seconds")

    inspected = inspect_patch(patch, args.patch_sha256)
    primary_manifest, primary_state, primary_counts, inventory_sha = verify_primary_identity(primary, inspected)

    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    output_stage = Path(tempfile.mkdtemp(prefix=".posthoc-cloud-" + output.name + "-", dir=output.parent))
    report_stage = report_output.parent / (".posthoc-cloud-report-" + report_output.name + "-" + output_stage.name.rsplit("-", 1)[-1])
    patch_extract_parent = Path(tempfile.mkdtemp(prefix=".cloud-patch-extract-", dir=output.parent))
    patch_extract = patch_extract_parent / "payload"
    try:
        extract_patch(patch, patch_extract, inspected)
        shutil.copytree(primary, output_stage, dirs_exist_ok=True)
        copied = _tree_inventory(output_stage)
        if copied != inspected["primary_inventory"]["entries"]:
            raise ValueError("derivative staging copy differs from verified primary")
        patched_invalid_ids = apply_trial_files(
            primary, output_stage, patch_extract, inspected["manifest"]["trial_ids"]
        )
        expected_invalid_ids = inspected["validation_summary"]["invalid_trial_ids"]
        if patched_invalid_ids != expected_invalid_ids:
            raise ValueError("patch invalid trial list differs from validation outputs")

        manifest = read_json(output_stage / "manifest.json")
        extension = _posthoc_extension(
            primary, primary_state, primary_counts, inventory_sha, inspected, frozen_toolchain
        )
        manifest["posthoc_validation_extension"] = extension
        manifest.setdefault("history", []).append({
            "utc": extension["started_utc"],
            "event": "CLOUD_POSTHOC_VALIDATION_PATCH_APPLIED",
            "classification": CLASSIFICATION,
            "patch_sha256": inspected["archive_sha256"],
            "validated_in_extension": len(inspected["manifest"]["trial_ids"]),
            "planner_trials_launched": 0,
        })
        refresh_manifest(output_stage, manifest, validated=True)
        manifest = read_json(output_stage / "manifest.json")
        counts = campaign_counts(output_stage, manifest)
        if counts["planned"] != 250 or counts["executed"] != 250 or counts["validation_pending"] != 0:
            raise ValueError("patched derivative does not have 250 terminal validations")
        if counts["geometric_invalid"] != inspected["manifest"]["counts"]["invalid"]:
            raise ValueError("patched derivative invalid count differs from patch manifest")
        extension["finished_apply_utc"] = utc()
        extension["counts_after_validation"] = counts
        manifest["posthoc_validation_extension"] = extension
        dump(output_stage / "manifest.json", manifest)
        dump(output_stage / "cloud_posthoc_provenance.json", extension)

        config = read_json(output_stage / "resolved_config.json")
        protocol = dict(campaign_descriptor(config)["protocol"])
        protocol["hard_finalize_s"] = float(args.analysis_budget_s)
        clock = CampaignClock(0.0)
        clock.start()
        analysis = run_analysis_with_deadline(output_stage, report_stage, protocol, clock, allow_existing=False)
        if analysis.get("status") != "COMPLETE":
            raise RuntimeError("posthoc analysis did not complete: " + str(analysis.get("status")))
        _annotate_report(report_stage, output_stage, output, analysis)
        counts = campaign_counts(output_stage, manifest)
        summary_status = analysis.get("summary_status")
        status = "FAILED_CORRECTNESS" if counts["geometric_invalid"] or summary_status == "FAILED_CORRECTNESS" else "COMPLETE"
        if counts["validation_pending"]:
            status = "PARTIAL"
        state = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "classification": CLASSIFICATION,
            "status": status,
            "phase": "POSTHOC_FINALIZED",
            "utc": utc(),
            "elapsed_s": clock.elapsed(),
            "counts": counts,
            "source_primary": extension["source_primary"],
            "cloud_patch": extension["cloud_patch"],
            "analysis": analysis,
            "planner_trials_launched": 0,
            "seed_changes": 0,
            "note": extension["note"],
        }
        dump(output_stage / "campaign_state.json", state)
        verify_raw_artifacts(output_stage)

        output_stage.rename(output)
        try:
            report_stage.rename(report_output)
        except BaseException:
            output.rename(output_stage)
            raise
        print(json.dumps({
            "status": status,
            "counts": counts,
            "output": str(output),
            "report_output": str(report_output),
            "patch_sha256": inspected["archive_sha256"],
        }, ensure_ascii=False), flush=True)
        return 0 if status == "COMPLETE" else 2
    except BaseException:
        print("preserved derivative staging at: " + str(output_stage), file=sys.stderr, flush=True)
        if report_stage.exists():
            print("preserved report staging at: " + str(report_stage), file=sys.stderr, flush=True)
        raise
    finally:
        if patch_extract_parent.exists():
            shutil.rmtree(patch_extract_parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-sha256", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--analysis-budget-s", type=float, default=3600.0)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
