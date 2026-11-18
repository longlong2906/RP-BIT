"""Read-only independent recheck of historical G5-v1 retained artifacts.

The only output is a new forensic report directory. No statistical inference.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from experiment_common import ROOT, dump, file_hash, read_json
from g5_analysis import read_g5_trial
from independent_verifier import Oracle
from validate_results import read_path, validate_trial

V1 = ROOT / "results/full_g5_v1"
OLD_PROBE = ROOT / "logs/g5/approximate_analysis_probe.json"
RETAINED = ROOT / "tests/fixtures/g5_taxonomy/retained_approximate"


def protected_snapshot():
    baseline = json.loads((ROOT / "logs/g5_v2/preservation_baseline.json").read_text(encoding="utf-8-sig"))["files"]
    prefixes = ("results/full_g5_v1/", "reports/full_g5_v1/", "data/full_g5_v1/", "logs/g5/")
    files = {name: sha for name, sha in baseline.items() if name.startswith(prefixes)}
    files["results/full_g5_v1/manifest.json"] = read_json(ROOT / "logs/g5/final_handoff_audit.json")["manifest_sha256"]
    mismatches = [name for name, sha in files.items() if not (ROOT/name).is_file() or file_hash(ROOT/name) != sha]
    if mismatches:
        raise ValueError("historical bytes differ from initial inventory: " + str(mismatches[:10]))
    return files


def check_group(map_path, entries, identity):
    oracle = Oracle(map_path)
    checked_final = checked_snapshots = 0
    max_final_error = max_snapshot_error = 0.
    errors = []
    for entry in entries:
        directory = V1 / entry["trial_id"]
        try:
            row = read_g5_trial(V1, entry, 10., 1e-8, 1e-8, identity)
            if row["correctness_error"]:
                errors.append({"trial_id": entry["trial_id"], "analyzer": row["errors"]})
            trial = read_json(directory / "trial.json")
            request_path = directory / "request.json"
            if trial.get("path_available"):
                request = read_json(request_path)
                check = oracle.validate_path(read_path(directory / "path_xy.csv"), request["start"],
                    request["goal"], trial["best_cost_terrain_m"], read_path(directory / "path_surface.csv"))
                expected = read_json(directory / "validation.json")
                if not check["valid"] or not expected["valid"] or abs(check["cost_terrain_m"]-expected["cost_terrain_m"]) > 1e-8:
                    errors.append({"trial_id": entry["trial_id"], "final_oracle": check["errors"]})
                checked_final += 1
                max_final_error = max(max_final_error, check["cost_abs_error_m"])
            if request_path.exists():
                request = read_json(request_path)
                for event in read_json(directory / "validated_events.json") if (directory / "validated_events.json").exists() else []:
                    snapshot = directory / "snapshots" / event["snapshot_id"]
                    check = oracle.validate_path(read_path(snapshot / "path_xy.csv"), request["start"],
                        request["goal"], event["best_cost_terrain_m"], read_path(snapshot / "path_surface.csv"))
                    expected = read_json(snapshot / "validation.json")
                    if (not check["valid"] or not expected["valid"] or
                            abs(check["cost_terrain_m"]-expected["cost_terrain_m"]) > 1e-8 or
                            file_hash(snapshot / "path_xy.csv") != expected["xy_sha256"] or
                            file_hash(snapshot / "path_surface.csv") != expected["surface_sha256"]):
                        errors.append({"trial_id": entry["trial_id"], "snapshot_id": event["snapshot_id"], "oracle": check["errors"]})
                    checked_snapshots += 1
                    max_snapshot_error = max(max_snapshot_error, check["cost_abs_error_m"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            errors.append({"trial_id": entry["trial_id"], "malformed": str(exc)})
    return {"map_path": map_path, "final_paths": checked_final, "snapshots": checked_snapshots,
        "max_final_cost_abs_error_m": max_final_error,
        "max_snapshot_cost_abs_error_m": max_snapshot_error, "errors": errors}


def check_probe(output):
    probe = read_json(OLD_PROBE)
    provenance = read_json(ROOT / "tests/fixtures/g5_taxonomy/retained_provenance.json")
    for name, sha in provenance["files"].items():
        if file_hash(RETAINED/name) != sha:
            raise ValueError("retained CTest fixture changed: " + name)
    if file_hash(RETAINED / "worker.json") != probe["worker_sha256"]:
        raise ValueError("retained CTest worker bytes differ from historical failed probe")
    raw = read_json(RETAINED / "worker.json")
    if raw["planner_status"] != "APPROXIMATE_SOLUTION" or raw["execution_status"] != "TIMEOUT":
        raise ValueError("retained probe is not the historical approximate timeout")
    # Work only on a new forensic copy. The raw worker/path bytes are identical
    # to the failed CTest probe; the copied request points to its copied map.
    root = output / "approximate_probe"
    trial_id = "retained_approximate"
    directory = root / trial_id
    shutil.copytree(RETAINED, directory)
    request = read_json(directory / "request.json")
    request["map_path"] = str(directory / "map.t25d")
    dump(directory / "request.json", request)
    entry = {"trial_id": trial_id, "family": "RETAINED_REAL_CTEST_FIXTURE",
             "map_id": "CTEST_FIXTURE_NOT_G5", "map_path": request["map_path"],
             "map_hash": file_hash(directory / "map.t25d"),
             "query_id": "blocked_goal_fixture", "planner_id": "rrtstar", "planner_seed": 771}
    dump(directory / "trial.json", dict(raw, **entry))
    checked = validate_trial(directory, Oracle(directory / "map.t25d"))
    row = read_g5_trial(root, entry, request["planning_budget_s"], 1e-8, 1e-8, {})
    valid = (checked["independent_valid"] is True and row["geometry_valid"] is True and
             row["exact_goal_reached"] is False and row["exact_incumbent_present"] is False and
             row["validated_success"] is False and row["correctness_error"] is False)
    return {"status": "PASS" if valid else "FAIL", "planner_status": raw["planner_status"],
        "execution_status": raw["execution_status"], "geometry_valid": row["geometry_valid"],
        "exact_goal_reached": row["exact_goal_reached"],
        "exact_incumbent_present": row["exact_incumbent_present"],
        "validated_success": row["validated_success"], "correctness_error": row["correctness_error"],
        "reader_errors": row["errors"],
        "worker_json_sha256": file_hash(RETAINED / "worker.json"),
        "source": "byte-identical raw CTest approximate worker/path; production read_g5_trial executed on new forensic copy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("forensic output already exists; never overwrite")
    if output == V1.resolve() or V1.resolve() in output.parents:
        raise ValueError("forensic output must be outside historical results")
    output.mkdir(parents=True, exist_ok=False)
    before = protected_snapshot()
    manifest = read_json(V1 / "manifest.json")
    lock = read_json(V1 / "manifest.lock.json")
    if file_hash(V1 / "manifest.json") != lock["sha256"]:
        raise ValueError("historical manifest lock mismatch")
    source_manifest = read_json(V1 / "source_manifest.json")
    if file_hash(V1 / "source_at_run.zip") != source_manifest["archive_sha256"]:
        raise ValueError("historical source archive mismatch")
    with zipfile.ZipFile(V1 / "source_at_run.zip") as archive:
        archive_members_valid = all(
            hashlib.sha256(archive.read(name.replace("\\", "/"))).hexdigest() == sha
            for name, sha in source_manifest["files"].items())
    if not archive_members_valid:
        raise ValueError("historical source member checksum mismatch")
    if manifest["identity"]["worker_sha256"] != file_hash(ROOT / "build-g5/terrain_benchmark.exe"):
        raise ValueError("historical worker binary mismatch")
    if manifest["identity"]["ompl_library_sha256"] != file_hash(ROOT / ".deps/install/lib/libompl.dll"):
        raise ValueError("historical OMPL DLL mismatch")
    executed = [entry for entry in manifest["trials"] if (V1 / entry["trial_id"] / "trial.json").exists()]
    groups = {}
    for entry in executed:
        groups.setdefault(entry["map_path"], []).append(entry)
    parts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(check_group, map_path, entries, manifest["identity"]) for map_path, entries in groups.items()]
        for future in as_completed(futures):
            parts.append(future.result())
            print("read-only validated", parts[-1]["final_paths"], "final paths and",
                  parts[-1]["snapshots"], "snapshots", flush=True)
    after = protected_snapshot()
    if before != after:
        raise ValueError("historical inventory changed during forensic validation")
    probe = check_probe(output)
    final_count = sum(p["final_paths"] for p in parts)
    snapshot_count = sum(p["snapshots"] for p in parts)
    errors = [e for p in parts for e in p["errors"]]
    success = len(executed) == 129 and final_count == 128 and snapshot_count == 3187 and not errors and probe["status"] == "PASS"
    summary = {"status": "PASS" if success else "FAIL", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "READ_ONLY_FORENSIC_NO_PRIMARY_INFERENCE", "primary_dataset": False,
        "v1_status": read_json(V1 / "execution_state.json")["status"],
        "retained_attempts": len(executed), "final_paths_reverified": final_count,
        "snapshots_reverified": snapshot_count, "errors": errors,
        "max_final_cost_abs_error_m": max((p["max_final_cost_abs_error_m"] for p in parts), default=None),
        "max_snapshot_cost_abs_error_m": max((p["max_snapshot_cost_abs_error_m"] for p in parts), default=None),
        "historical_files_unchanged": len(after), "manifest_sha256": file_hash(V1 / "manifest.json"),
        "historical_source_sha256": manifest["identity"]["code_sha256"],
        "analyzer_sha256": file_hash(ROOT / "tools/g5_analysis.py"),
        "forensic_helper_sha256": file_hash(__file__),
        "worker_sha256": manifest["identity"]["worker_sha256"],
        "ompl_dll_sha256": manifest["identity"]["ompl_library_sha256"],
        "historical_source_archive_valid": archive_members_valid,
        "approximate_probe": probe, "bootstrap": "NOT_RUN_FORENSIC", "correlation": "NOT_RUN_FORENSIC"}
    dump(output / "summary.json", summary)
    (output / "REPORT.md").write_text(
        "# G5-v1 forensic analyzer v2\n\n" +
        "Chỉ đọc artifact lịch sử; G5-v1 vẫn BLOCKED. Không chạy primary inference.\n\n" +
        f"Status: **{summary['status']}**. Retained attempts: {len(executed)}; final paths: {final_count}; snapshots: {snapshot_count}.\n\n" +
        f"Approximate CTest probe: {probe['status']}; analyzer correctness errors: {len(errors)}.\n\n" +
        "Không dùng kết quả forensic để chọn map/query/seed hoặc tune v2.\n", encoding="utf-8")
    print({"status": summary["status"], "retained_attempts": len(executed),
           "final_paths": final_count, "snapshots": snapshot_count, "errors": len(errors)}, flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
