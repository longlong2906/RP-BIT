"""Schema-12 blinded SEM scoring, independent geometry and paired reporting.

Final selection is made before validation. A failed final incumbent is never
replaced with an earlier safe path. Clearance is neither measured nor exported.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np

from experiment_common import digest, dump
from native_paper_analysis import (read, finite_tree, csv_path, trial_inventory, raw_hashes,
    sharp_turn_count, close, verify_native_artifacts, putn_path_cost, mean_sd, UNSTARTED)
from native_paper_oracle import NativePaperOracle as Oracle
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY
from native_paper_v3_sem_evaluator import ensure_shared_field
from native_paper_v3_memory import admission_snapshot
from terrain_compare_protocol import ROOT, sha256

PROTOCOL_ID = "a3t_sem_putn_kaggle_v3"
BACKENDS = ("a3", "sem", "putn")
LABELS = {"a3":"A3-T", "sem":"SEM-RRT*-paper-reconstruction", "putn":"PUTN/PF-RRT*-paper-static"}
GEOMETRY_METRICS = ("cost_terrain_m", "length_xy_m", "sharp_turn_count", "maximum_swept_footprint_slope_deg")
QUALITY_METRICS = GEOMETRY_METRICS+("pri",)
TIME_METRICS = ("solve_time_s", "first_geometry_s", "first_paper_s", "first_geometry_latency_s", "first_paper_latency_s")


def validation_code_identity():
    names = ("native_paper_v3_analysis.py", "native_paper_v3_sem_evaluator.py", "native_paper_v3_memory.py",
             "native_paper_analysis.py", "independent_verifier.py", "native_paper_oracle.py")
    return digest({name:sha256(ROOT/"tools"/name) for name in names})


def evaluator_inventory(root, rows):
    """Derived artifacts trusted by verdicts, verified against their saved hash."""
    result = {}
    for row in rows:
        expected = row.get("shared_evaluator_sha256")
        if expected is None:
            continue
        map_id = row["map_id"]
        directory = Path(root)/"shared_sem_evaluator"/map_id
        metadata = read(directory/"field.json")
        if (metadata.get("identity") != row["shared_evaluator_identity"] or
                metadata.get("field_sha256") != expected or sha256(directory/"field.npz") != expected):
            raise ValueError("stale shared SEM evaluator artifact")
        record = dict(field_sha256=expected, metadata_sha256=sha256(directory/"field.json"),
                      identity=row["shared_evaluator_identity"])
        if map_id in result and result[map_id] != record:
            raise ValueError("inconsistent shared evaluator identity across verdicts")
        result[map_id] = record
    return result


def final_event(events, budget):
    """Pick the native output by timestamp alone, before checking validity."""
    available = [event for event in events if event["elapsed_s"] <= budget]
    return available[-1] if available else None


def final_metrics(row):
    event = final_event(row["events"], row["budget_s"])
    return event["metrics"] if event else {}


def first_paper_metrics(row):
    event = next((event for event in row["events"] if event["elapsed_s"] <= row["budget_s"]
                  and event["independent_valid"] and event["paper_valid"]), None)
    return event["metrics"] if event else {}


def audit_sem_native_cost(path, raster_path):
    """Separate arithmetic oracle for the published native raster/objective.

    Uses major-axis integer rasterization and atan2 turns rather than the
    planner's error-step traversal and acos costs. It does not build a field.
    """
    with np.load(raster_path, allow_pickle=False) as archive:
        risk, valid = archive["risk"], archive["valid"]
        bounds, res = tuple(archive["bounds"]), float(archive["resolution"])
        ny, nx = risk.shape
        if res != .1 or valid.shape != risk.shape or valid.dtype != np.bool_:
            raise ValueError("invalid native SEM raster")
        def cell(point):
            if not bounds[0] <= point[0] <= bounds[2] or not bounds[1] <= point[1] <= bounds[3]:
                raise ValueError("native SEM path outside raster")
            return min(nx-1, int((point[0]-bounds[0])/res)), min(ny-1, int((point[1]-bounds[1])/res))
        length, total_risk, turns = [], [], []
        for a, b in (list(zip(path, path[1:])) or [(path[0], path[0])]):
            start, end = sorted((cell(a), cell(b)))
            x, y = start
            dx, dy = end[0]-x, end[1]-y
            if abs(dx) >= abs(dy):
                cells = [(x+k, y+(1 if dy >= 0 else -1)*((2*k*abs(dy)+abs(dx))//(2*abs(dx))))
                         for k in range(abs(dx)+1)] if dx else [(x, y)]
            else:
                cells = [(x+(1 if dx >= 0 else -1)*((2*k*abs(dx)+abs(dy))//(2*abs(dy))),
                          y+k*(1 if dy >= 0 else -1)) for k in range(abs(dy)+1)]
            samples = []
            for i, j in cells:
                value = float(risk[j, i])
                if not valid[j, i] or not math.isfinite(value) or value > 40:
                    raise ValueError("native SEM forbidden risk cell")
                samples.append(value)
            delta = ((b[0]-a[0])/res, (b[1]-a[1])/res)
            distance = math.hypot(*delta)
            length.append(distance)
            total_risk.append(distance/(max(map(abs, delta))+1)*math.fsum(samples))
        for a, b, c in zip(path, path[1:], path[2:]):
            ux, uy = b[0]-a[0], b[1]-a[1]
            vx, vy = c[0]-b[0], c[1]-b[1]
            angle = math.atan2(abs(ux*vy-uy*vx), ux*vx+uy*vy) if math.hypot(ux, uy)*math.hypot(vx, vy) else 0.
            turns.append(max(0., angle-math.pi/6))
        return .48*math.fsum(length)+.16*math.fsum(turns)+.36*math.fsum(total_risk)


def _read_source_events(directory, worker):
    if worker.get("snapshots") is not None:
        return worker["snapshots"]
    path = directory/"progress.jsonl"
    if not path.exists():
        return []
    return [value for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for value in [json.loads(line)] if value.get("event_type") == "improvement"]


def _validate_group(task):
    root, entries, code_identity = task
    root, oracle, field = Path(root), None, None
    rows = []
    expected_map_hash = None
    for entry in entries:
        directory = root/entry["trial_id"]
        request = entry["request"]
        if entry["backend"] not in BACKENDS:
            raise ValueError("unregistered schema-12 backend")
        map_file = Path(request["map_path"])
        if sha256(map_file) != entry["map_sha256"]:
            raise ValueError("raw map hash changed")
        if expected_map_hash is not None and expected_map_hash != entry["map_sha256"]:
            raise ValueError("validation group mixes different maps")
        expected_map_hash = entry["map_sha256"]
        if request.get("cloud_path") and sha256(request["cloud_path"]) != request["cloud_sha256"]:
            raise ValueError("raw cloud hash changed")
        # Derived evaluator bytes are part of verdict identity too. Check the
        # field even on a verdict-cache hit; a modified raster must not bypass
        # the evaluator's inventory check merely because raw paths stayed put.
        if (directory/"worker.json").exists():
            if oracle is None:
                oracle = Oracle(map_file)
            if field is None:
                field = ensure_shared_field(root/"shared_sem_evaluator"/entry["map_id"], request["cloud_path"],
                                            request["cloud_sha256"], oracle.bounds, expected_map_hash)
        inventory = trial_inventory(directory)
        cache_key = digest(dict(entry=entry, inventory=inventory, map_hash=expected_map_hash, code=code_identity,
                               evaluator_sha256=field.metadata["field_sha256"] if field is not None else None))
        receipt_path = directory/"validation_receipt.json"
        if receipt_path.exists() and (directory/"validation.json").exists():
            receipt = read(receipt_path)
            if receipt.get("cache_key") == cache_key and receipt.get("verdict_sha256") == sha256(directory/"validation.json"):
                cached = read(directory/"validation.json")
                cached["validation_cache_hit"] = True
                rows.append(cached)
                continue
        began = time.perf_counter()
        row = {key:entry.get(key) for key in ("trial_id", "backend", "track", "family", "map_id", "query_id", "planner_seed", "execution_status", "timing_stratum")}
        row.update(success=False, success_geometry=False, success_paper=False, attained_geometry=False,
                   attained_paper=False, events=[], errors=[], budget_s=request["planning_budget_s"],
                   native_status=None, verification_status="NO_WORKER", validation_cache_hit=False,
                   first_geometry_s=None, first_paper_s=None, first_geometry_latency_s=None, first_paper_latency_s=None,
                   selected_snapshot_id=None, first_paper_snapshot_id=None, solve_time_s=None,
                   selection_policy="last_native_incumbent_in_budget_before_validation")
        if (directory/"worker.json").exists():
            if digest(read(directory/"request.json")) != digest(request):
                raise ValueError("raw request differs from frozen plan")
            worker = read(directory/"worker.json")
            row["shared_evaluator_identity"] = field.metadata["identity"]
            row["shared_evaluator_sha256"] = field.metadata["field_sha256"]
            source_events = _read_source_events(directory, worker)
            seen, previous_time, previous_cost, paths = set(), -1., math.inf, {}
            native_raster_verified = False
            for source in source_events:
                sid = source["snapshot_id"]
                if not isinstance(sid, str) or Path(sid).name != sid or sid in seen or sid in (".", ".."):
                    raise ValueError("unsafe/repeated snapshot id")
                seen.add(sid)
                path = source.get("path_xy")
                if path is None:
                    path = csv_path(directory/"snapshots"/sid/"path_xy.csv")
                path = np.asarray(path, dtype=float).tolist()
                if not path or any(len(point) != 2 or not all(map(math.isfinite, point)) for point in path):
                    raise ValueError("malformed snapshot path")
                elapsed = source.get("elapsed_s", source.get("timestamp_s"))
                native = source.get("native_objective_value", source.get("planner_objective_value", source.get("objective_value")))
                if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous_time:
                    raise ValueError("nonmonotone/nonfinite snapshot time")
                if type(native) not in (int, float) or not math.isfinite(native) or native < 0 or native > previous_cost+1e-8:
                    raise ValueError("nonmonotone/nonfinite native cost")
                checked = oracle.validate_path(path, request["start"], request["goal"],
                    traversability=DEFAULT_TRAVERSABILITY if entry["backend"] == "a3" else None,
                    geometric_safety_metrics=True, clearance_metrics=False)
                for key in ("cost_terrain_m", "length_xy_m"):
                    if source.get(key) is not None and not close(source[key], checked.get(key)):
                        checked["errors"].append(key+"_mismatch")
                try:
                    if entry["backend"] == "a3":
                        recomputed = checked.get("objective_value")
                    elif entry["backend"] == "putn":
                        features = source.get("putn_features")
                        if not features or len(features) != len(path):
                            raise ValueError("PUTN path/features mismatch")
                        recomputed = putn_path_cost(features)
                    else:
                        if not native_raster_verified:
                            verify_native_artifacts(directory/"native_model", worker.get("native_model", {}).get("raster_artifacts"))
                            native_raster_verified = True
                        recomputed = audit_sem_native_cost(path, directory/"native_model/raster.npz")
                    if not close(native, recomputed):
                        checked["errors"].append("native_objective_mismatch")
                except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
                    recomputed = None
                    checked["errors"].append("native_cost_audit: "+str(error))
                geometry_valid = bool(checked["valid"] and not checked["errors"])
                risk = field.score_path(path)
                metrics = {key:checked.get(key) for key in GEOMETRY_METRICS}
                metrics.update(sharp_turn_count=sharp_turn_count(path), pri=risk["pri"])
                event = dict(snapshot_id=sid, elapsed_s=elapsed, native_objective_value=native,
                             native_recomputed=recomputed, independent_valid=geometry_valid,
                             paper_valid=bool(geometry_valid and risk["valid"]), metrics=metrics, sem_score=risk)
                row["events"].append(event)
                row["errors"].extend(checked["errors"])
                paths[sid] = path
                previous_time, previous_cost = elapsed, native
            if (directory/"path_xy.csv").exists():
                raw_final = csv_path(directory/"path_xy.csv")
                if not source_events or raw_final != paths[source_events[-1]["snapshot_id"]]:
                    row["errors"].append("final_path_not_native_last_incumbent")
            elif source_events:
                row["errors"].append("missing_final_path")
            status = worker.get("planner_status", worker.get("status"))
            exact = status == "EXACT_SOLUTION" or worker.get("exact_goal") is True
            if exact and not source_events:
                row["errors"].append("exact_status_without_published_incumbent")
            completed = entry["execution_status"] == "COMPLETED" and exact and not row["errors"]
            chosen = final_event(row["events"], row["budget_s"])
            in_budget = [event for event in row["events"] if event["elapsed_s"] <= row["budget_s"]]
            first_geometry = next((event for event in in_budget if event["independent_valid"]), None)
            first_paper = next((event for event in in_budget if event["paper_valid"]), None)
            timings = worker.get("phase_timings", {key:worker.get(key) for key in ("load_s", "preprocessing_s", "setup_s", "actual_solve_wall_s")})
            row.update(native_status=status, verification_status="FAILED" if row["errors"] else "PASS", timings=timings,
                       solve_time_s=timings.get("solve_s", timings.get("actual_solve_wall_s")),
                       selected_snapshot_id=chosen["snapshot_id"] if chosen else None,
                       first_paper_snapshot_id=first_paper["snapshot_id"] if first_paper else None,
                       first_geometry_s=first_geometry["elapsed_s"] if first_geometry else None,
                       first_paper_s=first_paper["elapsed_s"] if first_paper else None,
                       success_geometry=bool(completed and chosen and chosen["independent_valid"]),
                       success_paper=bool(completed and chosen and chosen["paper_valid"]),
                       attained_geometry=bool(completed and first_geometry), attained_paper=bool(completed and first_paper))
            row["success"] = row["success_paper"]
            setup = sum(float(timings.get(key) or 0) for key in ("load_s", "conversion_s", "preprocessing_s", "setup_s"))
            row["latency_scope"] = "worker_setup_plus_first_valid_solve; offline validation excluded"
            if worker.get("solve_started_utc") and (directory/"launch.json").exists():
                launch = read(directory/"launch.json").get("process_created")
                if launch is not None:
                    epoch = dt.datetime.fromisoformat(worker["solve_started_utc"].replace("Z", "+00:00")).timestamp()
                    setup = max(0., epoch-launch)
                    row["latency_scope"] = "launcher_creation_to_published_valid_snapshot; UTC resolution applies"
            for kind in ("geometry", "paper"):
                value = row["first_"+kind+"_s"]
                row["first_"+kind+"_latency_s"] = setup+value if value is not None else None
        row["errors"] = sorted(set(row["errors"]))
        row["validation_wall_s"] = time.perf_counter()-began
        directory.mkdir(parents=True, exist_ok=True)
        dump(directory/"validation.json", finite_tree(row))
        if inventory != trial_inventory(directory):
            raise ValueError("raw trial changed during validation")
        dump(receipt_path, dict(cache_key=cache_key, verdict_sha256=sha256(directory/"validation.json")))
        rows.append(finite_tree(row))
    return rows


def validate_campaign(root, jobs=1):
    from terrain_compare_scheduler import _lock
    root = Path(root)
    lock = _lock(root)
    try:
        return _validate_campaign_locked(root, jobs)
    finally:
        lock.unlink(missing_ok=True)


def _validate_campaign_locked(root, jobs):
    from terrain_compare_scheduler import detect_runtime, psutil
    manifest = read(root/"manifest.json")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("not a schema-12 campaign")
    if any(entry["execution_status"] in ("RUNNING", "STARTED", "START_REQUESTED") for entry in manifest["entries"]):
        raise ValueError("validation cannot overlap collection")
    groups = {}
    for entry in manifest["entries"]:
        groups.setdefault(entry["map_id"], []).append(entry)
    before, code = raw_hashes(root), validation_code_identity()
    runtime = detect_runtime()
    def host():
        memory = psutil.virtual_memory()
        return dict(total=int(memory.total), available=int(memory.available))
    admission = admission_snapshot(runtime, host)
    slots = admission["reservation_slots"]
    dump(root/"validation_memory_admission.json", admission)
    if slots < 1:
        raise RuntimeError("BLOCKED_VALIDATION_RAM: 2 GiB reservation unavailable")
    jobs = max(1, min(4, int(jobs), runtime["cpu_capacity"], slots))
    tasks = [(str(root), entries, code) for entries in groups.values()]
    if jobs == 1:
        rows = [row for task in tasks for row in _validate_group(task)]
    else:
        with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
            rows = [row for group in pool.map(_validate_group, tasks) for row in group]
    if raw_hashes(root) != before:
        raise ValueError("raw artifacts changed during validation")
    result = dict(protocol_id=PROTOCOL_ID, manifest_sha256=sha256(root/"manifest.json"), raw_hashes=before, rows=rows,
                  slots=len(rows), validated_success=sum(row["success_paper"] for row in rows),
                  validated_geometry_success=sum(row["success_geometry"] for row in rows),
                  correctness_failures=sum(bool(row["errors"]) for row in rows), validation_jobs=jobs,
                  validation_cache_hits=sum(row["validation_cache_hit"] for row in rows), validation_code_identity=code,
                  memory_admission=admission, shared_evaluator_inventory=evaluator_inventory(root, rows))
    dump(root/"validation_summary.json", finite_tree(result))
    return result


def group_summary(rows, selection="final_paper"):
    attempted = [row for row in rows if row["execution_status"] not in UNSTARTED]
    success_key = {"final_paper":"success_paper", "final_geometry":"success_geometry", "first_paper":"attained_paper"}[selection]
    valid = [row for row in attempted if row[success_key]]
    getter = first_paper_metrics if selection == "first_paper" else final_metrics
    metrics = [getter(row) for row in valid]
    length = math.fsum(value["cost_terrain_m"] for value in metrics)
    result = dict(selection=selection, planned=len(rows), attempted=len(attempted), success=len(valid),
        psr=len(valid)/len(attempted) if attempted else None,
        psr_geometry=sum(row["success_geometry"] for row in attempted)/len(attempted) if attempted else None,
        psr_paper=sum(row["success_paper"] for row in attempted)/len(attempted) if attempted else None,
        attained_geometry=sum(row["attained_geometry"] for row in attempted), attained_paper=sum(row["attained_paper"] for row in attempted),
        attained_geometry_rate=sum(row["attained_geometry"] for row in attempted)/len(attempted) if attempted else None,
        attained_paper_rate=sum(row["attained_paper"] for row in attempted)/len(attempted) if attempted else None,
        status_counts=dict(collections.Counter(row["execution_status"] for row in rows)),
        sti=sum(value["sharp_turn_count"] for value in metrics)/length if length > 0 else None)
    for key in QUALITY_METRICS:
        result[key] = mean_sd(value.get(key) for value in metrics)
    for key in TIME_METRICS:
        result[key] = mean_sd(row.get(key) for row in valid)
    return result


def paired_bootstrap(rows, competitor, metric, replicates=2000, selection="final_paper"):
    success_key = {"final_paper":"success_paper", "final_geometry":"success_geometry", "first_paper":"attained_paper"}[selection]
    getter = first_paper_metrics if selection == "first_paper" else final_metrics
    keyed = {}
    for row in rows:
        keyed.setdefault((row["family"], row["map_id"], row["planner_seed"]), {})[row["backend"]] = row
    pairs, planned, attempted, mismatched = [], 0, 0, 0
    for (family, map_id, _), methods in keyed.items():
        if "a3" not in methods or competitor not in methods:
            continue
        planned += 1
        a, b = methods["a3"], methods[competitor]
        if any(row["execution_status"] in UNSTARTED for row in (a, b)):
            continue
        attempted += 1
        if metric in TIME_METRICS and a.get("timing_stratum") != b.get("timing_stratum"):
            mismatched += 1
            continue
        if metric in ("psr_paper", "psr_geometry"):
            key = "success_paper" if metric == "psr_paper" else "success_geometry"
            value = (float(a[key]), float(b[key]))
        else:
            if not a[success_key] or not b[success_key]:
                continue
            aa, bb = getter(a), getter(b)
            if metric == "sti":
                if aa["cost_terrain_m"] <= 0 or bb["cost_terrain_m"] <= 0:
                    continue
                value = ((aa["sharp_turn_count"], aa["cost_terrain_m"]), (bb["sharp_turn_count"], bb["cost_terrain_m"]))
            else:
                value = (a.get(metric), b.get(metric)) if metric in TIME_METRICS else (aa.get(metric), bb.get(metric))
                if any(number is None for number in value):
                    continue
        pairs.append((family, map_id, value))
    coverage = dict(planned_pairs=planned, attempted_pairs=attempted, usable_pairs=len(pairs),
                    coverage=len(pairs)/planned if planned else None, excluded_stratum_mismatch=mismatched)
    if not pairs:
        return dict(**coverage, maps=0, mean_difference=None, ci95=None)
    groups = {}
    for family, map_id, value in pairs:
        groups.setdefault(family, {}).setdefault(map_id, []).append(value)
    def estimate(values):
        if metric == "sti":
            return sum(a[0] for a, b in values)/sum(a[1] for a, b in values)-sum(b[0] for a, b in values)/sum(b[1] for a, b in values)
        return statistics.mean(a-b for a, b in values)
    rng, draws = random.Random(73138), []
    for _ in range(replicates):
        values = []
        for maps in groups.values():
            ids = list(maps)
            for _ in ids:
                block = maps[rng.choice(ids)]
                values.extend(rng.choice(block) for _ in block)
        draws.append(estimate(values))
    draws.sort()
    return dict(**coverage, maps=len({item[1] for item in pairs}), mean_difference=estimate([item[2] for item in pairs]),
                ci95=[draws[int(.025*(replicates-1))], draws[int(.975*(replicates-1))]])


def attainment_rows(rows, checkpoints):
    result = []
    for backend in BACKENDS:
        group = [row for row in rows if row["backend"] == backend and row["execution_status"] not in UNSTARTED]
        for kind in ("geometry", "paper"):
            for t in checkpoints:
                count = sum(row["attained_"+kind] and row["first_"+kind+"_s"] <= t for row in group)
                result.append(dict(backend=backend, criterion=kind, elapsed_s=t, attained=count, attempted=len(group),
                                   fraction=count/len(group) if group else None))
    return result


def _write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        for row in records:
            writer.writerow({key:json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def analyze_campaign(root, output):
    from terrain_compare_scheduler import _lock
    root, output = Path(root), Path(output)
    lock = _lock(root)
    try:
        return _analyze_campaign_locked(root, output)
    finally:
        lock.unlink(missing_ok=True)


def _analyze_campaign_locked(root, output):
    output.mkdir(parents=True, exist_ok=True)
    validation = read(root/"validation_summary.json")
    if validation["manifest_sha256"] != sha256(root/"manifest.json") or validation["raw_hashes"] != raw_hashes(root):
        raise ValueError("stale validation")
    if validation["validation_code_identity"] != validation_code_identity():
        raise ValueError("validation code changed")
    current_evaluators = evaluator_inventory(root, validation["rows"])
    if validation.get("shared_evaluator_inventory") != current_evaluators:
        raise ValueError("stale shared SEM evaluator inventory")
    rows, tables = validation["rows"], []
    for scope, key in (("all", None), ("family", "family"), ("map", "map_id"), ("timing_stratum", "timing_stratum")):
        for value in (["all"] if key is None else sorted({str(row.get(key)) for row in rows})):
            for backend in BACKENDS:
                group = [row for row in rows if row["backend"] == backend and (key is None or str(row.get(key)) == value)]
                if group:
                    for selection in ("final_paper", "final_geometry", "first_paper"):
                        tables.append(dict(scope=scope, group=value, backend=backend, **group_summary(group, selection)))
    contrast_metrics = ("psr_paper", "psr_geometry", "cost_terrain_m", "length_xy_m", "pri", "sti",
                        "maximum_swept_footprint_slope_deg", *TIME_METRICS)
    contrasts = [dict(competitor=backend, metric=metric, selection=selection,
                     **paired_bootstrap(rows, backend, metric, selection=selection))
                 for selection in ("final_paper", "final_geometry", "first_paper")
                 for backend in BACKENDS[1:] for metric in contrast_metrics]
    budget = max((row["budget_s"] for row in rows), default=20.)
    checkpoints = sorted(set([0., .1, .25, .5, 1., 2., 5., 10., 15., budget]+
        [row["first_"+kind+"_s"] for row in rows for kind in ("geometry", "paper")
         if row["attained_"+kind] and row["first_"+kind+"_s"] is not None]))
    checkpoints = [value for value in checkpoints if value <= budget]
    attainment = attainment_rows(rows, checkpoints)
    manifest = read(root/"manifest.json")
    summary = dict(protocol_id=PROTOCOL_ID, schema_version=12, tables=tables, paired_contrasts=contrasts,
        correctness_failures=validation["correctness_failures"],
        completion="FAILED_CORRECTNESS" if validation["correctness_failures"] else
            ("COMPLETE" if all(row["execution_status"] not in UNSTARTED for row in rows) else "PARTIAL"),
        sessions=manifest.get("sessions", []), validation_cache_hits=validation.get("validation_cache_hits", 0),
        shared_sem_evaluator=True, attainment=attainment, plots=False)
    flattened = []
    for table in tables:
        record = {key:value for key, value in table.items() if not isinstance(value, dict)}
        for key, value in table.items():
            if isinstance(value, dict) and set(value) == {"n", "mean", "sd"}:
                record.update({key+"_"+suffix:number for suffix, number in value.items()})
        record["status_counts"] = table["status_counts"]
        flattened.append(record)
    trial_rows = []
    for row in rows:
        record = {key:row.get(key) for key in ("trial_id", "backend", "family", "map_id", "planner_seed", "execution_status", "native_status",
            "errors", "success_geometry", "success_paper", "attained_geometry", "attained_paper", "selected_snapshot_id", "first_paper_snapshot_id", "timing_stratum", *TIME_METRICS)}
        record.update({"final_"+key:value for key, value in final_metrics(row).items()})
        record.update({"first_paper_"+key:value for key, value in first_paper_metrics(row).items()})
        trial_rows.append(record)
    for name, records in (("aggregate.csv", flattened), ("paired_comparisons.csv", contrasts), ("trials.csv", trial_rows), ("attainment.csv", attainment)):
        _write_csv(output/name, records)
    # Plot only measured campaign rows. Parser fixtures live under tests/fixtures
    # and explicitly request no plot through their manifest label.
    if rows and not manifest.get("synthetic_fixture", False):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for axis, kind in zip(axes, ("geometry", "paper")):
            for backend in BACKENDS:
                points = [item for item in attainment if item["backend"] == backend and item["criterion"] == kind and item["fraction"] is not None]
                if points:
                    axis.step([item["elapsed_s"] for item in points], [item["fraction"] for item in points], where="post", label=LABELS[backend])
            axis.set(xlabel="Solve time (s)", title=kind, xlim=(0, budget), ylim=(0, 1))
            axis.grid(alpha=.25)
        axes[0].set_ylabel("Validated published-solution attainment / attempts")
        axes[-1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output/"attainment.png", dpi=160)
        figure.savefig(output/"attainment.pdf")
        plt.close(figure)
        summary["plots"] = True
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"
    def mean(value):
        return fmt(value["mean"])+(f" ± {fmt(value['sd'])}" if value["sd"] is not None else "")
    lines = ["# A3-T, SEM-RRT* và PUTN: benchmark v3", "",
        "Mỗi thuật toán giữ objective native; không xếp hạng tổng native cost giữa thuật toán.",
        "Bảng chính chấm incumbent native cuối trong budget, chọn trước kiểm chứng; không thay bằng nghiệm an toàn trước đó.",
        "PSR_geometry yêu cầu exact goal và oracle toàn footprint. PSR_paper thêm mọi ô đường tâm chạm có dữ liệu và SEM risk ≤40.",
        "Bộ chấm SEM chung độc lập chạy hậu kiểm; mỗi ô đóng chạm đường được tính một lần. PRI là chỉ số raster, không phải xác suất tai nạn.",
        "APL là chiều dài bề mặt; STI = tổng số turn XY >30° / tổng chiều dài bề mặt. Dấu ± là sample SD.",
        "Mọi metric chất lượng/Time trong mỗi bảng dùng cùng tập success, kèm n. Không đo metric khoảng cách nhỏ nhất tới vật cản.",
        "T_first chỉ quan sát incumbent đã xuất vượt kiểm chứng. Time là solve thực tế; preprocessing và latency được lưu riêng.",
        "Timeout/crash/interrupted nằm trong attempts; lượt chưa bắt đầu báo riêng. Timeout không chứng minh vô nghiệm.",
        "Timing đo dưới concurrency đã ghi; chỉ so timing trong cùng timing_stratum. Các baseline là tái dựng/adaptation tĩnh.",
        f"Trạng thái: {summary['completion']}; correctness failures: {summary['correctness_failures']}.", ""]
    for selection, title in (("final_paper", "Đường cuối: điều kiện paper (bảng chính)"),
                             ("final_geometry", "Đường cuối: điều kiện hình học"), ("first_paper", "Nghiệm an toàn đầu tiên (bảng phụ)")):
        lines += ["", "## "+title, "", "| Phạm vi | Nhóm | Thuật toán | Success/attempt/planned | PSR hình học cuối | PSR paper cuối | Từng có nghiệm paper | APL (m) | PRI | STI (1/m) | Time (s) | T_first hình học (s) | T_first paper (s) | Dốc quét max (°) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for item in (table for table in tables if table["selection"] == selection):
            lines.append("| "+" | ".join([item["scope"], item["group"], LABELS[item["backend"]], f"{item['success']}/{item['attempted']}/{item['planned']}",
                fmt(item["psr_geometry"]), fmt(item["psr_paper"]), fmt(item["attained_paper_rate"]), mean(item["cost_terrain_m"]), mean(item["pri"]), fmt(item["sti"]),
                mean(item["solve_time_s"]), mean(item["first_geometry_s"]), mean(item["first_paper_s"]), mean(item["maximum_swept_footprint_slope_deg"])])+" |")
    lines += ["", "## So sánh ghép cặp", "", "Hiệu A3-T − baseline; family cố định, bootstrap map → paired seed, 2.000 lần, CI 95%. Quality chỉ dùng cặp cùng success; CSV giữ coverage và số map.", "",
              "| Lựa chọn | Baseline | Metric | Cặp dùng/attempt/planned | Hiệu số | CI95 |", "|---|---|---|---|---|---|"]
    for item in contrasts:
        lines.append(f"| {item['selection']} | {item['competitor']} | {item['metric']} | {item['usable_pairs']}/{item['attempted_pairs']}/{item['planned_pairs']} | {fmt(item['mean_difference'])} | {item['ci95']} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    dump(output/"summary.json", finite_tree(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        result = validate_campaign(args.input, args.jobs)
        print(json.dumps({key:result[key] for key in ("slots", "validated_success", "correctness_failures")}))
        if result["correctness_failures"]:
            raise SystemExit(1)
    if args.output:
        analyze_campaign(args.input, args.output)
