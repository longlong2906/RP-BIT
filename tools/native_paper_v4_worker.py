"""Schema-13 trial adapter and lossless postsolve representation compaction.

Linux A3 execution replaces this process, preserving the scheduler's CPU pin
and process identity. SEM uses its unchanged native worker in the same process.
Compaction is an explicit offline operation after measured solvers have stopped.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_old_safe_sem_kaggle_v4"
BACKENDS = ("a3t_old", "a3t_safe", "sem_native")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT")
COMPACTION_VERSION = "native_paper_v4_lossless_compaction_v1"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump(path, value):
    atomic_bytes(path, _json_bytes(value))


def utc():
    return datetime.now(timezone.utc).isoformat()


def phase(directory, value):
    dump(Path(directory)/"phase.json", dict(phase=value, updated_utc=utc(), monotonic_s=time.monotonic()))


def field_bytes(field):
    """Canonical exchange bytes, tested against the frozen planner builder."""
    ny, nx = field.support.shape
    lines = ["SAFE_SEM_FIELD_V1", f"{nx} {ny} "+" ".join(format(v, ".17g") for v in
        (field.bounds[0], field.bounds[1], field.resolution, field.bounds[2], field.bounds[3]))]
    lines.extend(f"{int(valid)} {format(float(risk) if valid else 0., '.17g')}"
                 for valid, risk in zip(field.support.flat, field.risk.flat))
    return ("\n".join(lines)+"\n").encode("ascii")


def prepare_safe_field(request):
    """Rebuild on every trial; only canonical immutable bytes are shared."""
    import numpy as np
    import safe_length_field
    from native_paper_v4_protocol import field_builder_key

    began = time.perf_counter()
    policy = request["safe_length"]
    field_path = Path(request["shared_field_path"]).resolve()
    if Path(policy["field_path"]).resolve() != field_path:
        raise ValueError("shared and solver field paths differ")
    metadata_path = field_path.with_suffix(".txt.meta.json")
    if sha256(metadata_path) != request["shared_field_metadata_sha256"]:
        raise ValueError("shared field metadata changed")
    metadata = read(metadata_path)
    if field_builder_key(metadata) != request["safety_field_builder_key"]:
        raise ValueError("shared field builder key mismatch")
    if metadata["builder_source_sha256"] != sha256(safe_length_field.__file__):
        raise ValueError("planner field builder source changed")
    for name, expected in (("map_path", request["map_sha256"]), ("cloud_path", request["cloud_sha256"])):
        if sha256(request[name]) != expected:
            raise ValueError("raw field input hash mismatch: "+name)
    if metadata["cloud_sha256"] != request["cloud_sha256"] or metadata["map_sha256"] != request["map_sha256"]:
        raise ValueError("shared field input identity differs from request")
    if policy["field_sha256"] != metadata["field_sha256"] or sha256(field_path) != policy["field_sha256"]:
        raise ValueError("shared field hash mismatch")
    if policy["cloud_sha256"] != metadata["cloud_sha256"] or policy["builder_source_sha256"] != metadata["builder_source_sha256"]:
        raise ValueError("solver field identity differs from canonical field")
    field = safe_length_field.build_field(np.load(request["cloud_path"], allow_pickle=False), metadata["bounds"])
    actual = hashlib.sha256(field_bytes(field)).hexdigest()
    if actual != policy["field_sha256"]:
        raise ValueError("per-trial field rebuild differs from frozen bytes")
    # Check inputs again after numerical processing to reject concurrent changes.
    if (sha256(request["cloud_path"]) != request["cloud_sha256"] or
            sha256(request["map_path"]) != request["map_sha256"] or sha256(field_path) != actual):
        raise ValueError("field or raw input changed during rebuild")
    return dict(field_build_s=time.perf_counter()-began, field_sha256=actual,
        cloud_sha256=request["cloud_sha256"], map_sha256=request["map_sha256"],
        builder_source_sha256=metadata["builder_source_sha256"], safety_field_builder_key=request["safety_field_builder_key"],
        shared_field_path=str(field_path), shared_storage=True, rebuilt_for_this_trial=True,
        timing_scope="raw_input_verification+field_build+canonical_encoding+hash_verification_before_solve")


def validate_request(request):
    if request.get("protocol_id") != PROTOCOL_ID or request.get("campaign_schema_version") != 13:
        raise ValueError("wrong v4 protocol/schema")
    backend = request.get("backend")
    if backend not in BACKENDS:
        raise ValueError("unknown v4 backend")
    for name in ("start", "goal"):
        value = request.get(name)
        if not isinstance(value, list) or len(value) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in value):
            raise ValueError("invalid "+name)
    if type(request.get("planning_budget_s")) not in (int, float) or not math.isfinite(request["planning_budget_s"]) or request["planning_budget_s"] <= 0:
        raise ValueError("invalid planning budget")
    if type(request.get("planner_seed")) is not int or not 0 <= request["planner_seed"] < 2**32:
        raise ValueError("invalid planner seed")
    for pathkey, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if sha256(request[pathkey]) != request[hashkey]:
            raise ValueError("raw request input changed: "+pathkey)
    from native_paper_v4_protocol import OBJECTIVES, default_config
    if (request.get("track") != "native" or request.get("native_objective_id") != OBJECTIVES[backend] or
            request.get("representation_config") != default_config()["representation"] or
            request.get("clearance_metrics") is not False or request.get("cloud_phase_handshake") is not True):
        raise ValueError("frozen objective/representation changed")
    if type(request.get("work_iterations", 0)) is not int or request.get("work_iterations", 0) < 0:
        raise ValueError("invalid fixed work")
    if backend in ("a3t_old", "a3t_safe"):
        options = dict(rewire_factor=1.1, sample_attempts=100, progress_interval_s=.05,
            collision_index=True, state_cache=True, scalar_cost=True, sampler_query_cache=True,
            exact_flat_certificate=False)
        if any(request.get(key) != value for key, value in options.items()):
            raise ValueError("frozen A3 implementation options changed")
    if backend == "a3t_old":
        from a3_traversability_protocol import DEFAULT_TRAVERSABILITY
        if (request.get("planner_id") != "bitstar_a3" or request.get("objective_type") != "surface_length_traversability" or
                request.get("traversability") != DEFAULT_TRAVERSABILITY or "safe_length" in request or
                request.get("samples_per_batch") != 200 or request.get("batch_after_first_solution") != 0 or
                request.get("traversability_build_policy") != "skip_dominated"):
            raise ValueError("old A3-T configuration changed")
    elif backend == "a3t_safe":
        policy = request.get("safe_length", {})
        if (request.get("planner_id") != "bitstar_a3" or request.get("objective_type") != "surface_length" or
                "traversability" in request or request.get("samples_per_batch") != 100 or
                request.get("batch_after_first_solution") != 200 or policy.get("policy_id") != "a3t_safe_length_v1" or
                policy.get("risk_cap") != 40. or policy.get("slope_cap_deg") != 35. or policy.get("atlas_sampling") is not True):
            raise ValueError("safe A3-T configuration changed")
    elif (request.get("planner_id") != "sem_native_paper_v2" or request.get("objective_type") != "native" or
          request.get("native_objective_id") != "sem_length_turn_risk_paper_v2" or "safe_length" in request):
        raise ValueError("native SEM objective/configuration changed")


def run(request_file, output):
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    began, started = time.perf_counter(), utc()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"worker.json").exists() or (output/"preparation.json").exists() or (output/"progress.jsonl").exists():
        raise ValueError("attempt already started; no worker retry allowed")
    phase(output, "SETUP")
    request = read(request_file)
    preparation = dict(protocol_id=PROTOCOL_ID, schema_version=13, started_utc=started,
        wrapper_source_sha256=sha256(__file__), request_sha256=sha256(request_file),
        backend=request.get("backend"), field_build_s=0., status="PREPARING",
        thread_environment={name:os.environ[name] for name in THREAD_ENV})
    try:
        validate_request(request)
        if request["backend"] == "a3t_safe":
            preparation.update(prepare_safe_field(request))
        preparation.update(status="PREPARED", completed_utc=utc(), preparation_wall_s=time.perf_counter()-began)
        dump(output/"preparation.json", preparation)
        if request["backend"] == "sem_native":
            import native_paper_worker
            translated = copy.deepcopy(request)
            translated["backend"] = "sem"
            native_request = output/"native_request.json"
            dump(native_request, translated)
            return native_paper_worker.run(native_request, output)
        binary = Path(request["worker_binary"]).resolve()
        if not binary.is_file():
            raise FileNotFoundError("C++ terrain_benchmark is unavailable")
        expected = request.get("worker_binary_sha256")
        if expected is not None and sha256(binary) != expected:
            raise ValueError("C++ worker binary changed")
        command = [str(binary), str(Path(request_file).resolve()), str(output)]
        if sys.platform == "win32":
            return subprocess.call(command, env=os.environ)
        os.execvpe(command[0], command, os.environ)
        raise AssertionError("exec returned")
    except Exception as error:
        dependency = isinstance(error, (FileNotFoundError, ModuleNotFoundError))
        preparation.update(status="BLOCKED" if dependency else "FAILED", error=str(error),
                           completed_utc=utc(), preparation_wall_s=time.perf_counter()-began)
        dump(output/"preparation.json", preparation)
        dump(output/"worker.json", dict(schema_version=13, backend=request.get("backend"),
            planner_status="BLOCKED_DEPENDENCY" if dependency else "ERROR", status="BLOCKED_DEPENDENCY" if dependency else "ERROR",
            execution_status="BLOCKED" if dependency else "FAILED_CORRECTNESS", exact_goal=False,
            path_available=False, error=str(error), traceback=traceback.format_exc()))
        phase(output, "FINISHED")
        return 3 if dependency else 1


def _inside(directory, relative):
    directory = Path(directory).resolve()
    path = directory/relative
    if path.is_symlink() or directory not in path.resolve().parents:
        raise ValueError("artifact path escapes trial directory")
    return path


def _coordinates(path, width):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) not in (["x", "y", "z"][:width], ["x_m", "y_m", "z_m"][:width]):
            raise ValueError("noncanonical coordinate CSV header")
        rows = [[float(value) for value in row] for row in reader]
    if not rows or any(len(row) != width or not all(map(math.isfinite, row)) for row in rows):
        raise ValueError("invalid canonical coordinate CSV")
    return rows


def _coordinate_bytes(values, width):
    if not isinstance(values, list) or not values or any(not isinstance(row, list) or len(row) != width or
            not all(type(value) in (int, float) and math.isfinite(value) for value in row) for row in values):
        raise ValueError("invalid embedded coordinates")
    return b"".join(struct.pack("!d", float(value)) for row in values for value in row)


def _strip_event(event, directory, coordinate_hashes):
    if not isinstance(event, dict):
        raise ValueError("invalid event object")
    result, removed = dict(event), 0
    fields = (("path_xy", "path_xy.csv", 2), ("raw_path_xyz", "path_raw_xyz.csv", 3))
    if any(key in event for key, _, _ in fields):
        sid = event.get("snapshot_id")
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", ".."):
            raise ValueError("invalid snapshot id")
        for key, filename, width in fields:
            if key not in event:
                continue
            relative = "snapshots/"+sid+"/"+filename
            path = _inside(directory, relative)
            if _coordinate_bytes(event[key], width) != _coordinate_bytes(_coordinates(path, width), width):
                raise ValueError("embedded/CSV coordinate mismatch: "+relative)
            coordinate_hashes[relative] = sha256(path)
            del result[key]
            removed += 1
    return result, removed


def native_raster_identity(path, metadata, request):
    """Content identity includes array names/dtypes/shapes/bytes and model config."""
    import numpy as np
    ephemeral = {"construction_wall_s", "raster_sha256", "raster_file", "raster_artifacts"}
    configuration = {key:value for key, value in metadata.items() if key not in ephemeral}
    result = hashlib.sha256()
    result.update(json.dumps(dict(model=configuration, map_sha256=request["map_sha256"],
        cloud_sha256=request["cloud_sha256"], representation=request["representation_config"],
        native_objective_id=request["native_objective_id"]), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    arrays = []
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = archive[name]
            if value.dtype.hasobject:
                raise ValueError("object array in native raster")
            descriptor = dict(name=name, dtype=value.dtype.str, shape=list(value.shape))
            data = value.tobytes(order="C")
            result.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
            result.update(struct.pack("!Q", len(data)))
            result.update(data)
            arrays.append(dict(descriptor, sha256=hashlib.sha256(data).hexdigest()))
    return result.hexdigest(), arrays


def _copy_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or sha256(target) != expected or target.stat().st_size != source.stat().st_size:
            raise ValueError("shared native artifact content conflict")
        return
    fd, temporary = tempfile.mkstemp(prefix=target.name+".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as destination, source.open("rb") as original:
            shutil.copyfileobj(original, destination, 1024*1024)
            destination.flush()
            os.fsync(destination.fileno())
        if sha256(temporary) != expected:
            raise ValueError("copied native artifact checksum mismatch")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_compaction(directory, receipt, shared_root):
    for relative, expected in {**receipt["coordinate_sha256"], **receipt["output_sha256"]}.items():
        if sha256(_inside(directory, relative)) != expected:
            raise ValueError("compacted artifact changed: "+relative)
    reference = receipt.get("native_raster_reference")
    if reference:
        path = (Path(directory)/reference["native_raster_path"]).resolve()
        if Path(shared_root).resolve() not in path.parents or sha256(path) != reference["native_raster_sha256"]:
            raise ValueError("shared native artifact changed")


def compact_trial(directory, request, shared_root):
    """Losslessly remove duplicate encodings after all measured solves stop.

    The caller must retain the returned hashes in its post-compaction inventory.
    Scalar metrics/event order/canonical XY and raw XYZ CSVs never change. Only
    a byte-identical verified copy permits removing a per-trial native NPZ.
    """
    directory, shared_root = Path(directory).resolve(), Path(shared_root).resolve()
    receipt_path, pending_path = directory/"compaction.json", directory/"compaction.pending.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("version") != COMPACTION_VERSION or receipt.get("request_digest") != digest(request):
            raise ValueError("compaction identity changed")
        _verify_compaction(directory, receipt, shared_root)
        return receipt
    if not (directory/"worker.json").is_file():
        return dict(version=COMPACTION_VERSION, status="NOT_RUN_NO_WORKER", request_digest=digest(request))
    if (directory/"phase.json").is_file() and read(directory/"phase.json").get("phase") in ("SETUP", "SOLVING"):
        raise ValueError("refusing compaction while worker phase is active")
    pending = read(pending_path) if pending_path.exists() else None
    if pending and (pending.get("version") != COMPACTION_VERSION or pending.get("request_digest") != digest(request)):
        raise ValueError("stale compaction journal")
    coordinate_hashes = dict(pending.get("coordinate_sha256", {})) if pending else {}
    for relative, expected in coordinate_hashes.items():
        if sha256(_inside(directory, relative)) != expected:
            raise ValueError("canonical coordinates changed during compaction recovery")
    original = {}
    for name in ("worker.json", "progress.jsonl"):
        path = _inside(directory, name)
        if path.exists():
            original[name] = sha256(path)
    if pending:
        for name, checksum in original.items():
            if checksum not in (pending["input_sha256"].get(name), pending["output_sha256"].get(name)):
                raise ValueError("unexpected partial compaction bytes")
    worker = read(directory/"worker.json")
    removed = 0
    if "snapshots" in worker:
        snapshots = []
        for event in worker["snapshots"]:
            event, count = _strip_event(event, directory, coordinate_hashes)
            snapshots.append(event)
            removed += count
        worker["snapshots"] = snapshots
    outputs = {"worker.json":_json_bytes(worker)}
    progress = directory/"progress.jsonl"
    if progress.exists():
        events = []
        for line in progress.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event, count = _strip_event(json.loads(line), directory, coordinate_hashes)
            events.append(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False))
            removed += count
        outputs["progress.jsonl"] = (("\n".join(events)+"\n") if events else "").encode("utf-8")
    reference = None
    native_path = _inside(directory, "native_model/raster.npz")
    metadata_path = _inside(directory, "native_model/raster.json")
    if request["backend"] == "sem_native" and native_path.exists():
        expected = worker.get("native_model", {}).get("raster_artifacts", {}).get("raster.npz")
        if expected is None or sha256(native_path) != expected:
            raise ValueError("native raster inventory mismatch")
        original_meta = worker.get("native_model", {}).get("raster_artifacts", {}).get("raster.json")
        if original_meta is None or sha256(metadata_path) != original_meta:
            raise ValueError("native raster metadata inventory mismatch")
        metadata = read(metadata_path)
        semantic, arrays = native_raster_identity(native_path, metadata, request)
        # Prefix directories keep Windows paths below its usual 260-character
        # limit; the full content hash is the filename and full identities are
        # retained in the reference. Existing bytes must still compare exactly.
        target = shared_root/"sem_native"/request["map_sha256"][:12]/semantic[:16]/(expected+".npz")
        if shared_root not in target.resolve().parents:
            raise ValueError("shared raster path escapes shared root")
        _copy_verified(native_path, target, expected)
        reference = dict(native_raster_path=os.path.relpath(target, directory).replace(os.sep, "/"),
            native_raster_sha256=expected, native_raster_semantic_sha256=semantic,
            native_raster_size_bytes=target.stat().st_size, native_raster_arrays=arrays,
            native_metadata_path="native_model/raster.json", native_metadata_sha256=original_meta,
            identical_serialized_bytes_required=True)
    elif (directory/"artifact_refs.json").exists():
        reference = read(directory/"artifact_refs.json")
        candidate = (directory/reference["native_raster_path"]).resolve()
        if shared_root not in candidate.parents or sha256(candidate) != reference["native_raster_sha256"]:
            raise ValueError("invalid existing shared native reference")
    if reference is not None:
        outputs["artifact_refs.json"] = _json_bytes(reference)
    expected_outputs = {name:hashlib.sha256(data).hexdigest() for name, data in outputs.items()}
    receipt = dict(version=COMPACTION_VERSION, status="COMPLETE", request_digest=digest(request),
        compactor_source_sha256=sha256(__file__), input_sha256=pending["input_sha256"] if pending else original,
        output_sha256=expected_outputs, coordinate_sha256=coordinate_hashes,
        removed_coordinate_fields=pending["removed_coordinate_fields"] if pending else removed,
        native_raster_reference=reference, preserves="all_events_scalars_timestamps_and_canonical_xy_raw_xyz_csv",
        completed_utc=utc())
    dump(pending_path, receipt)
    for name, data in outputs.items():
        atomic_bytes(directory/name, data)
    if reference and native_path.exists():
        target = (directory/reference["native_raster_path"]).resolve()
        if sha256(native_path) != sha256(target) or native_path.stat().st_size != target.stat().st_size:
            raise ValueError("native byte identity lost before duplicate removal")
        # A single checked regular file inside this trial, never a recursive delete.
        _inside(directory, "native_model/raster.npz").unlink()
    _verify_compaction(directory, receipt, shared_root)
    dump(receipt_path, receipt)
    pending_path.unlink(missing_ok=True)
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: native_paper_v4_worker.py REQUEST.json OUTPUT_DIRECTORY")
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
