"""Optional smoke-only reuse of previously verified finite-graph L0 proofs.

No path, guide, mask, incumbent or trial verdict is reused.  Preparation audits
the old raw inventory and rederives the independent SEM field fingerprint.
New validators still check their own raw artifacts and every path normally.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION="smoke_independent_graph_L0_artifact_cache_v1"
ORACLE_SOURCES=("corridor_v3_validation.py","independent_verifier.py","native_paper_oracle.py",
                "native_paper_v3_sem_evaluator.py","safe_length_v2_validation.py")
GRAPH_CONTRACT=dict(grid_spacing_m=.1,endpoint_radius_m=.2,connectivity="eight_neighbors",
    endpoint_links="true_endpoints_radius_0.2_plus_direct",endpoint_numeric_tie_ulps=16,
    risk_cap=40.,slope_cap_deg=35.,cost="piecewise_affine_surface_length",
    safety="full_capsule_support_closed_risk_cells_swept_slope")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_hash(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("cache identity requires finite JSON numbers")
    return float(value).hex()


def _vector(value,size):
    if not isinstance(value,(list,tuple)) or len(value)!=size:
        raise ValueError("invalid cache identity vector")
    return [_number(item) for item in value]


def _oracle_hashes():
    return {name:_file_hash(ROOT/"tools"/name) for name in ORACLE_SOURCES}


def _runtime():
    return dict(python=sys.version,platform=platform.platform(),numpy=np.__version__)


def field_fingerprint(field):
    """Hash actual independent risk/support bytes, shape, dtype and coordinates."""
    arrays={}
    for name in ("risk","valid"):
        array=np.ascontiguousarray(getattr(field,name))
        arrays[name]=dict(dtype=array.dtype.str,shape=list(array.shape),
                          sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    return _digest(dict(bounds=_vector(field.bounds,4),resolution=_number(field.resolution),arrays=arrays))


def _input_identity(info,start,goal,claimed_upper,field_sha256):
    return dict(map_sha256=info["map_sha256"],cloud_sha256=info["cloud_sha256"],
        bounds=_vector(info["bounds"],4),start=_vector(start,2),goal=_vector(goal,2),
        claimed_upper=_number(claimed_upper),graph_contract=dict(GRAPH_CONTRACT),
        independent_field_sha256=field_sha256)


def _raw_inventory(directory):
    return {path.relative_to(directory).as_posix():_file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and
            path.name not in ("validation.json","raw_inventory.json")}


def _verify_raw_inputs(info):
    for key in ("map","cloud"):
        if _file_hash(info[key+"_path"])!=info[key+"_sha256"]:
            raise ValueError("L0 cache raw input hash mismatch: "+key)


def prepare_smoke_l0_cache(source_directory):
    """Audit an existing complete smoke and return a JSON-serializable bundle."""
    from native_paper_v3_sem_evaluator import SharedSEMField
    source=Path(source_directory).resolve()
    manifest_path=source/"manifest.json"
    manifest=_read(manifest_path)
    if manifest.get("profile")!="smoke" or manifest.get("status")!="COMPLETE":
        raise ValueError("L0 reuse requires a complete source smoke campaign")
    hashes=_oracle_hashes()
    for name,value in hashes.items():
        if manifest.get("sources",{}).get("tools/"+name)!=value:
            raise ValueError("source smoke oracle hash mismatch: "+name)
    runtime=_runtime()
    environment=manifest.get("environment",{})
    if environment.get("python")!=runtime["python"] or environment.get("platform")!=runtime["platform"]:
        raise ValueError("L0 reuse requires the same local Python/platform environment")
    maps={info["map_id"]:info for info in manifest["maps"]}
    proofs=[]
    identities=set()
    for entry in manifest["entries"]:
        # The slow independent searches already exist in unchanged v3 controls.
        # Cheap direct-chord certificates are deliberately recomputed normally.
        if entry.get("method")!="a3t_corridor_v3" or entry.get("status")!="COMPLETED":
            continue
        directory=source/"trials"/entry["trial_id"]
        validation_path=directory/"validation.json"
        if _file_hash(validation_path)!=entry.get("validation_sha256"):
            raise ValueError("source L0 validation hash mismatch")
        validation=_read(validation_path)
        checked=validation.get("corridor",{})
        certificate=checked.get("L0_certificate",{})
        if certificate.get("method")!="independent_registered_graph_Astar_XY_lower_bound" or certificate.get("cached") is not False:
            continue
        if (validation.get("success") is not True or validation.get("correctness_errors")!=[] or
                checked.get("independent_valid") is not True or checked.get("status")!="GUIDANCE_READY"):
            raise ValueError("source L0 proof lacks a valid independent verdict")
        for name,value in hashes.items():
            if validation.get("oracle_sources",{}).get(name)!=value:
                raise ValueError("source L0 verdict oracle hash mismatch: "+name)
        inventory_path=directory/"raw_inventory.json"
        inventory=_read(inventory_path)
        if _digest(inventory)!=entry.get("raw_inventory_sha256") or _raw_inventory(directory)!=inventory:
            raise ValueError("source L0 exact raw inventory mismatch")
        request=_read(directory/"request.json")
        worker=_read(directory/"worker.json")
        corridor=worker.get("corridor",{})
        policy=request.get("safe_length",{})
        if (policy.get("policy_id")!="a3t_corridor_v3" or policy.get("risk_cap")!=40. or
                policy.get("slope_cap_deg")!=35. or corridor.get("grid_spacing_m")!=.1 or
                corridor.get("endpoint_radius_m")!=.2 or corridor.get("status")!="GUIDANCE_READY"):
            raise ValueError("source L0 graph registration mismatch")
        upper=corridor.get("L0")
        length=certificate.get("length")
        _number(upper);_number(length)
        if upper<0 or length<0 or abs(upper-length)>1e-8+1e-8*max(abs(upper),abs(length)):
            raise ValueError("source L0 certificate length mismatch")
        info=maps[entry["map_id"]]
        _verify_raw_inputs(info)
        field=SharedSEMField(np.load(info["cloud_path"],allow_pickle=False),info["bounds"])
        identity=_input_identity(info,request["start"],request["goal"],upper,field_fingerprint(field))
        identity_sha256=_digest(identity)
        if identity_sha256 in identities:
            continue
        identities.add(identity_sha256)
        provenance=dict(source_trial_id=entry["trial_id"],source_map_id=entry["map_id"],
            validation_path=str(validation_path),validation_sha256=_file_hash(validation_path),
            raw_inventory_path=str(inventory_path),raw_inventory_file_sha256=_file_hash(inventory_path),
            raw_inventory_sha256=_digest(inventory),raw_inventory_verified_exact=True,
            request_sha256=_file_hash(directory/"request.json"),worker_sha256=_file_hash(directory/"worker.json"))
        proofs.append(dict(identity=identity,binding_sha256=identity_sha256,certificate=certificate,
            provenance=provenance,field_fingerprint_origin="rederived_from_source_raw_cloud_using_frozen_independent_field_source"))
    bundle=dict(schema_version=1,version=VERSION,allowed_profile="smoke",oracle_sources=hashes,
        runtime=runtime,source_manifest_path=str(manifest_path),source_manifest_sha256=_file_hash(manifest_path),
        proofs=proofs,policy="reuse_only_independent_L0_not_path_verdict_or_guide",
        claimed_upper_policy="exact_binary64_cache_key; mismatch_runs_fresh_search")
    validate_smoke_l0_cache(bundle)
    return bundle


def validate_smoke_l0_cache(bundle, expected_sha256=None):
    """Read-only provenance validation; expected hash is the canonical digest."""
    if expected_sha256 is not None and _digest(bundle)!=expected_sha256:
        raise ValueError("L0 bundle content hash mismatch")
    if bundle.get("schema_version")!=1 or bundle.get("version")!=VERSION or bundle.get("allowed_profile")!="smoke":
        raise ValueError("unknown or non-smoke L0 bundle")
    if bundle.get("oracle_sources")!=_oracle_hashes() or bundle.get("runtime")!=_runtime():
        raise ValueError("L0 bundle oracle source or runtime changed")
    if _file_hash(bundle["source_manifest_path"])!=bundle.get("source_manifest_sha256"):
        raise ValueError("source smoke manifest changed")
    source_manifest=_read(bundle["source_manifest_path"])
    source_maps={info["map_id"]:info for info in source_manifest["maps"]}
    for proof in bundle["proofs"]:
        if _digest(proof["identity"])!=proof.get("binding_sha256") or proof["identity"].get("graph_contract")!=GRAPH_CONTRACT:
            raise ValueError("L0 binding or graph contract mismatch")
        origin=proof["provenance"]
        if _file_hash(origin["validation_path"])!=origin["validation_sha256"]:
            raise ValueError("source L0 validation changed")
        if _file_hash(origin["raw_inventory_path"])!=origin["raw_inventory_file_sha256"]:
            raise ValueError("source L0 raw inventory receipt changed")
        inventory=_read(origin["raw_inventory_path"])
        if _digest(inventory)!=origin["raw_inventory_sha256"]:
            raise ValueError("source L0 raw inventory digest changed")
        directory=Path(origin["raw_inventory_path"]).parent
        if _raw_inventory(directory)!=inventory:
            raise ValueError("source L0 raw evidence changed")
        for name in ("request","worker"):
            if _file_hash(directory/(name+".json"))!=origin[name+"_sha256"]:
                raise ValueError("source L0 "+name+" changed")
        validation=_read(origin["validation_path"])
        if proof["certificate"]!=validation.get("corridor",{}).get("L0_certificate"):
            raise ValueError("cached certificate differs from verified source")
        request,worker=_read(directory/"request.json"),_read(directory/"worker.json")
        identity=proof["identity"]
        expected=_input_identity(source_maps[origin["source_map_id"]],request["start"],request["goal"],
                                 worker["corridor"]["L0"],identity["independent_field_sha256"])
        if expected!=identity:
            raise ValueError("cached identity differs from verified source")
    return dict(valid=True,bundle_sha256=_digest(bundle),proof_count=len(bundle["proofs"]))


def seed_map_l0_cache(bundle, info, oracle, field, *, profile="smoke", expected_sha256=None):
    """Seed exact existing oracle cache keys; mismatched inputs stay cache misses."""
    if profile!="smoke":
        raise ValueError("artifact L0 reuse is allowed only for smoke")
    began=time.perf_counter()
    verified=validate_smoke_l0_cache(bundle,expected_sha256)
    _verify_raw_inputs(info)
    if getattr(oracle,"map_sha256",None)!=info["map_sha256"]:
        raise ValueError("oracle is bound to a different terrain map")
    fingerprint=field_fingerprint(field)
    cache=getattr(oracle,"_corridor_shortest_cache",{})
    seeded=[]
    for proof in bundle["proofs"]:
        identity=proof["identity"]
        upper=float.fromhex(identity["claimed_upper"])
        actual=_input_identity(info,info["start"],info["goal"],upper,fingerprint)
        if actual!=identity:
            continue
        key=(id(field),tuple(info["start"]),tuple(info["goal"]),upper)
        if key in cache:
            continue
        certificate=dict(proof["certificate"],cached=True,
            smoke_reuse_binding_sha256=proof["binding_sha256"],smoke_reuse_source=proof["provenance"],
            smoke_reuse_bundle_sha256=verified["bundle_sha256"],search_executed_in_this_process=False,
            source_search_validation_wall_s=proof["certificate"].get("validation_wall_s"))
        cache[key]=certificate
        seeded.append(dict(binding_sha256=proof["binding_sha256"],claimed_upper=upper,
                           source_trial_id=proof["provenance"]["source_trial_id"]))
    oracle._corridor_shortest_cache=cache
    return dict(version=VERSION,bundle_sha256=verified["bundle_sha256"],map_id=info.get("map_id"),
        independent_field_sha256=fingerprint,seeded_certificates=seeded,preload_only_not_a_verdict=True,
        reuse_check_wall_s=time.perf_counter()-began,
        actual_use_marker="row.corridor.L0_certificate.smoke_reuse_binding_sha256")
